/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"

namespace at::native {

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (const auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

static bool all_strides_match(std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (const auto& tensor : tensors) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  TORCH_CHECK(
      is_expandable_to(value.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      value.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      value.scalar_type(),
      " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (const auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list) {
  const int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0;
  int64_t dims_after = 0;
  int64_t dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      const auto src_dim = static_cast<int64_t>(dim);
      indexed_sizes.push_back(src.size(src_dim));
      indexed_strides.push_back(src.stride(src_dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (const auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA/MPS/XPU tensors, force all index tensors to have the same striding to
  // simplify the CUDA/MPS/XPU kernel.
  if (indices.size() >= 2) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}
} // namespace at::native

namespace at::supa {

using namespace at::native;

Tensor& SUPANativeFunctions::_index_put_impl_(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  if ((accumulate || globalContext().deterministicAlgorithms())) {
    TORCH_CHECK(
        value_.device() == self.device(),
        "expected device ",
        self.device(),
        " but got device ",
        value_.device(),
        " for value tensor");
    index_put_with_sort_stub(self.device().type(), self, indices, value_, accumulate, unsafe);
    return self;
  }
  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}

Tensor& quantized_privateuse1_index_put_impl_(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  TORCH_CHECK_INDEX(
      indices.size() <= static_cast<size_t>(self.dim()),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  TORCH_CHECK(!value.is_quantized(), "Value argument for quantized input_put should not be quantized");
  TORCH_CHECK(
      self.qscheme() == c10::kPerTensorAffine,
      "index_put for quantized tensors is currently only supported for per tensor quantized tensors");
  TORCH_CHECK(!accumulate, "index_put for quantized tensors is currently only supported for accumulate=False");

  auto dequantized = self.dequantize().cpu();
  auto value_ = value;
  if (!value_.is_cpu()) {
    value_ = value_.cpu();
  }
  torch::List<c10::optional<Tensor>> cpu_indices;
  cpu_indices.reserve(indices.size());
  for (const auto& index : indices) {
    const c10::optional<Tensor> index_opt = index;
    if (index_opt.has_value()) {
      cpu_indices.push_back(index_opt->cpu());
    } else {
      cpu_indices.push_back(c10::nullopt);
    }
  }

  at::native::_index_put_impl_(dequantized, cpu_indices, value_, accumulate, unsafe);
  auto requantized = at::quantize_per_tensor(dequantized, self.q_scale(), self.q_zero_point(), self.scalar_type());
  self.copy_(requantized);
  return self;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_index_put_impl_", TORCH_FN(quantized_privateuse1_index_put_impl_));
}

SUPA_IMPL_FUNC(index_Tensor)
(const Tensor& self, DimVector sizes, DimVector strides, const Tensor& result) {
  const auto num_indices = sizes.size();
  if (num_indices >= 2) {
    std::vector<at::Tensor> indices;
    indices.reserve(num_indices);
    for (int i = 2; i < this->ntensors(); ++i) {
      indices.emplace_back(this->tensor(i));
    }
    if (!all_strides_match(indices)) {
      for (int i = 0; i < num_indices; ++i) {
        if (!indices[i].is_contiguous()) {
          indices[i] = indices[i].contiguous();
          this->_unsafe_set_arg_data(i + 2, indices[i].data_ptr());
        }
      }
    }
  }
  index_stub(device_type(), *this, sizes, strides);
}

} // namespace at::supa
