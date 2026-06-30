/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"

namespace at::supa {

using namespace at::native;

std::tuple<Tensor, Tensor> SUPANativeFunctions::mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return SUPANativeFunctions::mode_out(self, dim, keepdim, values, indices);
}

std::tuple<Tensor&, Tensor&> SUPANativeFunctions::mode_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  TORCH_CHECK(self.layout() == Layout::Strided, "mode only supports strided layout, got: ", self.layout());
  TORCH_CHECK(
      self.device() == values.device(),
      "expected device '",
      self.device(),
      "' but got '",
      values.device(),
      "' for values output");
  TORCH_CHECK(
      self.device() == indices.device(),
      "expected device '",
      self.device(),
      "' but got '",
      indices.device(),
      "' for indices output");
  TORCH_CHECK(
      self.scalar_type() == values.scalar_type(),
      "expected scalar type '",
      self.scalar_type(),
      "' but got '",
      values.scalar_type(),
      "' for values output");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "expected scalar type '",
      ScalarType::Long,
      "' but got '",
      indices.scalar_type(),
      "' for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    resize_output(values, sizes);
    resize_output(indices, sizes);
    return std::tie(values, indices);
  }

  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  }

  auto result = [&]() {
    NoNamesGuard guard;
    mode_stub(self.device().type(), values, indices, self, dim, keepdim);
    return std::tuple<Tensor&, Tensor&>{values, indices};
  }();
  namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
  namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
  return result;
}

} // namespace at::supa