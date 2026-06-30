/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/aten/ops/ReduceOpsUtils.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/ReduceOps.h>

namespace at::supa {

SUPA_IMPL_FUNC(linalg_vector_norm)
(const Tensor& self,
 const Scalar& scalar_ord,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  auto ord = scalar_ord.toDouble();
  auto dim = opt_dim.value_or(IntArrayRef{});
  auto size = self.sizes();
  auto ndim = self.dim();

  auto opt_dim_ = dim.vec();
  maybe_wrap_dims(opt_dim_, ndim);

  using Int = IntArrayRef::value_type;
  std::vector<Int> all_dim(ndim);
  std::iota(all_dim.begin(), all_dim.end(), 0);

  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
  auto reduce_dim = is_all_reduce ? all_dim : opt_dim_;

  bool is_reduce_over_1D_vector = true;
  for (auto i : reduce_dim) {
    if (size[i] != 1) {
      is_reduce_over_1D_vector = false;
      break;
    }
  }

  if (is_reduce_over_1D_vector) {
    Tensor self_;
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype);
    } else {
      self_ = self;
    }
    if (ord != 0.0) {
      keepdim ? at::abs_outf(self_, const_cast<Tensor&>(result))
              : at::abs_outf(self_.squeeze(reduce_dim), const_cast<Tensor&>(result));
    } else {
      keepdim ? at::ne_outf(self_, 0, const_cast<Tensor&>(result))
              : at::ne_outf(self_.squeeze(reduce_dim), 0, const_cast<Tensor&>(result));
    }
    return;
  }

  Tensor self_;
  if (self.is_cpu() && self.is_complex() && std::abs(ord) == INFINITY) {
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype).abs();
    } else {
      self_ = self.abs();
    }
  } else {
    self_ = self;
  }

  auto iter = make_reduction("vector_norm", const_cast<Tensor&>(result), self_, dim, keepdim, result.scalar_type());
  at::native::norm_stub(iter.device_type(), iter, ord);
}

} // namespace at::supa
