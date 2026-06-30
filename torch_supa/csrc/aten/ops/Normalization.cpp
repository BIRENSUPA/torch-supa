/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Normalization.h>
#include <torch/csrc/autograd/custom_function.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"

namespace at::supa {

using namespace at::native;

// Copy from torch/csrc/autograd/FuncitonsManual.cpp
// May write more efficient ops later?
Tensor unsqueeze_multiple(const Tensor& t, OptionalIntArrayRef opt_dim, size_t n_dims) {
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    auto dim_size = dim.size();
    // Optimisation for two common cases
    if (dim_size == 0) {
      return t;
    }
    if (dim_size == 1) {
      return t.unsqueeze(dim[0]);
    }
  }
  auto dims_to_unsqueeze = at::dim_list_to_bitset(opt_dim, n_dims);
  Tensor res = t;
  for (const auto i : c10::irange(n_dims)) {
    if (dims_to_unsqueeze[i]) {
      res = res.unsqueeze(static_cast<int64_t>(i));
    }
  }
  return res;
}

Tensor norm_backward(
    Tensor grad,
    const Tensor& self,
    const c10::optional<Scalar>& p_,
    Tensor norm,
    IntArrayRef dim,
    bool keepdim) {
  // NB: We mask fill the NaNs in the output to be zero but still do float
  // division
  //     by zero, which ASAN complains about. One way to appease ASAN is to fill
  //     the problematic values with something arbitrary before the division,
  //     but we decide not to due to the perf hit. Instead we just silence ASAN
  //     where necessary
  size_t ndim = self.dim();
  double p = p_.value_or(2.0).toDouble();
  Tensor self_scaled;
  Tensor scale_v;

  if (!keepdim && self.dim() != 0) {
    grad = unsqueeze_multiple(grad, dim, ndim);
    norm = unsqueeze_multiple(norm, dim, ndim);
  }

  if (p == 0.0) {
    return {};
  }
  if (p == 1.0) {
    return self.sgn() * grad;
  }
  if (p == 2.0) {
    return grad * (self / norm).masked_fill_(norm == 0, 0);
  }
  if (std::isinf(p)) {
    // Derivative of amax(abs(self), dim, keepdim) but respecting nans
    // We create a mask of `argmax`: it's argmax if self.abs() == norm or it's
    // NaN
    auto self_abs = self.abs();
    auto mask = self_abs.eq(norm).logical_or(self_abs.isnan());
    return self.sgn() * ((grad / mask.sum(dim, true)) * mask);
  }
  if (p < 1.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1).masked_fill_(self == 0, 0);
    return self_scaled * grad * norm.pow(1 - p);
  }
  if (p < 2.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1);
    scale_v = grad / norm.pow(p - 1);
    scale_v.masked_fill_(norm == 0, 0);
    return self_scaled * scale_v;
  }

  self_scaled = self * self.abs().pow_(p - 2);
  scale_v = grad / norm.pow(p - 1);
  scale_v.masked_fill_(norm == 0, 0);
  return self_scaled * scale_v;
}

Tensor& supa_renorm_out(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, static_cast<int64_t>(self_sizes.size()));

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // For half, calculate norm in float precision then cast
  // normalization factor to half
  auto dtype = self.scalar_type();
  auto acc_type = dtype;
  if (acc_type == c10::ScalarType::BFloat16 || acc_type == c10::ScalarType::Half) {
    acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
  }
  auto norm = at::linalg_vector_norm(
      self,
      p.toDouble(),
      reduce_dims,
      /*keepdim=*/true,
      /*dtype=*/acc_type);

  auto factor = (acc_type == c10::toRealValueType(dtype)) ? norm : at::empty(norm.sizes(), self.options());
  auto iter = TensorIteratorConfig()
                  .add_output(factor)
                  .add_input(norm)
                  .set_check_mem_overlap(false)
                  .cast_common_dtype_to_outputs(true)
                  .build();

  renorm_scale_factor_stub(iter.device_type(), iter, maxnorm.toDouble());
  at::mul_outf(self, factor, const_cast<at::Tensor&>(out));
  return const_cast<at::Tensor&>(out);
}

class RenormFunction : public torch::autograd::Function<RenormFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& self,
      const Scalar& p,
      int64_t dim,
      const Scalar& maxnorm) {
    auto out = at::empty_like(self);
    supa_renorm_out(self, p, dim, maxnorm, out);
    auto self_sizes = self.sizes();
    dim = c10::maybe_wrap_dim(dim, static_cast<int64_t>(self_sizes.size()));
    // save for backward
    ctx->save_for_backward({self});
    ctx->saved_data["p"] = p.toDouble();
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["maxnorm"] = maxnorm.toDouble();

    return out;
  }

  static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx, std::vector<at::Tensor> grad_outputs) {
    auto grad = grad_outputs[0];

    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    auto p = ctx->saved_data["p"].toScalar();
    auto dim = ctx->saved_data["dim"].toInt();
    auto maxnorm = ctx->saved_data["maxnorm"].toScalar();

    auto n = self.dim();
    dim = c10::maybe_wrap_dim(dim, n);
    auto reduce_dims = at::DimVector(n);
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
    reduce_dims.erase(reduce_dims.begin() + dim);

    auto dtype = self.scalar_type();
    auto acc_type = dtype;
    if (acc_type == c10::ScalarType::BFloat16 || acc_type == c10::ScalarType::Half) {
      acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
    }

    auto norm = at::linalg_vector_norm(
        self,
        p,
        reduce_dims,
        /*keepdim=*/true,
        /*dtype=*/acc_type);

    const auto real_acc_type = c10::toRealValueType(acc_type);
    auto grad_output = (self.conj() * grad);
    // vector_norm output is real, so grad_output must also be real
    if (real_acc_type != acc_type) {
      grad_output = at::real(grad_output);
    }
    grad_output = grad_output.sum(reduce_dims, /*keepdim=*/true, /*dtype=*/real_acc_type);
    auto nb = norm_backward(
        std::move(grad_output),
        self,
        p,
        norm,
        reduce_dims,
        /*keepdim=*/true);

    auto invnorm = (norm + 1e-7).reciprocal();
    auto grad_norm = maxnorm * invnorm * (grad - invnorm * nb);
    auto grad_self = at::where(norm > maxnorm, grad_norm.to(grad.scalar_type()), grad);

    return {grad_self, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

Tensor SUPANativeFunctions::renorm(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
  return RenormFunction::apply(self, p, dim, maxnorm);
}

SUPA_IMPL_FUNC(renorm)(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  supa_renorm_out(self, p, dim, maxnorm, out);
}

Tensor& SUPANativeFunctions::renorm_(Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
  auto result = RenormFunction::apply(self, p, dim, maxnorm);
  self.copy_(result);
  return self;
}

} // namespace at::supa