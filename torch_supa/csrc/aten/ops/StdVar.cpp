/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/TensorOptions.h>
#include <limits>
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"

namespace at::supa {

using namespace at::native;
namespace {
inline void warn_invalid_degrees_of_freedom(const char* fname, const TensorIterator& iter, double correction) {
  int64_t reducing_over_num_elements = iter.num_output_elements() == 0 ? 0 : iter.numel() / iter.num_output_elements();
  if (static_cast<double>(reducing_over_num_elements) - correction <= 0) {
    TORCH_WARN(
        fname,
        "(): degrees of freedom is <= 0. Correction should be strictly less "
        "than the reduction factor (input numel divided by output numel).");
  }
}
} // namespace

static TensorOptions options_to_value_type(TensorOptions opts) {
  auto scalar_type = typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

static Tensor& std_var_out(
    const char* fname,
    Tensor& result,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(self.layout() == Layout::Strided, "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "std and var only support floating point and complex dtypes");

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result, {});
  auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);
  TORCH_CHECK(
      at::canCast(self.scalar_type(), result.scalar_type()),
      "result type ",
      self.scalar_type(),
      " can't be cast to the "
      "desired output type ",
      result.scalar_type());
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } // need defined same cpu macros with pytorch
  std_var_stub(iter.device_type(), iter, correction, take_sqrt);

  return result;
}

static std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.layout() == Layout::Strided, fname, " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
      fname,
      " only support floating point and complex dtypes");
  TORCH_CHECK(
      result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
      fname,
      " expected result1 to be real and match the precision of result2. Got ",
      result1.scalar_type(),
      " and ",
      result2.scalar_type(),
      ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components separately then combine as:
    // variance = var_real + var_imag
    // mean = mean_real + j * mean_imag
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result1, {}));
    Tensor real_in = at::real(self);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        real_out_var,
        real_out_mean,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        imag_out_var,
        imag_out_mean,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result1, real_out_var, imag_out_var);
    if (take_sqrt) {
      at::sqrt_out(result1, result1);
    }
    at::complex_out(result2, real_out_mean, imag_out_mean);
    return std::tuple<Tensor&, Tensor&>(result1, result2);
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result1, {});
  auto iter = make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result1.fill_(std::numeric_limits<double>::quiet_NaN());
    result2.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    std_var_stub(iter.device_type(), iter, correction, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

Tensor SUPANativeFunctions::std(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& SUPANativeFunctions::std_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& SUPANativeFunctions::var_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor SUPANativeFunctions::var(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> SUPANativeFunctions::std_mean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out("std_mean", result1, result2, self, dim, correction, keepdim, true);
}

std::tuple<Tensor&, Tensor&> SUPANativeFunctions::std_mean_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result1,
    Tensor& result2) {
  return std_var_mean_out("std_mean", result1, result2, self, dim, correction, keepdim, true);
}

std::tuple<Tensor&, Tensor&> SUPANativeFunctions::var_mean_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result1,
    Tensor& result2) {
  return std_var_mean_out("var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> SUPANativeFunctions::var_mean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out("var_mean", result1, result2, self, dim, correction, keepdim, false);
}

} // namespace at::supa
