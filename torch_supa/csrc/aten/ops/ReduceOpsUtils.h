/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>

namespace at::supa {

using DimMask = TensorIterator::DimMask;

inline DimMask make_dim_mask(OptionalIntArrayRef opt_dims, int64_t ndim, bool allow_empty_dims = false) {
  DimMask mask;
  if (opt_dims.has_value()) {
    auto dims = opt_dims.value();
    if (dims.empty() && !allow_empty_dims) {
      mask = DimMask().flip();
    } else {
      mask = at::dim_list_to_bitset(dims, ndim);
    }
  } else {
    mask = DimMask().flip();
  }
  return mask;
}

inline DimVector shape_from_dim_mask(const Tensor& self, DimMask mask, bool keepdim) {
  auto shape = DimVector(self.sizes());
  for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  return shape;
}

inline void resize_reduction_result(
    Tensor& result,
    const Tensor& self,
    DimMask mask,
    bool keepdim,
    ScalarType /*dtype*/) {
  auto shape = shape_from_dim_mask(self, mask, keepdim);
  TORCH_CHECK(
      result.defined(),
      "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
  at::native::resize_output(result, shape);
}

inline Tensor review_reduce_result(const Tensor& result, int64_t ndim, DimMask mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (const auto dim : c10::irange(ndim)) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

inline TensorIterator make_reduction(
    const char* name,
    Tensor& result,
    const Tensor& self,
    OptionalIntArrayRef dim_opt,
    bool keepdim,
    ScalarType in_dtype,
    ScalarType out_dtype) {
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == out_dtype,
      name,
      ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(out_dtype),
      ".");
  IntArrayRef dim = dim_opt.value_or(IntArrayRef{});
  int64_t ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  resize_reduction_result(result, self, mask, keepdim, out_dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

inline TensorIterator make_reduction(
    const char* name,
    Tensor& result,
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    ScalarType out_dtype) {
  const bool gpu_lowp_to_f32 = (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) && out_dtype == kFloat;
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : self.is_complex() ? c10::toComplexType(out_dtype) : out_dtype;
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
}

} // namespace at::supa
