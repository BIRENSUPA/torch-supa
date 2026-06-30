/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace at::native {
namespace {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T bce_logits_loss(T input, T target) {
  const T max_val = input < T(0) ? -input : T(0);
  return (T(1) - target) * input +
      max_val + c10::cuda::compat::log(
                    c10::cuda::compat::exp(-max_val) +
                    c10::cuda::compat::exp(-input - max_val));
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T
bce_logits_loss(T input, T target, T pos_weight) {
  const T max_val = input < T(0) ? -input : T(0);
  const T log_weight = (pos_weight - T(1)) * target + T(1);
  return (T(1) - target) * input +
      log_weight *
      (max_val + c10::cuda::compat::log(
                     c10::cuda::compat::exp(-max_val) +
                     c10::cuda::compat::exp(-input - max_val)));
}

template <typename scalar_t>
void bce_logits_no_weight_kernel(TensorIteratorBase& iter) {
  using opmath_t = at::opmath_type<scalar_t>;
  gpu_kernel(iter, [] GPU_LAMBDA(scalar_t input, scalar_t target) -> scalar_t {
    return static_cast<scalar_t>(bce_logits_loss(
        static_cast<opmath_t>(input), static_cast<opmath_t>(target)));
  });
}

template <typename scalar_t>
void bce_logits_weight_kernel(TensorIteratorBase& iter) {
  using opmath_t = at::opmath_type<scalar_t>;
  gpu_kernel(
      iter,
      [] GPU_LAMBDA(scalar_t input, scalar_t target, scalar_t weight)
          -> scalar_t {
        const auto loss = bce_logits_loss(
            static_cast<opmath_t>(input), static_cast<opmath_t>(target));
        return static_cast<scalar_t>(loss * static_cast<opmath_t>(weight));
      });
}

template <typename scalar_t>
void bce_logits_pos_weight_kernel(TensorIteratorBase& iter) {
  using opmath_t = at::opmath_type<scalar_t>;
  gpu_kernel(
      iter,
      [] GPU_LAMBDA(scalar_t input, scalar_t target, scalar_t pos_weight)
          -> scalar_t {
        return static_cast<scalar_t>(bce_logits_loss(
            static_cast<opmath_t>(input),
            static_cast<opmath_t>(target),
            static_cast<opmath_t>(pos_weight)));
      });
}

template <typename scalar_t>
void bce_logits_pos_weight_weight_kernel(TensorIteratorBase& iter) {
  using opmath_t = at::opmath_type<scalar_t>;
  gpu_kernel(
      iter,
      [] GPU_LAMBDA(
          scalar_t input,
          scalar_t target,
          scalar_t pos_weight,
          scalar_t weight) -> scalar_t {
        const auto loss = bce_logits_loss(
            static_cast<opmath_t>(input),
            static_cast<opmath_t>(target),
            static_cast<opmath_t>(pos_weight));
        return static_cast<scalar_t>(loss * static_cast<opmath_t>(weight));
      });
}

} // namespace

TORCH_SUPA_API void binary_cross_entropy_with_logits_kernel_supa(
    TensorIteratorBase& iter,
    bool has_pos_weight,
    bool has_weight) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      iter.common_dtype(),
      "binary_cross_entropy_with_logits_supa",
      [&]() {
        if (has_pos_weight && has_weight) {
          bce_logits_pos_weight_weight_kernel<scalar_t>(iter);
        } else if (has_pos_weight) {
          bce_logits_pos_weight_kernel<scalar_t>(iter);
        } else if (has_weight) {
          bce_logits_weight_kernel<scalar_t>(iter);
        } else {
          bce_logits_no_weight_kernel<scalar_t>(iter);
        }
      });
}

} // namespace at::native
