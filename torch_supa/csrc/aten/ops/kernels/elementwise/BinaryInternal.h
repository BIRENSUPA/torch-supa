/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

// DON'T include this except from Binary*.cu files. It should not leak into
// headers.
#pragma once

namespace at::native::binary_internal {

template <typename scalar_t>
struct DivFunctor {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template <typename T>
struct MulFunctor {
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead
// [-Werror=int-in-bool-context]
template <>
struct MulFunctor<bool> {
  __device__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};
void div_true_kernel_cuda(TensorIteratorBase& iter);
void div_trunc_kernel_cuda(TensorIteratorBase& iter);
} // namespace at::native::binary_internal
