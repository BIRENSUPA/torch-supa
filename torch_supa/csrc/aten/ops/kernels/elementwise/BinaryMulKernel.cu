/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"
#include <ATen/native/TensorIterator.h>
#include "torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h"
#include <torch_supa/csrc/core/supa/SUPAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

namespace {
constexpr char mul_name[] = "mul_kernel";
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, binary_internal::MulFunctor<opmath_t>());
        });
  }
}
}

REGISTER_PRIVATEUSE1_DISPATCH(mul_stub, &mul_kernel_cuda)
} // namespace at::native
