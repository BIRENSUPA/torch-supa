/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>

#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {
namespace {

void polar_kernel_cuda(TensorIterator& iter) {
  // To optimize polar kernel, we change the elements per thread to 8
  constexpr int polar_tuned_elems_per_thread = 8;
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar_cuda", [&]() {
    gpu_kernel(
      iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
      }, polar_tuned_elems_per_thread);
  });
}

} // anonymous namespace

REGISTER_PRIVATEUSE1_DISPATCH(polar_stub, &polar_kernel_cuda)

} // namespace at::native
