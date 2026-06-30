/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void add(const float* A, const float* B, float* C, int64_t N) {
  int i = threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
} // namespace

namespace cuda_kernel {

void vector_add(const float* a, const float* b, float* c, int64_t n) {
  add<<<1, n>>>(a, b, c, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cuda_kernel
