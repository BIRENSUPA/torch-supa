/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/extension.h>

#ifdef _SUPA_CUDA_
namespace cuda_kernel {
void vector_add(const float* a, const float* b, float* c, int64_t n);
}
#define kernel cuda_kernel
#else

#include <supa_runtime.h>
namespace supa_kernel {
supaError_t vector_add(const float* a, const float* b, float* c, int64_t n);
}
#define kernel supa_kernel
#endif

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_privateuseone(), "must be supa tensor");
  TORCH_CHECK(b.device().is_privateuseone(), "must be supa tensor");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same size");

  auto n = a.numel();
  auto c = torch::empty_like(a);
  kernel::vector_add(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "Vector addition extension (SUPA)");
}
