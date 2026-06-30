/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <supa_runtime.h>
#include <sys/types.h>
#include <torch/extension.h>
#include <iostream>
#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include "torch_supa/csrc/supa/SUPAPluggableAllocator.h"

extern "C" {
using c10::supa::SUPACachingAllocator::DeviceStats;
static bool useflag = false;

C10_EXPORT void* my_malloc(size_t size, int device, void* stream) {
  void* ptr = nullptr;
  supaMalloc(&ptr, size);
  std::cout << "alloc ptr = " << ptr << ", size = " << size << std::endl;
  useflag = true;
  return ptr;
}

C10_EXPORT void my_free(void* ptr, size_t size, int device, void* stream) {
  std::cout << "free ptr = " << ptr << std::endl;
  supaFree(ptr);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_malloc", &my_malloc, "");
  m.def("my_free", &my_free, "");
}
