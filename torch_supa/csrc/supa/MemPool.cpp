/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_supa/csrc/core/supa/MemPool.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/utils/LazyInit.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

TORCH_SUPA_API void THBPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::supa::MemPool>(torch_C_m, "_MemPool")
      .def(py::init([](std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> allocator,
                       bool is_user_created,
                       bool use_on_oom,
                       bool no_split) {
        torch_supa::utils::supa_lazy_init();
        return std::make_shared<::c10::supa::MemPool>(std::move(allocator), is_user_created, use_on_oom, no_split);
      }))
      .def_property_readonly("id", &::c10::supa::MemPool::id)
      .def_property_readonly("allocator", &::c10::supa::MemPool::allocator)
      .def("use_count", &::c10::supa::MemPool::use_count);
  shared_ptr_class_<::c10::supa::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::supa::MemPool*>())
      .def_static("active_pool", &::c10::supa::MemPoolContext::getActiveMemPool);
}
