/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <supa.h>
#include <supa_runtime.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace torch_supa::supa {

TORCH_SUPA_API void initSupartBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();

  auto supart = m.def_submodule("_supart", "libsupa-runtime.so bindings");

  py::enum_<supaError_t>(supart, "supa"
                                 "Error")
      .value("success", supaSuccess);

  supart.def("supa"
             "GetErrorString",
             supaGetErrorString);
  supart.def("supa"
             "ProfilerStart",
             supaProfilerStart);
  supart.def("supa"
             "ProfilerStop",
             supaProfilerStop);
  supart.def("supa"
             "HostRegister",
             [](uintptr_t ptr, size_t size, unsigned int flags) -> supaError_t {
               py::gil_scoped_release no_gil;
               return C10_SUPA_ERROR_HANDLED(
                   // NOLINTNEXTLINE(performance-no-int-to-ptr)
                   supaHostRegister((void *)ptr, size, flags));
             });
  supart.def("supa"
             "HostUnregister",
             [](uintptr_t ptr) -> supaError_t {
               py::gil_scoped_release no_gil;
               // NOLINTNEXTLINE(performance-no-int-to-ptr)
               return C10_SUPA_ERROR_HANDLED(supaHostUnregister((void *)ptr));
             });
  supart.def("supa"
             "StreamCreate",
             [](uintptr_t ptr) -> supaError_t {
               py::gil_scoped_release no_gil;
               // NOLINTNEXTLINE(performance-no-int-to-ptr)
               return C10_SUPA_ERROR_HANDLED(
                   supaStreamCreate((supaStream_t *)ptr));
             });
  supart.def("supa"
             "StreamDestroy",
             [](uintptr_t ptr) -> supaError_t {
               py::gil_scoped_release no_gil;
               // NOLINTNEXTLINE(performance-no-int-to-ptr)
               return C10_SUPA_ERROR_HANDLED(
                   supaStreamDestroy((supaStream_t)ptr));
             });

  supart.def("supa"
             "MemGetInfo",
             [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
               c10::supa::SUPAGuard guard(device);
               size_t device_free = 0;
               size_t device_total = 0;
               py::gil_scoped_release no_gil;
               C10_SUPA_CHECK(supaMemGetInfo(&device_free, &device_total));
               return {device_free, device_total};
             });
}

} // namespace torch_supa::supa
