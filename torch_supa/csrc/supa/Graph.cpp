/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_supa/csrc/core/supa/SUPAGraph.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THBPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_supa_m = py::handle(module).cast<py::module>();

  torch_supa_m.def("_graph_pool_handle", &::at::supa::graph_pool_handle);

  shared_ptr_class_<::at::supa::SUPAGraph>(torch_supa_m, "_SUPAGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](::at::supa::SUPAGraph& self,
             std::optional<c10::supa::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            supaStreamCaptureMode capture_mode{};
            c10::supa::MempoolId_t pool = pool_opt.has_value() ? pool_opt.value() : c10::supa::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = supaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = supaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = supaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, "
                  "`thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def("capture_end", torch::wrap_pybind_function_no_gil(&at::supa::SUPAGraph::capture_end))
#if TORCH_VER >= TORCH_2_4_0
      .def(
          "register_generator_state",
          [](::at::supa::SUPAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
#endif
      .def("replay", torch::wrap_pybind_function_no_gil(&at::supa::SUPAGraph::replay))
      .def("reset", torch::wrap_pybind_function_no_gil(&at::supa::SUPAGraph::reset))
      .def("pool", torch::wrap_pybind_function_no_gil(&at::supa::SUPAGraph::pool))
      .def("debug_dump", torch::wrap_pybind_function_no_gil(&::at::supa::SUPAGraph::debug_dump))
      .def("enable_debug_mode", torch::wrap_pybind_function_no_gil(&::at::supa::SUPAGraph::enable_debug_mode))
      .def("debug_dump", torch::wrap_pybind_function_no_gil(&::at::supa::SUPAGraph::debug_dump), py::arg("debug_path"));
}