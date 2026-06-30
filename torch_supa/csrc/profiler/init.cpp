/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>

#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>

#include "torch_supa/csrc/profiler/collection.h"
#include "torch_supa/csrc/profiler/kineto_shim.h"
#include "torch_supa/csrc/profiler/profiler_kineto.h"
#include "torch_supa/csrc/profiler/profiler_python.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

using namespace torch_supa::profiler::impl;

namespace torch_supa {
namespace profiler {

PyObject* profiler_initExtension(PyObject* _unused, PyObject* unused) {
  auto torch_supa_C_module = THPObjectPtr(PyImport_ImportModule("torch_supa._C"));
  if (!torch_supa_C_module) {
    return nullptr;
  }
  auto torch_supa_C_m = py::handle(torch_supa_C_module).cast<py::module>();
  auto m = torch_supa_C_m.def_submodule("_profiler", "_profiler bindings");

  py::enum_<SupaActivityType>(m, "ProfilerActivity")
      .value("CPU", SupaActivityType::CPU)
      .value("CUDA", SupaActivityType::CUDA)
      .value("SUPA", SupaActivityType::SUPA)
      .value("XPU", SupaActivityType::XPU);

  py::class_<torch_supa::profiler::impl::SupaProfilerConfig>(m, "ProfilerConfig")
      .def(
          py::init<
              torch::profiler::impl::ProfilerState,
              bool, /* report_input_shapes */
              bool, /* profile_memory */
              bool, /* with_stack */
              bool, /* with_flops */
              bool, /* with_modules */
              bool, /* use_supa_simple */
              torch::profiler::impl::ExperimentalConfig /* experimental_config */,
              std::string /* trace_id */
              >(),
          py::arg("state"),
          py::arg("report_input_shapes"),
          py::arg("profile_memory"),
          py::arg("with_stack"),
          py::arg("with_flops"),
          py::arg("with_modules"),
          py::arg("use_supa_simple"),
          py::arg("experimental_config"),
          py::arg("trace_id") = "" // Make trace_id the only optional param
      );
  m.def("_supported_activities", []() {
    std::set<SupaActivityType> activities{SupaActivityType::CPU, SupaActivityType::SUPA};
    return activities;
  });

  py::class_<KinetoEvent>(m, "_KinetoEvent")
      // name of the event
      .def("name", [](const KinetoEvent& e) { return e.name(); })
      // PyTorch thread id of the start callback
      .def("start_thread_id", [](const KinetoEvent& e) { return e.startThreadId(); })
      // PyTorch thread id of the end callback
      .def("end_thread_id", [](const KinetoEvent& e) { return e.endThreadId(); })
      // for events of scope BACKWARD_FUNCTION - PyTorch thread id
      // of the corresponding forward op
      .def("fwd_thread_id", [](const KinetoEvent& e) { return e.fwdThreadId(); })
      // together with fwd_thread_id, used to uniquely identify
      // the forward op
      .def("sequence_nr", [](const KinetoEvent& e) { return e.sequenceNr(); })
      // absolute start time (since unix epoch) in ns
      .def("start_ns", [](const KinetoEvent& e) { return e.startNs(); })
      // absolute end time (since unix epoch) in ns
      .def("end_ns", [](const KinetoEvent& e) { return e.endNs(); })
      // duration in ns
      .def("duration_ns", [](const KinetoEvent& e) { return e.durationNs(); })
      // used for correlation between high-level PyTorch events
      // and low-level device events
      .def("correlation_id", [](const KinetoEvent& e) { return e.correlationId(); })
      // shapes of input tensors
      .def("shapes", [](const KinetoEvent& e) { return e.shapes().vec(); })
      .def("dtypes", [](const KinetoEvent& e) { return e.dtypes().vec(); })
      .def(
          "concrete_inputs",
          [](const KinetoEvent& e) {
            std::vector<py::object> as_pyobj;
            std::transform(
                e.concreteInputs().begin(),
                e.concreteInputs().end(),
                std::back_inserter(as_pyobj),
                [](const c10::IValue& val) { return torch::jit::toPyObject(val); });
            return as_pyobj;
          })
      .def(
          "kwinputs",
          [](const KinetoEvent& e) {
            std::unordered_map<std::string, py::object> inputs;
            for (const auto& [key, value] : e.kwinputs()) {
              inputs[key] = torch::jit::toPyObject(value);
            }
            return inputs;
          })
      // stack traces of the PyTorch CPU events
      .def("stack", [](const KinetoEvent& e) { return e.stack().vec(); })
      // type of the RecordFunction that generated a PyTorch CPU event
      // (op, torchscript function, user label, etc)
      .def("scope", [](const KinetoEvent& e) { return e.scope(); })
      // device number, for CPU - process id
      .def("device_index", [](const KinetoEvent& e) { return e.deviceIndex(); })
      // for CUDA - stream id, for CPU - start thread id
      .def("device_resource_id", [](const KinetoEvent& e) { return e.deviceResourceId(); })
      // device type
      .def("device_type", [](const KinetoEvent& e) { return e.deviceType(); })
      // correlation id of a linked event
      .def("linked_correlation_id", [](const KinetoEvent& e) { return e.linkedCorrelationId(); })
      // compute flops
      .def("flops", [](const KinetoEvent& e) { return e.flops(); })
      // Whether this is async event or not
      .def("is_async", [](const KinetoEvent& e) { return e.isAsync(); })
      .def("supa_elapsed_us", &KinetoEvent::supaElapsedUs)
      .def("privateuse1_elapsed_us", &KinetoEvent::privateuse1ElapsedUs)
      .def(
          "is_user_annotation",
          [](const KinetoEvent& e) {
            return e.activityType() == (uint8_t)libkineto::ActivityType::USER_ANNOTATION ||
                e.activityType() == (uint8_t)libkineto::ActivityType::GPU_USER_ANNOTATION;
          })
      .def("nbytes", [](const KinetoEvent& e) { return e.nBytes(); });

  py::class_<ProfilerResult>(m, "_ProfilerResult")
      .def("trace_start_ns", &ProfilerResult::trace_start_ns)
      .def("events", &ProfilerResult::events)
      .def("experimental_event_tree", &ProfilerResult::event_tree)
#ifdef USE_KINETO
      .def("save", &ProfilerResult::save)
#endif // USE_KINETO
      ;

  m.def(
      "_enable_profiler",
      &enableProfiler,
      py::arg("config"),
      py::arg("activities"),
      py::arg("scopes") = std::unordered_set<at::RecordScope>());
  m.def("_disable_profiler", disableProfiler);
  m.def("_prepare_profiler", prepareProfiler, py::call_guard<py::gil_scoped_release>());
  m.def("_toggle_collection_dynamic", toggleCollectionDynamic, py::call_guard<py::gil_scoped_release>());
  m.def("_add_metadata_json", addMetadataJson);
  m.def("_kineto_step", profilerStep); // Only if `USE_KINETO` is set
  torch_supa::profiler::python_tracer::init();
  Py_RETURN_TRUE;
}

// autograd methods on torch._C
static PyMethodDef TorchProfilerMethods[] = { // NOLINT
    {"_profiler_init", profiler_initExtension, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

TORCH_SUPA_API PyMethodDef* profiler_functions() {
  return TorchProfilerMethods;
}

} // namespace profiler
} // namespace torch_supa
