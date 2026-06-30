/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
#include <c10/util/intrusive_ptr.h>
#include <pybind11/chrono.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>

#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/distributed/ProcessGroupBCCL.hpp"
#include "torch_supa/csrc/distributed/c10d.h"
#include "torch_supa/csrc/distributed/symm_mem/IntraNodeComm.hpp"

using namespace torch;

namespace {
// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl) : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      // NOLINTNEXTLINE(bugprone-exception-escape)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

namespace c10d::supa {

namespace {

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ = py::class_<T, IntrusivePtrNoGilDestructor<T>>;

PyObject* Init(PyObject* _unused, PyObject* noargs) {
  C10_LOG_API_USAGE_ONCE("c10d.supa.python.import");

  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch_supa.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch_supa._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule("_distributed_c10d", "distributed c10d::supa bindings");

  auto module = py::handle(m).cast<py::module>();
  py::module_ dist = py::module_::import("torch.distributed");
  auto processGroupBCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::supa::ProcessGroupBCCL>(
          module, "ProcessGroupBCCL", dist.attr("_Backend"))
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          c10::intrusive_ptr<::c10d::supa::ProcessGroupBCCL::Options> options) {
                return c10::make_intrusive<::c10d::supa::ProcessGroupBCCL>(store, rank, size, std::move(options));
              }),
              py::call_guard<py::gil_scoped_release>(),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("options"),
              R"(Create a new ProcessGroupBCCL instance.)")
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                auto options = ::c10d::supa::ProcessGroupBCCL::Options::create();
                options->is_high_priority_stream = false;
                options->timeout = timeout;
                return c10::make_intrusive<::c10d::supa::ProcessGroupBCCL>(store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = ::c10d::supa::kProcessGroupBCCLDefaultTimeout,
              py::call_guard<py::gil_scoped_release>(),
              R"(Create a new ProcessGroupBCCL instance.)")
          .def(
              "_shutdown",
              [](const c10::intrusive_ptr<::c10d::supa::ProcessGroupBCCL>& self) { return self->shutdown(); },
              py::call_guard<py::gil_scoped_release>())
          .def("_group_start", &::c10d::supa::ProcessGroupBCCL::groupStart)
          .def("_group_end", &::c10d::supa::ProcessGroupBCCL::groupEnd)
          .def("comm_split_count", &::c10d::supa::ProcessGroupBCCL::getCommSplitCounter)
          .def(
              "_set_default_timeout",
              [](const c10::intrusive_ptr<::c10d::supa::ProcessGroupBCCL>& self, std::chrono::milliseconds timeout) {
                self->getOptions()->timeout = timeout;
              },
              py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_add_ephemeral_timeout",
              [](const c10::intrusive_ptr<::c10d::supa::ProcessGroupBCCL>& self,
                 const std::chrono::milliseconds& timeout) { self->addEphemeralTimeout(timeout); },
              py::arg("timeout"))
          .def(
              "_verify_work_timeout",
              [](const c10::intrusive_ptr<::c10d::supa::ProcessGroupBCCL>& self,
                 const c10::intrusive_ptr<::c10d::Work>& work,
                 const std::chrono::milliseconds& timeout) { return self->verifyWorkTimeoutForTest(work, timeout); },
              py::arg("work"),
              py::arg("timeout"))
          .def_property_readonly(
              "options",
              &::c10d::supa::ProcessGroupBCCL::getOptions,
              R"(Return the options used to create this ProcessGroupBCCL instance.)")
          .def_property_readonly("uid", &::c10d::supa::ProcessGroupBCCL::getUid, R"(Return the uid.)")
#if TORCH_VER >= TORCH_2_3_0
          .def_property(
              "bound_device_id",
              &::c10d::supa::ProcessGroupBCCL::getBoundDeviceId,
              &::c10d::supa::ProcessGroupBCCL::setBoundDeviceId,
              R"(Return the bound device id.)")
#endif
          .def("perform_nocolor_split", &::c10d::supa::ProcessGroupBCCL::performNocolorSplit)
          .def(
              "register_mem_pool",
              &::c10d::supa::ProcessGroupBCCL::registerMemPool,
              py::arg("pool"),
              py::arg("symm") = false)
          .def("deregister_mem_pool", &::c10d::supa::ProcessGroupBCCL::deregisterMemPool)
          .def(
              "abort",
              &::c10d::supa::ProcessGroupBCCL::abort,
              py::call_guard<py::gil_scoped_release>(),
              R"(Abort the process group.)")
          .def(
              "_is_initialized",
              &::c10d::supa::ProcessGroupBCCL::isInitialized,
              py::call_guard<py::gil_scoped_release>());

  module.def("_get_intra_node_comm_usage_counter", &::c10d::supa::intra_node_comm::getIntraNodeCommUsageCounter);

#ifdef BCCL_HAS_CONFIG
  py::class_<bcclConfig_t>(
      processGroupBCCL,
      "BCCLConfig",
      R"(
bcclConfig_t data type for configuring BCCL communicators.
See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
for details.
)")
      .def(py::init<>())
      .def_readwrite("blocking", &bcclConfig_t::blocking)
      .def_readwrite("cga_cluster_size", &bcclConfig_t::cgaClusterSize)
      .def_readwrite("min_ctas", &bcclConfig_t::minCTAs)
      .def_readwrite("max_ctas", &bcclConfig_t::maxCTAs)
#ifdef BCCL_HAS_COMM_SPLIT
      .def_readwrite("split_share", &bcclConfig_t::splitShare)
#endif
#ifdef BCCL_HAS_QOS
      .def_readwrite("traffic_class", &bcclConfig_t::trafficClass)
#endif
#ifdef BCCL_HAS_COLLNET
      .def_readwrite("collnet_enable", &bcclConfig_t::collnetEnable)
#endif
#ifdef BCCL_HAS_CTA_POLICY
      .def_readwrite("cta_policy", &bcclConfig_t::CTAPolicy)
#endif
#ifdef BCCL_HAS_NVLS_CTAS
      .def_readwrite("nvls_ctas", &bcclConfig_t::nvlsCTAs)
#endif
      .def_property(
          "net_name",
          [](const bcclConfig_t& self) { return self.netName; },
          // Note: BCCL calls free on the netName pointer
          // when destroying the communicator. So memory
          // shouldn't leak because of allocation in strdup.
          [](bcclConfig_t& self, const char* tmp) { self.netName = strdup(tmp); });
#endif // BCCL_HAS_CONFIG

  auto backendOptions = dist.attr("_Backend").attr("Options");

  intrusive_ptr_class_<::c10d::supa::ProcessGroupBCCL::Options>(
      processGroupBCCL,
      "Options",
      backendOptions,
      R"(
ProcessGroup options for the BCCL backend

Arguments:
    is_high_priority_stream (bool, optional): flag to enable/disable process
            group to pick up high priority cuda streams. It lets SUPA driver
            to prioritize BCCL kernels when there are compute kernels waiting.
            Default is False.

Attributes:
    config (BCCLConfig): configures BCCL communicators. This can be used to improve
            communication-computation overlap for BCCL kernels by tuning
            available parameters in the config. See
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
            for details.

Example::
    >>> import torch.distributed as dist
    >>>
    >>> bccl_options = dist.ProcessGroupBCCL.Options(is_high_priority_stream=True)
    >>> # For example, configure communicators
    >>> bccl_options.config.cga_cluster_size = 2
    >>> bccl_options.config.max_ctas = 4
    >>> bccl_options.config.min_ctas = 2
    >>> bccl_options.config.split_share = 1
    >>> # initialize a bccl process group with the options just created
    >>> dist.init_process_group("bccl", pg_options=bccl_options)
      )")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
#ifdef BCCL_HAS_CONFIG
      .def_readwrite("config", &::c10d::supa::ProcessGroupBCCL::Options::config)
#endif
      .def_readwrite("is_high_priority_stream", &::c10d::supa::ProcessGroupBCCL::Options::is_high_priority_stream)
      .def_readwrite("split_from", &::c10d::supa::ProcessGroupBCCL::Options::split_from)
      .def_readwrite("split_color", &::c10d::supa::ProcessGroupBCCL::Options::split_color)
      .def_readwrite("global_ranks_in_group", &::c10d::supa::ProcessGroupBCCL::Options::global_ranks_in_group)
      .def_readwrite("group_name", &::c10d::supa::ProcessGroupBCCL::Options::group_name)
      .def(
          "__copy__",
          [](const ::c10d::supa::ProcessGroupBCCL::Options& self) {
            return ::c10d::supa::ProcessGroupBCCL::Options(self);
          })
      .def(
          "__deepcopy__",
          [](const ::c10d::supa::ProcessGroupBCCL::Options& self, const py::dict& memo) {
            return ::c10d::supa::ProcessGroupBCCL::Options(self);
          },
          py::arg("memo"));

  module.def(
      "_hash_tensors",
      [](const std::vector<at::Tensor>& tensors) { return ::c10d::supa::hashTensors(tensors); },
      py::arg("tensors"),
      R"(
        Arguments:
          tensors(List[torch.Tensor]): List of tensors we want to hash.
      )");
  module.def(
      "_dump_bccl_trace_json",
      [](std::optional<bool> includeCollectives, std::optional<bool> onlyActive) {
        return py::bytes(
            ::c10d::supa::dump_bccl_trace_json(includeCollectives.value_or(true), onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
      Arguments:
            includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
            onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
      Returns:
            Stringified json work traces.
            Default settings return everything - i.e. contains BCCL comm dumps and collective traces.
      )");
  module.def(
      "_dump_bccl_trace",
      [](std::optional<bool> includeCollectives,
         std::optional<bool> includeStackTraces,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::supa::dump_bccl_trace(
            includeCollectives.value_or(true), includeStackTraces.value_or(true), onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("includeStackTraces") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
        Arguments:
            includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
            includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
            onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
            Stringified pickle work traces.
            Default settings return everything - i.e. contains BCCL comm dumps and collective traces.
      )");

  module.attr("_DEFAULT_PG_BCCL_TIMEOUT") = py::cast(::c10d::supa::kProcessGroupBCCLDefaultTimeout);

  Py_RETURN_TRUE;
}

} // namespace

// c10d::supa methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_supa_init", Init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// NOLINTNEXTLINE(misc-use-internal-linkage)
PyMethodDef* InitFunctions() {
  return methods;
}

} // namespace c10d::supa
