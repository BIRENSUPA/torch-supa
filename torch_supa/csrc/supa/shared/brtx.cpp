/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <brtx/brToolsExt.h>
#include <supa_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <memory>
#include <string>
#include "torch_supa/csrc/core/supa/SUPAException.h"

namespace torch_supa::supa::shared {

struct RangeHandle {
  brtxRangeId_t id{};
  std::string msg;
};

static void device_callback_range_end(void* userData) {
  auto handle = std::unique_ptr<RangeHandle>(static_cast<RangeHandle*>(userData));
  brtxRangeEnd(handle->id);
}

static void device_brtxRangeEnd(void* handle, std::intptr_t stream) {
  supaLaunchHostFunc(reinterpret_cast<supaStream_t>(stream), device_callback_range_end, handle);
}

static void device_callback_range_start(void* userData) {
  auto* handle = static_cast<RangeHandle*>(userData);
  handle->id = brtxRangeStartA(handle->msg.c_str());
}

static void* device_brtxRangeStart(const char* msg, std::intptr_t stream) {
  auto handle = std::make_unique<RangeHandle>();
  handle->msg = msg;
  handle->id = 0;
  auto* raw_handle = handle.release();
  supaLaunchHostFunc(reinterpret_cast<supaStream_t>(stream), device_callback_range_start, raw_handle);
  return raw_handle;
}

TORCH_SUPA_API void InitBrtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto brtx = m.def_submodule("_brtx", "BRTX bindings");
  brtx.def("rangePushA", brtxRangePushA);
  brtx.def("rangePop", brtxRangePop);
  brtx.def("rangeStartA", brtxRangeStartA);
  brtx.def("rangeEnd", brtxRangeEnd);
  brtx.def("markA", brtxMarkA);
  brtx.def("deviceRangeStart", device_brtxRangeStart);
  brtx.def("deviceRangeEnd", device_brtxRangeEnd);
}

} // namespace torch_supa::supa::shared