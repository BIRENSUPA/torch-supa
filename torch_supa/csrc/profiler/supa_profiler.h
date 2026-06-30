/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <array>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/stubs/base.h>
#include "record_function_wrapper.h"

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/profiler/util.h"

namespace torch_supa {
namespace profiler {

namespace impl {

using torch::profiler::impl::ExperimentalConfig;
using torch::profiler::impl::ProfilerConfig;
using torch::profiler::impl::ProfilerState;

enum class C10_API_ENUM SupaActivityType {
  CPU = 0,
  CUDA,
  SUPA,
  XPU,
  NUM_KINETO_ACTIVITIES, // must be the last one
};

inline std::string actToString(SupaActivityType t) {
  const std::array<std::string, static_cast<size_t>(SupaActivityType::NUM_KINETO_ACTIVITIES)> ActivityTypeNames = {
      "CPU", "CUDA", "SUPA"};
  return ActivityTypeNames.at(static_cast<size_t>(t));
}

struct TORCH_SUPA_API SupaProfilerConfig {
  explicit SupaProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false,
      bool use_supa_simple = false,
      ExperimentalConfig experimental_config = ExperimentalConfig(),
      std::string trace_id = "");

  bool disabled() const;
  bool global() const;

  ProfilerConfig base_config;
  bool use_supa_simple;

  void init_supa_simple() const;
};

} // namespace impl

class SUPARecordFunction : public at::RecordFunction {
 public:
  explicit SUPARecordFunction(bool enable_ = false) : enable(enable_) {
    if (SUPARecordFunction::use_supa_simple) {
      at::enableRecordFunction(enable);
    }
  }

  SUPARecordFunction(const SUPARecordFunction&) = delete;
  SUPARecordFunction& operator=(const SUPARecordFunction&) = delete;
  SUPARecordFunction(SUPARecordFunction&&) = delete;
  SUPARecordFunction& operator=(SUPARecordFunction&&) = delete;

  ~SUPARecordFunction() override {
    if (SUPARecordFunction::use_supa_simple) {
      at::enableRecordFunction(!enable);
    }
  }
  bool enable = false;
  static bool use_supa_simple;

  std::string additional_args;
  bool on_device = false;
};
} // namespace profiler
} // namespace torch_supa
