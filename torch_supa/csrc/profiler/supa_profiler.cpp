/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/profiler/supa_profiler.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"

namespace torch_supa {
namespace profiler {

bool SUPARecordFunction::use_supa_simple = false;
namespace impl {

void SupaProfilerConfig::init_supa_simple() const {
  if (base_config.state == ProfilerState::KINETO) {
    torch_supa::profiler::SUPARecordFunction::use_supa_simple = use_supa_simple;
  }
}

SupaProfilerConfig::SupaProfilerConfig(
    torch::profiler::impl::ProfilerState state,
    bool report_input_shapes,
    bool profile_memory,
    bool with_stack,
    bool with_flops,
    bool with_modules,
    bool use_supa_simple,
    ExperimentalConfig experimental_config,
    std::string trace_id)
    : base_config(
          state,
          report_input_shapes,
          profile_memory,
          with_stack,
          with_flops,
          with_modules,
          std::move(experimental_config)
#if TORCH_VER >= TORCH_2_6_0
              ,
          std::move(trace_id)
#endif
              ),
      use_supa_simple(use_supa_simple) {
  init_supa_simple();
}

bool SupaProfilerConfig::disabled() const {
  return base_config.state == torch::profiler::impl::ProfilerState::Disabled;
}

bool SupaProfilerConfig::global() const {
  return base_config.state == torch::profiler::impl::ProfilerState::KINETO_ONDEMAND;
}

} // namespace impl

} // namespace profiler
} // namespace torch_supa