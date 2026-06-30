/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <torch/csrc/profiler/collection.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <variant>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/profiler/kineto_shim.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

namespace torch_supa::profiler::impl {

class SupaRecordQueue : public torch::profiler::impl::RecordQueue {
 public:
  explicit SupaRecordQueue(const SupaProfilerConfig& config, std::set<SupaActivityType> activities);

 private:
  SupaProfilerConfig config_;
  std::set<SupaActivityType> activities_;
};

TORCH_SUPA_API bool get_supa_sync_enabled();
TORCH_SUPA_API void set_supa_sync_enabled_fn(std::function<bool()> /*fn*/);
TORCH_SUPA_API void set_supa_sync_enabled_val(bool /*val*/);

} // namespace torch_supa::profiler::impl