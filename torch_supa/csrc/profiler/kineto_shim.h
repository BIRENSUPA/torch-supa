/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <memory>
#include <string>

#include <ActivityType.h>

#include <torch/csrc/profiler/kineto_shim.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

#ifdef USE_KINETO
// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto {
class GenericTraceActivity;
struct CpuTraceBuffer;
class ActivityTraceInterface;
} // namespace libkineto
#endif

namespace torch_supa {
namespace profiler {

namespace impl::kineto {

#ifdef USE_KINETO
using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;
using activity_t = libkineto::GenericTraceActivity;
#else
struct DummyTraceBuffer {};
struct DummyTraceInterface {};

using trace_t = DummyTraceBuffer;
using interface_trace_t = DummyTraceBuffer;
struct activity_t;
#endif // USE_KINETO

using torch::profiler::impl::kineto::DeviceAndResource;
DeviceAndResource kineto_ids();

void addMetadata(activity_t* activity, const std::string& key, const std::string& value);

using ActivitySet = std::set<torch_supa::profiler::impl::SupaActivityType>;

// Wraps: libkineto::CpuTraceBuffer
struct TraceWrapper {
  TraceWrapper(int64_t start_time, const std::string& name);

  // The caller is expected to hold a mutex when calling `addCPUActivity`.
  activity_t* addCPUActivity(
      const std::string& name,
      libkineto::ActivityType type,
      DeviceAndResource device_and_resource,
      uint64_t correlation_id,
      int64_t start_time,
      int64_t end_time);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() {
    return cpu_trace_;
  }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

using ActivityTraceWrapper = torch::profiler::impl::kineto::ActivityTraceWrapper;
void prepareTrace(
    bool cpuOnly,
    const ActivitySet& activities,
    const ExperimentalConfig& config,
    const std::string& trace_id);

void toggleCollectionDynamic(bool enable);
void startTrace();
ActivityTraceWrapper stopTrace();

void pushCorrelationId(uint64_t correlation_id);
void pushUserCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void popUserCorrelationId();
void recordThreadInfo();
} // namespace impl::kineto

TORCH_SUPA_API void addMetadataJson(const std::string& key, const std::string& value);
TORCH_SUPA_API void profilerStep();

} // namespace profiler

} // namespace torch_supa
