/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <string>
#include <vector>

#include <torch/csrc/autograd/profiler_kineto.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/profiler/kineto_shim.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

namespace torch_supa {
namespace profiler {

using experimental_event_t = std::shared_ptr<torch::profiler::impl::Result>;
using extra_meta_t = std::unordered_map<std::string, std::string>;
using post_process_t = std::function<void(
    /*debug_handle */ int64_t,
    /*jit_stack    */ std::vector<std::string>&,
    /*jit_modules  */ std::vector<std::string>&)>;

struct TORCH_API KinetoEvent {
  KinetoEvent(const std::shared_ptr<const torch::profiler::impl::Result>& /*result*/, bool verbose);

  uint64_t startThreadId() const;
  uint64_t endThreadId() const;
  uint8_t activityType() const;
  uint64_t fwdThreadId() const;
  bool hasShapes() const;
  c10::ArrayRef<std::vector<int64_t>> shapes() const;
  bool hasTypes() const;
  c10::ArrayRef<std::string> dtypes() const;
  bool hasConcreteInputs() const;
  c10::ArrayRef<c10::IValue> concreteInputs() const;
  bool hasKwinputs() const;
  std::unordered_map<std::string, c10::IValue> kwinputs() const;
  uint64_t flops() const;
  int64_t sequenceNr() const;
  bool hasStack() const;
  c10::ArrayRef<std::string> stack() const;
  uint8_t scope() const;
  bool hasModuleHierarchy() const;
  c10::ArrayRef<std::string> moduleHierarchy() const;
  int64_t debugHandle() const;
  std::string name() const;
  c10::DeviceType deviceType() const;
  int deviceIndex() const;
  int64_t nBytes() const;
  uint64_t startNs() const;
  uint64_t endNs() const;
  uint64_t durationNs() const;
  bool isAsync() const;
  uint64_t correlationId() const;
  uint64_t linkedCorrelationId() const;
  int64_t deviceResourceId() const;
  std::string backend() const;
  bool isPythonFunction() const;
  int64_t supaElapsedUs() const;
  int64_t privateuse1ElapsedUs() const;
  void getPerfEventCounters(torch::profiler::perf_counters_t& /*in*/) const;
  extra_meta_t extraMeta() const;

 private:
  torch::profiler::impl::ProfilerVoidEventStub fallbackStart() const;
  torch::profiler::impl::ProfilerVoidEventStub fallbackEnd() const;

  std::shared_ptr<const torch::profiler::impl::Result> result_;
  std::vector<std::string> python_stack_;

  // Copy fields from result so we can return ArrayRefs.
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<std::string> dtypes_;
  std::vector<c10::IValue> concrete_inputs_;
  std::unordered_map<std::string, c10::IValue> kwinputs_;
};

struct TORCH_SUPA_API ProfilerResult {
  ProfilerResult();
  ProfilerResult(
      uint64_t start_time,
      std::vector<KinetoEvent> events,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&& trace,
      std::vector<experimental_event_t>&& event_tree);
  ProfilerResult(const ProfilerResult&) = delete;
  ProfilerResult& operator=(const ProfilerResult&) = delete;
  ProfilerResult(ProfilerResult&&) = delete;
  ProfilerResult& operator=(ProfilerResult&&) = delete;
  ~ProfilerResult();

  uint64_t trace_start_ns() const {
    return trace_start_ns_;
  }

  const std::vector<KinetoEvent>& events() const {
    return events_;
  }

  const std::vector<experimental_event_t>& event_tree() const {
    return event_tree_;
  }

  void save(const std::string& path);

 private:
  uint64_t trace_start_ns_ = 0;
  std::vector<KinetoEvent> events_;
  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper> trace_;
  std::vector<experimental_event_t> event_tree_;
};

TORCH_SUPA_API void prepareProfiler(
    const torch_supa::profiler::impl::SupaProfilerConfig& supa_config,
    const std::set<torch_supa::profiler::impl::SupaActivityType>& activities);

TORCH_SUPA_API void toggleCollectionDynamic(
    bool enable,
    const std::set<torch_supa::profiler::impl::SupaActivityType>& activities);

TORCH_SUPA_API void enableProfiler(
    const torch_supa::profiler::impl::SupaProfilerConfig& supa_config,
    const std::set<torch_supa::profiler::impl::SupaActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes = {});

TORCH_SUPA_API std::unique_ptr<ProfilerResult> disableProfiler();

} // namespace profiler
} // namespace torch_supa
