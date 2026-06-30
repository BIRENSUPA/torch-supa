/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/profiler/kineto_shim.h"
#include "torch_supa/csrc/profiler/collection.h"

#ifdef USE_KINETO
#include <libkineto.h>
#endif

namespace torch_supa {

namespace profiler {

namespace impl::kineto {

#ifdef USE_KINETO
const std::set<libkineto::ActivityType> kCpuTypes{
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::XPU_RUNTIME,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::SUPA_RUNTIME,
    libkineto::ActivityType::SUPA_DRIVER,
    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> kSupaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::GPU_USER_ANNOTATION,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::GPU_GRAPH,
    // SUPA_RUNTIME appears in both kCpuTypes and kSupaTypes.
    libkineto::ActivityType::SUPA_RUNTIME,
    libkineto::ActivityType::SUPA_DRIVER,
    libkineto::ActivityType::OVERHEAD,
};
#endif

bool collectivesProfilerExists() {
  return true;
}

#ifdef USE_KINETO
static std::string setTraceID(const std::string& trace_id) {
  if (trace_id.empty()) {
    return "";
  }
  std::stringstream configss;
  configss << "REQUEST_TRACE_ID=" << trace_id << "\n";
  configss << "REQUEST_GROUP_TRACE_ID=" << trace_id << "\n";
  return configss.str();
}
#endif

namespace {
// Handles processing of Experimental Config options for Kineto
class ExperimentalConfigWrapper {
 public:
  explicit ExperimentalConfigWrapper(const ExperimentalConfig& config) : config_(config) {}

  bool assertValid() {
    return !config_.profiler_metrics.empty();
  }

  void prepareTraceWithExperimentalOptions(bool add_cpu_activity) {
#ifdef USE_KINETO
    std::set<libkineto::ActivityType> k_activities{libkineto::ActivityType::SUPA_PROFILER_RANGE};

    // Only add CPU activities if we are measuring per kernel ranges
    if (add_cpu_activity && config_.profiler_measure_per_kernel) {
      k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
    }

    const size_t num_metrics = config_.profiler_metrics.size();
    std::stringstream configss;

    LOG(INFO) << "SUPTI profiler metrics size = " << num_metrics;

    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
             << "SUPTI_PROFILER_METRICS=";

    for (size_t i = 0; i < num_metrics; i++) {
      configss << config_.profiler_metrics[i];
      if (num_metrics > 1 && i < (num_metrics - 1)) {
        configss << ",";
      }
    }
    configss << "\nSUPTI_PROFILER_ENABLE_PER_KERNEL=" << (config_.profiler_measure_per_kernel ? "true" : "false")
             << "\n";
    LOG(INFO) << "Generated config = " << configss.str();

    libkineto::api().activityProfiler().prepareTrace(k_activities, configss.str());
#endif // USE_KINETO
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ExperimentalConfig& config_;
};
} // namespace

DeviceAndResource kineto_ids() {
#ifdef USE_KINETO
  return {
      /*device=*/libkineto::processId(),
      /*resource=*/libkineto::systemThreadId()};
#else
  return {};
#endif // USE_KINETO
}

void addMetadata(activity_t* activity, const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  activity->addMetadata(key, value);
#endif // USE_KINETO
}

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
#ifdef USE_KINETO
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}
#else
{
}
#endif // USE_KINETO

activity_t* TraceWrapper::addCPUActivity(
    const std::string& name,
    const libkineto::ActivityType type,
    const DeviceAndResource device_and_resource,
    const uint64_t correlation_id,
    const int64_t start_time,
    const int64_t end_time) {
#ifdef USE_KINETO
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);
  auto& act = libkineto::CpuTraceBuffer::toRef(cpu_trace_->activities.back());
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = static_cast<int32_t>(correlation_id);
  act.startTime = start_time;
  if (type != libkineto::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return cpu_trace_->activities.back().get();
#else
  return nullptr;
#endif // USE_KINETO
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
#ifdef USE_KINETO
  cpu_trace_->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));
#endif // USE_KINETO
}

TraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return cpu_trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const ExperimentalConfig& config,
    const std::string& trace_id) {
#ifdef USE_KINETO
  libkineto::api().resetKinetoTLS();
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  bool has_cpu_activity = activities.count(torch_supa::profiler::impl::SupaActivityType::CPU);

  if (has_cpu_activity) {
    k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
  }
  if (activities.count(torch_supa::profiler::impl::SupaActivityType::SUPA)) {
    k_activities.insert(kSupaTypes.begin(), kSupaTypes.end());
    if (config.enable_cuda_sync_events || get_supa_sync_enabled()) {
      LOG(INFO) << "Enabling SUPA Sync Events";
      k_activities.insert(libkineto::ActivityType::SUPA_SYNC);
    }
  }
  if (collectivesProfilerExists()) {
    k_activities.insert(libkineto::ActivityType::COLLECTIVE_COMM);
  }

  ExperimentalConfigWrapper configWrap(config);

  // Experimental Configuration options are present
  if (config && configWrap.assertValid()) {
    configWrap.prepareTraceWithExperimentalOptions(has_cpu_activity);
    return;
  }

  const std::string configStr = setTraceID(trace_id);

  libkineto::api().activityProfiler().prepareTrace(k_activities, configStr);
#endif // USE_KINETO
}

void toggleCollectionDynamic(const bool enable) {
#ifdef USE_KINETO
  // TODO: We may want to consider adding another input arg for this function
  // if we want to support turning off certain devices and keeping others on.
  // For now, we can keep it simple at have it turn off all tracing of "CUDA"
  // devices
  libkineto::api().activityProfiler().toggleCollectionDynamic(enable);
#endif // USE_KINETO
}

void startTrace() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().startTrace();
#endif // USE_KINETO
}

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{
#ifdef USE_KINETO
      libkineto::api().activityProfiler().stopTrace()
#else
      std::make_unique<interface_trace_t>()
#endif // USE_KINETO
  };
}

void pushCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushCorrelationId(correlation_id);
#endif // USE_KINETO
}

void pushUserCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushUserCorrelationId(correlation_id);
#endif // USE_KINETO
}

void popCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
}

void popUserCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popUserCorrelationId();
#endif // USE_KINETO
}

void recordThreadInfo() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().recordThreadInfo();
#endif // USE_KINETO
}

} // namespace impl::kineto

void addMetadataJson(const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
#else
  LOG(WARNING) << "Adding profiling metadata requires using "
               << "torch.profiler with Kineto support (USE_KINETO=1)";
#endif // USE_KINETO
}

void profilerStep() {
#ifdef USE_KINETO
  libkineto::api().initProfilerIfRegistered();

  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().step();
  } else {
    VLOG(1) << "Profiler is not initialized: skipping step() invocation";
  }
#endif // USE_KINETO
}

} // namespace profiler
} // namespace torch_supa

namespace torch {
namespace profiler {
namespace impl::kineto {
ActivityTraceWrapper::ActivityTraceWrapper(std::unique_ptr<interface_trace_t>&& trace) : trace_(std::move(trace)) {}

ActivityTraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

void ActivityTraceWrapper::save(const std::string& path) {
#ifdef USE_KINETO
  TORCH_CHECK(!saved_, "Trace is already saved.");
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
#else
  TORCH_CHECK(
      false,
      "Saving a trace requires using torch.profiler with Kineto "
      "support (USE_KINETO=1)");
#endif // USE_KINETO
}
} // namespace impl::kineto
} // namespace profiler
} // namespace torch
