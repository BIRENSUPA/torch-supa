/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "record_function_wrapper.h"

#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <unistd.h>
#include <queue>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#include "torch_supa/csrc/profiler/collection.h"

#ifdef USE_KINETO
#include <libkineto.h>
#endif

namespace torch_supa::profiler::impl {

using torch::profiler::impl::ThreadLocalSubqueue;
using trace_ptr_t = std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>;

namespace {

std::function<bool()>& supa_sync_enabled_fn() {
  static std::function<bool()> fn = []() { return false; };
  return fn;
}

std::set<torch::profiler::impl::ActivityType> translate(std::set<torch_supa::profiler::impl::SupaActivityType> src) {
  std::set<torch::profiler::impl::ActivityType> result;
  if (src.count(SupaActivityType::CPU) > 0) {
    result.emplace(torch::profiler::impl::ActivityType::CPU);
  }
  return result;
}
} // namespace

SupaRecordQueue::SupaRecordQueue(const SupaProfilerConfig& config, std::set<SupaActivityType> activities)
    : torch::profiler::impl::RecordQueue(config.base_config, translate(activities)), config_(config) {}

bool get_supa_sync_enabled() {
  return supa_sync_enabled_fn()();
}

void set_supa_sync_enabled_fn(std::function<bool()> fn) {
  supa_sync_enabled_fn() = std::move(fn);
}

void set_supa_sync_enabled_val(bool val) {
  supa_sync_enabled_fn() = [val]() { return val; };
}

} // namespace torch_supa::profiler::impl

namespace torch {
namespace profiler {
namespace impl {

using result_ptr_t = std::shared_ptr<Result>;
using trace_ptr_t = std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>;

// ----------------------------
// |  Input / Output encoder  |
// ----------------------------
void InputOutputEncoder::push(c10::ArrayRef<const c10::IValue> values) {
  for (const auto& value : values) {
    if (value.isTensor()) {
      push(value.toTensor());
    } else if (value.isScalar()) {
      tags_.emplace_back(Tag::Scalar);
      // Scalars are small enough that they are stored in ivalues without an
      // extra memory alloc
      // TODO: further optimize this by maybe giving Profiler access to the
      // guts of IValue.
      ivalues_.emplace_back(value);
    } else if (value.isTensorList()) {
      tags_.emplace_back(Tag::TensorListBegin);
      for (const auto& t : value.toTensorList()) {
        push(t);
      }
      tags_.emplace_back(Tag::TERMINATOR);
    } else if (isSupportedScalarList(value)) {
      tags_.emplace_back(Tag::ScalarList);
      ivalues_.emplace_back(value);
    } else {
      tags_.emplace_back(Tag::Other);
    }
  }
  tags_.emplace_back(Tag::TERMINATOR);
}

void InputOutputEncoder::push(const at::Tensor& t) {
  // TODO fix nested and symbolic sizes
  if (t.defined() && !t.is_nested() && !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    tags_.emplace_back(Tag::Tensor);
    tensor_metadata_.emplace_back(t);
    tensor_sizes_strides_.copy(t.sizes());
    if (t.layout() == at::kStrided) {
      // Only Strided layout tensors have strides
      tensor_sizes_strides_.copy(t.strides());
    }
  } else {
    tags_.emplace_back(Tag::UndefinedTensor);
  }
}

bool InputOutputEncoder::isSupportedScalarList(const c10::IValue& list_candidate) {
  // Scalar list can be very long. If a list is too long, we shouldn't
  // collect it. This function checks whether the list is a scalar list
  // and whether its length is sufficiently short.

  if (!get_record_concrete_inputs_enabled()) {
    return false;
  }

  if (!list_candidate.isList()) {
    return false;
  }
  auto list_ref = list_candidate.toListRef();
  if (C10_UNLIKELY(list_ref.empty())) {
    return true;
  }
  if (C10_UNLIKELY(!list_ref[0].isScalar())) {
    return false;
  }
  if (C10_UNLIKELY(list_ref.size() > SCALAR_LIST_LENGTH_LIMIT)) {
    return false;
  }
  return true;
}

// ---------------------------------------------------
// |  Correlation ID tracking (OpList & EventBlock)  |
// ---------------------------------------------------
template <typename T, size_t ChunkSize>
ThreadLocalSubqueue::TorchOpStorage::EventBlock<T, ChunkSize>::EventBlock() {
  static std::atomic<uint64_t> counter_{0};
  id_start_ = 1 + ChunkSize * counter_++;
}

template <class... Args>
std::pair<KinetoObserverContext::Event*, uint64_t> ThreadLocalSubqueue::TorchOpStorage::OpList::emplace_back(
    Args&&... args) {
  auto event_ptr = AppendOnlyList::emplace_back(std::forward<Args>(args)...);
  auto corr_id = buffer_last_->correlation_id(event_ptr);
  return {event_ptr, corr_id};
}

uint64_t ThreadLocalSubqueue::TorchOpStorage::OpList::correlationID(const OpList::Iterator& e) {
  return e.address().first->correlation_id(&*e);
}

template <typename T, size_t ChunkSize>
uint64_t ThreadLocalSubqueue::TorchOpStorage::EventBlock<T, ChunkSize>::correlation_id(const T* ptr) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ptr >= this->data() && ptr < this->data() + ChunkSize);
  return id_start_ + (ptr - this->data());
}

// ---------------------------------
// |  Collection (Observer logic)  |
// ---------------------------------
std::unique_ptr<KinetoObserverContext> ThreadLocalSubqueue::begin_op(const at::RecordFunction& fn) {
  auto [event, corr_id] = torch_ops_.op_events_.emplace_back(torch::profiler::impl::TorchOpBasicFields {
    fn.seqNr(), fn.forwardThreadId(), fn.scope(), fn.isAsync(),
#if TORCH_VER >= TORCH_2_4_0
        fn.handle(),
#endif
        fn.debugHandle(), fn.name()
  });
  if (config_.report_input_shapes) {
    torch_ops_.inputs_outputs_.push(fn.inputs());
#if TORCH_VER >= TORCH_2_5_0
    torch_ops_.kwinputs_.emplace_back(fn.kwinputs());
#endif
  }
  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch_supa::profiler::impl::kineto::pushUserCorrelationId(corr_id);
  } else {
    torch_supa::profiler::impl::kineto::pushCorrelationId(corr_id);
  }

#if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
  // backward nodes source range corresponds to the forward node
  // TODO: consider using C++ stack trace
  if (config_.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    auto cs = torch::profiler::impl::prepareCallstack(jit::currentCallstack());
    torch_ops_.jit_stack_.emplace_back(callstackStr(cs));
  }
  if (config_.with_modules && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    torch_ops_.jit_modules_.emplace_back(jit::currentModuleHierarchy());
  }
#endif
  if (config_.with_flops) {
    torch_ops_.extra_args_.emplace_back(torch::profiler::impl::saveExtraArgs(fn));
  }

  auto out = std::make_unique<KinetoObserverContext>(event);
#if TORCH_VER >= TORCH_2_6_0
  if (fn.isNcclMeta()) {
    // Record NCCL metadata for specific CPU ops, switch off output
    // introspection in this begin_op callback, we will do that in exit callback
    // if needed.
    torch_supa::profiler::impl::SaveBcclMetaConfig bcclMetaConfig{true, true, true, false};
    out->event_->extra_nccl_meta_ =
        torch_ops_.extra_meta_.emplace_back(torch_supa::profiler::impl::saveBcclMeta(fn, bcclMetaConfig));
  } else {
    out->event_->extra_nccl_meta_ = torch_ops_.extra_meta_.emplace_back();
  }
#else
  auto extra_mata = fn.isNcclMeta() ? torch_ops_.extra_meta_.emplace_back(torch_supa::profiler::impl::saveBcclMeta(fn))
                                    : torch_ops_.extra_meta_.emplace_back();
#endif

  const auto* supa_rf = dynamic_cast<const torch_supa::profiler::SUPARecordFunction*>(&fn);
  if (supa_rf) {
    std::unordered_map<std::string, std::string> map;
    map.emplace("", std::move(supa_rf->additional_args));
#if TORCH_VER >= TORCH_2_6_0
    out->event_->extra_nccl_meta_->merge(map);
#else
    extra_mata->merge(map);
#endif
  }

  if (config_.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      out->fallback_ = torch_ops_.device_fallback_.emplace_back();
      // torch::profiler::impl::cudaStubs()->record(
      //     nullptr, &out->fallback_->device_event_start_, nullptr);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  } else if (config_.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    out->fallback_ = torch_ops_.device_fallback_.emplace_back();
    torch::profiler::impl::privateuse1Stubs()->record(nullptr, &out->fallback_->device_event_start_, nullptr);
  }

  event->start_time_ = c10::getApproximateTime();
  event->allow_tf32_cublas_ = at::globalContext().allowTF32CuBLAS();
  if (!config_.experimental_config.performance_events.empty()) {
    // const size_t n = config_.experimental_config.performance_events.size();
    // event->counters_ = std::make_unique<perf_counters_t>(n, 0);
    // perf_profiler_->Enable();
  }
  return out;
}

ThreadLocalSubqueue::ThreadLocalSubqueue(const uint64_t tid, ProfilerConfig config)
    : tid_{tid}, config_{std::move(config)}, kineto_info_{torch_supa::profiler::impl::kineto::kineto_ids()} {
  torch_supa::profiler::impl::kineto::recordThreadInfo();
  if (!config_.experimental_config.performance_events.empty()) {
    // perf_profiler_ =
    //     std::make_unique<torch::profiler::impl::linux_perf::PerfProfiler>();
    // perf_profiler_->Configure(config_.experimental_config.performance_events);
  }
}

namespace {
struct SubQueueThreadCache {
  uint32_t key_;
  ThreadLocalSubqueue* ref_;
};

// The astute observer will note that this leaves a dangling reference; nothing
// in the teardown of `RecordQueue` or `ThreadLocalSubqueue` clears this value.
// (And the raw pointer in `SubQueueThreadCache` will not extend the lifetime
// of `*ref_`.) This is safe, however, because `getSubqueue` will check
// `sub_queue_cache_.key_` before attempting to access `ref_`, and if `key_`
// does not match the RecordQueue's *unique* `id_` it will evict
// `sub_queue_cache_` and fall back to a different mechanism.
std::atomic<uint32_t> queue_id_{0};
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};
} // anonymous namespace

RecordQueue::RecordQueue(ProfilerConfig config, std::set<ActivityType> activities)
    : id_(++queue_id_), config_{std::move(config)}, activities_{std::move(activities)} {
  if (tracePython()) {
    python_tracer_ = python_tracer::PythonTracerBase::make(this);
  }
}

bool RecordQueue::tracePython() const {
  return config_.with_stack && activities_.count(ActivityType::CPU);
}

ThreadLocalSubqueue* RecordQueue::getSubqueue() {
  // In the most common case, a thread will want to write to the same sub-queue
  // that it wrote to last call. The only time that isn't true is if:
  //  A) The profiler context has ended and we are in a new one.
  //  B) Two profilers are active in different TLS contexts, and this thread
  //     is a worker helping with intra-op parallelism.
  // Since we expect this to be the OVERWHELMINGLY common case (>99%), we add a
  // special thread_local cache so that we can skip the overall `flat_hash_map`
  // (and corresponding lock).
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it = sub_queues_.emplace(tid, std::make_unique<ThreadLocalSubqueue>(tid, config_)).first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

#define ATTRIBUTE(event_type, expr)                  \
  [&](const ExtraFields<EventType::event_type>& e) { \
    (void)e;                                         \
    return expr;                                     \
  }
namespace {
auto scopeToType(at::RecordScope scope) {
  return scope == at::RecordScope::USER_SCOPE ? libkineto::ActivityType::USER_ANNOTATION
                                              : libkineto::ActivityType::CPU_OP;
}
} // namespace

/***
  get correct activity type from activity.
  NOTE: Here need a translation because definition of ActivityType in torch_supa is different from it in pytorch/kineto.
 */
libkineto::ActivityType Result::kinetoType() const {
  return visit(c10::overloaded(
      ATTRIBUTE(TorchOp, scopeToType(e.scope_)),
      ATTRIBUTE(Backend, scopeToType(e.scope_)),
      ATTRIBUTE(Vulkan, libkineto::ActivityType::CPU_OP),
      ATTRIBUTE(Allocation, libkineto::ActivityType::CPU_INSTANT_EVENT),
      ATTRIBUTE(OutOfMemory, libkineto::ActivityType::CPU_INSTANT_EVENT),
      ATTRIBUTE(PyCall, libkineto::ActivityType::PYTHON_FUNCTION),
      ATTRIBUTE(PyCCall, libkineto::ActivityType::PYTHON_FUNCTION),
#if TORCH_VER >= TORCH_2_9_0
      ATTRIBUTE(PythonGC, libkineto::ActivityType::PYTHON_FUNCTION),
#endif
      ATTRIBUTE(Kineto, e.activity_type_)));
}

namespace {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
  // fallthrough
  switch (activity_type) {
    case libkineto::ActivityType::GPU_MEMCPY:
    case libkineto::ActivityType::GPU_MEMSET:
    case libkineto::ActivityType::CONCURRENT_KERNEL:
    case libkineto::ActivityType::CUDA_SYNC:
    case libkineto::ActivityType::GPU_USER_ANNOTATION:
    case libkineto::ActivityType::SUPA_PROFILER_RANGE:
    case libkineto::ActivityType::CUDA_PROFILER_RANGE: {
      // PrivateUse1 kineto backend reuse above ActivityTypes,
      // If PrivateUse1 backend enabled, this should return
      // c10::DeviceType::PrivateUse1.
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::CUDA;
      }();
      return device_type;
    }
    case libkineto::ActivityType::HPU_OP:
      return c10::DeviceType::HPU;
    case libkineto::ActivityType::CPU_OP:
    case libkineto::ActivityType::USER_ANNOTATION:
    case libkineto::ActivityType::EXTERNAL_CORRELATION:
    case libkineto::ActivityType::CUDA_RUNTIME:
    case libkineto::ActivityType::XPU_RUNTIME:
    case libkineto::ActivityType::CPU_INSTANT_EVENT:
    case libkineto::ActivityType::GLOW_RUNTIME:
    case libkineto::ActivityType::MTIA_RUNTIME:
    case libkineto::ActivityType::PYTHON_FUNCTION:
    case libkineto::ActivityType::CUDA_DRIVER:
    case libkineto::ActivityType::SUPA_RUNTIME:
    case libkineto::ActivityType::SUPA_DRIVER:
    case libkineto::ActivityType::PRIVATEUSE1_RUNTIME:
    case libkineto::ActivityType::PRIVATEUSE1_DRIVER:
    case libkineto::ActivityType::OVERHEAD:
      return c10::DeviceType::CPU;
    default: {
      TORCH_WARN("Unknown activity type (", (uint8_t)activity_type, "), assuming CPU device");
      return c10::DeviceType::CPU;
    }
  }
}
} // anonymous namespace

c10::DeviceType Result::deviceType() const {
  return visit(c10::overloaded(
      ATTRIBUTE(Vulkan, c10::DeviceType::Vulkan),
      ATTRIBUTE(Allocation, e.device_type_),
      ATTRIBUTE(OutOfMemory, e.device_type_),
      ATTRIBUTE(Kineto, deviceTypeFromActivity(e.activity_type_)),
      [&](const auto&) { return c10::DeviceType::CPU; }));
}
#undef ATTRIBUTE

namespace {
void mark_finished(std::shared_ptr<Result>& r) {
  TORCH_INTERNAL_ASSERT(!r->finished_, r->name());
  r->finished_ = true;
  TORCH_INTERNAL_ASSERT(r->endTimeNS() >= r->start_time_ns_, r->name());
}

#ifdef USE_KINETO
// Assumption: Total threads number will not exceed 2^16-1, and total ops will
// not exceed 2^48 -1.
uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

void generateForwardBackwardLink(
    const Result& profiler_result,
    uint64_t& fwd_bwd_link_id,
    libkineto::GenericTraceActivity& activity,
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>& tidSeq2activity) {
  const ExtraFields<EventType::TorchOp>& extra_fields =
      std::get<ExtraFields<EventType::TorchOp>>(profiler_result.extra_fields_);
  if (extra_fields.forward_tid_ > 0) {
    // act is backward op.
    uint64_t key = getForwardThreadKey(extra_fields.forward_tid_, extra_fields.sequence_number_);
    auto iter = tidSeq2activity.find(key);
    if (iter != tidSeq2activity.end()) {
      libkineto::GenericTraceActivity* fwd = iter->second;
      fwd->flow.start = true;
      activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
      activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
      ++fwd_bwd_link_id;

      // If there are multiple events that match this sequence/tid pair, we
      // should delete this entry in the map to avoid inserting multiple "end"
      // flow events.
      tidSeq2activity.erase(iter);
    }
  } else if (profiler_result.start_tid_ != 0) {
    // act is forward op.
    uint64_t key = getForwardThreadKey(profiler_result.start_tid_, extra_fields.sequence_number_);
    // Assumption: Among all ops with same sequence number,
    // the one with biggest start time is most likely launching backward op.
    auto iter = tidSeq2activity.find(key);
    if (iter == tidSeq2activity.end()) {
      tidSeq2activity[key] = &activity;
    } else {
      // Now the sequence number is only incremented on creating a "Node"
      // object for backward pass, by calling
      // "at::sequence_number::get_and_increment()". Among all ops with same
      // sequence number, the one with biggest startTime is the one launching
      // backward op.
      if (activity.startTime >= iter->second->startTime) {
        tidSeq2activity[key] = &activity;
      }
    }
  }
}
#endif // USE_KINETO

void generateForwardBackwardLinks(
    std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace,
    const std::vector<std::shared_ptr<Result>>& results){
#ifndef USE_KINETO
}
#else // USE_KINETO
    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == results.size());

// startThreadId_seqNum to pointer of activity.
// Low-16bits of startThreadId and low-48bits seqNum are concatenated into
// one uint64_t variable as key.

std::unordered_map<uint64_t, libkineto::GenericTraceActivity*> tidSeq2activity;
uint64_t fwd_bwd_link_id = 1;

using result_activity_t = std::pair<Result*, libkineto::GenericTraceActivity*>;
std::vector<result_activity_t> torch_events;

for (const auto idx : c10::irange(cpu_trace->activities.size())) {
  const auto& profiler_result = results[idx];
  auto& activity = cpu_trace->activities[idx];

  // add information about an associated forward op, if a sequence number
  // is available (e.g. during training)

  profiler_result->visit_if_base<ExtraFields<EventType::TorchOp>>([&](const auto& e) {
    if (e.sequence_number_ >= 0) {
      torch_events.emplace_back(profiler_result.get(), activity.get());
    }
  });
}

// We need to visit the events in chronological order.
// So we sort them by end_time_ns_ before processing.
std::sort(torch_events.begin(), torch_events.end(), [](const result_activity_t& left, const result_activity_t& right) {
  auto left_end_time = std::get<ExtraFields<EventType::TorchOp>>(left.first->extra_fields_).end_time_ns_;
  auto right_end_time = std::get<ExtraFields<EventType::TorchOp>>(right.first->extra_fields_).end_time_ns_;
  return left_end_time < right_end_time;
});

for (auto& [profiler_result, activity] : torch_events) {
  generateForwardBackwardLink(*profiler_result, fwd_bwd_link_id, *activity, tidSeq2activity);
}
}
#endif // USE_KINETO

constexpr const char* indexKey = "Ev Idx";

void passEventsToKineto(
    const std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    const ProfilerConfig& config) {
  using namespace torch_supa::profiler::impl::kineto;
  TraceWrapper cpu_trace(static_cast<int64_t>(start_time_ns), "PyTorch Profiler");

  // Generate Kineto events for each event recorded by the PyTorch profiler.
  for (const auto i : c10::irange(results.size())) {
    const auto& e = results[i];
    // (TODO): This is a temporary fix for async traces to make sure that we do
    // not use int64 MIN as end time in Kineto. If we use that value, the
    // duration will overflow and become a very large positive number. For a
    // long term solution, add guards in kineto for each activity type
    int64_t act_end_time = std::max(e->endTimeNS(), e->start_time_ns_);
    auto* activity = cpu_trace.addCPUActivity(
        e->name(), e->kinetoType(), e->kineto_info_, e->correlationID(), e->start_time_ns_, act_end_time);

    TORCH_INTERNAL_ASSERT(activity || !kKinetoAvailable);
    if (activity) {
      addMetadata(activity, indexKey, std::to_string(i));

      // There is a longstanding regression for initializing
      // on-demand Kineto activity handling. Enabling this path
      // for Profiler API could cause side effects as much has changed since.
      // Make a surgical fix here until we holistically assess the on-demand
      // vs API path framentation, which has been snowballing in complexity
      // and thus flakiness.
      if (config.global()) {
        e->kineto_activity_ = activity;
      }
    }
  }

  if (get_fwd_bwd_enabled()) {
    generateForwardBackwardLinks(cpu_trace.get(), results);
  }

  // Kineto adds the events that it collected.
  cpu_trace.transferCpuTrace(static_cast<int64_t>(end_time_ns));
}

#ifdef USE_KINETO
// There are two mechanisms that we use to connect Profiler and Kineto events.
// The first is the correlation ID. The profiler pushes a unique integer at the
// start of an op and pops it at the end. Kineto then associates the events
// that it collects with that correlation ID and sets the linked activity of
// the events that it collected to point to the profiler op.
//
// However, this is not a sufficient description because it does not retain
// dependency information between kineto ops. Consider a call to `torch.add`.
// Three events will be collected:
//   `aten::add`          (TorchOp, collected by profiler)
//   `cudaLaunchKernel`   (CUDA runtime event, collected by Kineto)
//   `at::vectorized_...` (GPU kernel, collected by Kineto)
// If we only relied on correlation IDs we would set both Kineto events as
// children of the `at::add`, rather than the correct
//   `at::add -> cudaLaunchKernel -> at::vectorized_...`
//
// Kineto surfaces this information through a second concept called a "flow".
// In this example, the `cudaLaunchKernel` event is the start of a flow and the
// GPU kernel has the same flow id but is not a start event. Thus, when merging
// the Kineto events into the call tree we first add all events which are flow
// start nodes. We then merge the rest, trying to pair them with flow starts
// and falling back to correlation ID if necessary. For any nodes without
// linked events the caller is determined using the normal tree construction
// algorithm.
class TransferEvents {
  using itrace_t = libkineto::ITraceActivity;
  using activity_t = torch::profiler::impl::kineto::activity_t;

 public:
  TransferEvents(std::vector<std::shared_ptr<Result>>& results, trace_ptr_t& trace) : results_{results} {
    const auto* trace_activities_ptr = trace->get()->activities();
    TORCH_INTERNAL_ASSERT(trace_activities_ptr != nullptr);
    trace_activities_ = *trace_activities_ptr;
    reassociate();
    extractEventsFromTrace();
    setParents();
  }

 private:
  static long long extractIndex(const std::string& metadata_json) {
    static const auto prefix = fmt::format("\"{}\": ", indexKey);
    auto pos = metadata_json.find(prefix);
    return (pos == std::string::npos) ? unmatchedIndex : [&]() {
      auto end = metadata_json.find(',', pos);
      end = (end == std::string::npos) ? metadata_json.size() : end;
      return std::stoll(metadata_json.substr(pos + prefix.size(), end));
    }();
  }

  std::shared_ptr<Result> lookup(const itrace_t* key) {
    if (key == nullptr) {
      return nullptr;
    }

    // First check the map.
    auto it = kineto_events_.find(key);
    if (it != kineto_events_.end()) {
      return it->second;
    }

    // Then fallback to the encoded metadata.
    const auto index = extractIndex(key ? key->metadataJson() : "");
    if (index != unmatchedIndex) {
      auto out = results_.get().at(index);
      kineto_events_[key] = out;
      return out;
    }

    // And finally give up.
    return nullptr;
  }

  void reassociate() {
    // Match profiler events with the corresponding kineto events. Kineto may
    // have moved or copied the activities, so we have to recover the
    // relationship between `libkineto::ITraceActivity` and `Result`.
    for (const auto* activity : trace_activities_) {
      TORCH_INTERNAL_ASSERT(activity != nullptr);
      auto e = lookup(activity);
      if (e != nullptr) {
        TORCH_INTERNAL_ASSERT(e->kineto_activity_ == nullptr);
        e->kineto_activity_ = dynamic_cast<const activity_t*>(activity);
      }
    }
    if (results_.get().size() != kineto_events_.size()) {
      TORCH_WARN(fmt::format(
          "Failed to recover relationship between all "
          "profiler and kineto events: "
          "{} vs. {}  reassociated.",
          results_.get().size(),
          kineto_events_.size()));
    }
  }

  static std::shared_ptr<Result> resultFromActivity(const itrace_t* activity) {
    TORCH_INTERNAL_ASSERT(activity != nullptr);

    // Kineto is inconsistent with types, so we have to cast to int32.
    torch::profiler::impl::kineto::DeviceAndResource device_and_resource{
        static_cast<int32_t>(activity->deviceId()), static_cast<int32_t>(activity->resourceId())};

    auto event = Result::create(
        activity->timestamp(),
        noTID, // Placeholder
        device_and_resource,
        ExtraFields<EventType::Kineto>{
            activity->name(),
            activity->duration(),
            static_cast<uint64_t>(activity->correlationId()),
            activity->type(),
            {/*id=*/static_cast<uint32_t>(activity->flowId()),
             /*type=*/static_cast<uint32_t>(activity->flowType()),
             /*start=*/activity->flowStart()}});

    // NB: It's tempting to set `event->kineto_activity_`; however we can only
    // guarantee that the events we passed to Kineto are of type
    // `GenericTraceActivity`. Others may derive from ITraceActivity and thus
    // are not safe to cast.
    return event;
  }

  std::shared_ptr<Result> toResult(const itrace_t* activity) {
    auto e = lookup(activity);

    // Until we are very sure that we can reassociate kineto and profiler
    // events we need to be very defensive.
    const auto type = activity->type();
    if (e == nullptr &&
        (type == libkineto::ActivityType::CPU_OP || type == libkineto::ActivityType::CPU_INSTANT_EVENT ||
         type == libkineto::ActivityType::USER_ANNOTATION || type == libkineto::ActivityType::PYTHON_FUNCTION)) {
      TORCH_WARN_ONCE(
          "Detected an event which was likely passed to kineto by the PyTorch "
          "profiler, but is not present in the set of known events: ",
          activity->name(),
          " This most likely means that Kineto has not "
          "maintained address stability for this event. Please report this to "
          "the PyTorch team.");
      return nullptr;
    }

    if (e == nullptr) {
      e = resultFromActivity(activity);
      results_.get().push_back(e);
      kineto_events_[activity] = e;
    }
    return e;
  }

  void extractEventsFromTrace() {
    for (const auto* activity : trace_activities_) {
      auto e = toResult(activity);
      const auto* linked_activity = activity->linkedActivity();
      if (e && linked_activity) {
        e->visit(c10::overloaded(
            [&](ExtraFields<EventType::Kineto>& i) { i.linked_activity_ = toResult(linked_activity); },
            [](auto&) { TORCH_INTERNAL_ASSERT(false); }));
      }
    }
  }

  void setKinetoTID(std::shared_ptr<Result>& r, std::shared_ptr<Result> parent) {
    r->visit(c10::overloaded(
        [&]([[maybe_unused]] ExtraFields<EventType::Kineto>& i) {
          TORCH_INTERNAL_ASSERT(r->start_tid_ == noTID);
          r->start_tid_ = parent ? parent->start_tid_ : at::RecordFunction::currentThreadId();
        },
        [](auto&) {}));

    for (auto& child : r->children_) {
      setKinetoTID(child, r);
    }
  }

  void setParents() {
    // First pass: Collect start events and set parent to linked event.
    ska::flat_hash_map<uint32_t, std::shared_ptr<Result>> flow_map;
    for (auto& e : results_.get()) {
      TORCH_INTERNAL_ASSERT(e != nullptr);
      e->visit(c10::overloaded(
          [&](const ExtraFields<EventType::Kineto>& i) {
            if (i.flow.type == libkineto::kLinkAsyncCpuGpu && i.flow.start) {
              auto inserted = flow_map.insert({i.flow.id, e});
#ifdef USE_ROCM
              if (inserted.second) {
                TORCH_WARN_ONCE("ROCTracer produced duplicate flow start: ", i.flow.id);
              }
#else // USE_ROCM
              TORCH_INTERNAL_ASSERT(inserted.second);
#endif // USE_ROCM
            }
            TORCH_INTERNAL_ASSERT(e->parent_.expired());
            e->parent_ = i.linked_activity_;
          },
          [](const auto&) {}));
    }

    // Second pass
    for (auto& e : results_.get()) {
      e->visit(c10::overloaded(
          [&](const ExtraFields<EventType::Kineto>& i) {
            // Flow takes priority over linked event.
            const auto it = flow_map.find(i.flow.id);
            if (it != flow_map.end() && i.flow.type == libkineto::kLinkAsyncCpuGpu && !i.flow.start) {
              e->parent_ = it->second;
            }

            // If a parent was set we have to do some bookkeeping.
            auto parent = e->parent_.lock();
            if (parent) {
              parent->children_.push_back(e);
              mark_finished(e);
            }
          },
          [](const auto&) {}));
    }

    // Set TIDs now that we have established lineage.
    for (auto& e : results_.get()) {
      if (e->parent_.expired()) {
        setKinetoTID(e, nullptr);
      }
    }
  }

  static constexpr long long unmatchedIndex = -1;
  static constexpr auto noTID = std::numeric_limits<uint64_t>::max();
  std::reference_wrapper<std::vector<std::shared_ptr<Result>>> results_;
  std::vector<const itrace_t*> trace_activities_;
  ska::flat_hash_map<const itrace_t*, std::shared_ptr<Result>> kineto_events_;
};
#else
class TransferEvents {
 public:
  template <class... Args>
  TransferEvents(Args&&...) {}
};
#endif

trace_ptr_t addKinetoEvents(
    std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    const ProfilerConfig& config) {
  using namespace torch::profiler::impl::kineto;
  passEventsToKineto(results, start_time_ns, end_time_ns, config);

  // In on demand mode kineto is directly controlled by other machinery.
  if (config.global()) {
    return nullptr;
  }

  auto trace = std::make_unique<ActivityTraceWrapper>(torch_supa::profiler::impl::kineto::stopTrace());
  TORCH_INTERNAL_ASSERT(trace || !kKinetoAvailable);
  TransferEvents transfer{results, trace};
  return trace;
}

struct ResultGreater {
  bool operator()(const result_ptr_t& a, const result_ptr_t& b) const {
    return a->endTimeNS() > b->endTimeNS();
  }
};

void set_in_tree_building(std::vector<result_ptr_t>& results, const bool value) {
  for (result_ptr_t& r : results) {
    r->visit(c10::overloaded(
        [value](ExtraFields<EventType::Vulkan>& i) { i.in_tree_building_ = value; },
        [&](auto&) {
          // pass
        }));
  }
}

void build_tree(std::vector<std::shared_ptr<Result>>& sorted_events) {
  set_in_tree_building(sorted_events, true);

  using op_fields = ExtraFields<EventType::TorchOp>;
  ska::flat_hash_map<uint64_t, std::shared_ptr<Result>> stacks;
  std::priority_queue<result_ptr_t, std::vector<result_ptr_t>, ResultGreater> end_events_;

  auto push_event = [&stacks, &end_events_](std::shared_ptr<Result>& event) {
    // Kineto builds subtrees using correlation ids and flows, so some Kineto
    // events are already marked finished before the main tree building
    // algorithm. It's fine to ignore them; the root event of these subtrees
    // not a Kineto op and will be handled normally.
    if (std::holds_alternative<ExtraFields<EventType::Kineto>>(event->extra_fields_) && event->finished_) {
      return;
    }

    TORCH_INTERNAL_ASSERT(event->parent_.expired());
    for (const auto& child : event->children_) {
      TORCH_INTERNAL_ASSERT(child->finished_);
    }
    TORCH_INTERNAL_ASSERT(!event->finished_);

    auto parent_it = stacks.find(event->start_tid_);
    if (parent_it == stacks.end()) {
      auto fwd_tid = event->visit(c10::overloaded(
          [](const op_fields& i) { return i.forward_tid_; }, [](const auto&) -> uint64_t { return 0; }));
      if (fwd_tid) {
        parent_it = stacks.find(fwd_tid);
      }
    }

    if (parent_it != stacks.end()) {
      event->parent_ = parent_it->second;
      parent_it->second->children_.push_back(event);
    }

    if (event->endTimeNS() > event->start_time_ns_) {
      stacks[event->start_tid_] = event;
      end_events_.push(event);
    } else if (event->endTimeNS() == std::numeric_limits<c10::time_t>::min()) {
      // We use min time to indicate the lack of a termination event, so if we
      // encounter such a case we don't push to `end_events_`.
      stacks[event->start_tid_] = event;
    } else {
      mark_finished(event);
    }
  };

  auto pop_event = [&stacks](std::shared_ptr<Result> event) {
    if (event->finished_) {
      // This event was marked finished by a previous `pop_event` call.
      return;
    }

    auto start_tid = event->start_tid_;
    auto frame = stacks.at(start_tid);

    while (frame.get() != event.get()) {
      TORCH_INTERNAL_ASSERT(frame != nullptr);
      mark_finished(frame);
      TORCH_INTERNAL_ASSERT(!frame->parent_.expired());
      frame = frame->parent_.lock();
    }

    mark_finished(event);
    stacks.erase(start_tid);
    auto new_frame = event->parent_.lock();
    if (new_frame != nullptr) {
      stacks[start_tid] = new_frame;
    }
  };

  // Stack replay loop.
  for (auto& event : sorted_events) {
    while (!end_events_.empty() && end_events_.top()->endTimeNS() < event->start_time_ns_) {
      pop_event(end_events_.top());
      end_events_.pop();
    }
    push_event(event);
  }

  // Cleanup remaining exit events.
  while (!end_events_.empty()) {
    pop_event(end_events_.top());
    end_events_.pop();
  }

  set_in_tree_building(sorted_events, false);
}

int64_t adjust_durations_dfs(std::shared_ptr<Result>& r) {
  if (SOFT_ASSERT(r != nullptr)) {
    int64_t original_duration = r->endTimeNS() - r->start_time_ns_;
    int64_t children_total_duration =
        std::accumulate(r->children_.begin(), r->children_.end(), 0, [](int64_t acc, std::shared_ptr<Result>& child) {
          return acc + adjust_durations_dfs(child);
        });

    if (children_total_duration > original_duration) {
      r->visit(c10::overloaded(
          [&r, &children_total_duration](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ = r->start_time_ns_ + children_total_duration;
          },
          [&children_total_duration](ExtraFields<EventType::Vulkan>& i) { i.duration_ns_ = children_total_duration; },
          []([[maybe_unused]] ExtraFields<EventType::Allocation>& _) {
            // Pass- Allocation events can't have children
          },
          [&](auto&) {
            SOFT_ASSERT(
                false,
                "unexpected event type in mobile profiler "
                "adjust_durations_dfs: ",
                r->name());
          }));
      return children_total_duration;
    }
    return original_duration;
  }
  return 0;
}

/**
 * 1) Adjust r's start time to be [new_start_time] (also adjusting end time and
      keeping duration the same)
 * 2) Recursively adjust r's children's start times, making them line up such
      that the last one ends at the same time as r
 * 3) Return r's final end time
 */
int64_t adjust_timestamps_dfs(std::shared_ptr<Result>& r, int64_t new_start_time) {
  if (SOFT_ASSERT(r != nullptr)) {
    if (r->start_time_ns_ != new_start_time) {
      // Adjust start time (keeping duration constant)
      r->visit(c10::overloaded(
          [&r, &new_start_time](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ = new_start_time + (i.end_time_ns_ - r->start_time_ns_);
          },
          []([[maybe_unused]] ExtraFields<EventType::Vulkan>& i) {
            // Pass- We don't need to manually adjust end time for Vulkan events
          },
          []([[maybe_unused]] ExtraFields<EventType::Allocation>& _) {
            // Pass- No duration or end time to adjust
          },
          [&](auto&) {
            SOFT_ASSERT(
                false,
                "unexpected event type in mobile profiler "
                "adjust_timestamps_dfs: ",
                r->name());
          }));
      r->start_time_ns_ = new_start_time;
    }
    int64_t children_total_duration =
        std::accumulate(r->children_.begin(), r->children_.end(), 0, [](int64_t acc, std::shared_ptr<Result>& child) {
          return acc + (child->endTimeNS() - child->start_time_ns_);
        });

    int64_t child_start_time = r->endTimeNS() - children_total_duration;
    for (std::shared_ptr<Result>& child : r->children_) {
      child_start_time = adjust_timestamps_dfs(child, child_start_time);
    }
  }
  return r->endTimeNS();
}

/**
 * Adjust timestamps and durations of nodes in [out] such that
 *  - Vulkan event timelines are synchronized with CPU event times
 *  - Parent event timelines fully contain their child timelines
 *  - No overlaps in timelines for nodes at the same depth
 */
void adjust_timestamps(std::vector<std::shared_ptr<Result>>& out) {
  if (out.empty()) {
    return;
  }

  int64_t min_start_time = out[0]->start_time_ns_;
  for (std::shared_ptr<Result>& r : out) {
    // Only begin traversal for root nodes.
    if (r->parent_.expired()) {
      adjust_durations_dfs(r);
      min_start_time = adjust_timestamps_dfs(
          r,
          std::max(
              r->tag() != EventType::Vulkan ? r->start_time_ns_ : std::numeric_limits<int64_t>::min(), min_start_time));
    }
  }
}
} // namespace

std::pair<std::vector<std::shared_ptr<Result>>, std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>>
RecordQueue::getRecords(
    std::function<c10::time_t(c10::approx_time_t)> time_converter,
    uint64_t start_time_ns,
    uint64_t end_time_ns) {
  auto converter = [&](c10::approx_time_t t) {
    return t == std::numeric_limits<c10::approx_time_t>::min() ? std::numeric_limits<c10::time_t>::min()
                                                               : time_converter(t);
  };

#if TORCH_VER >= TORCH_2_5_0
  // Lambda that checks that only the right side of the base intersects with
  // ev_start and ev_end
  auto right_intersection_only = [&](ProfilerStepInfo base, int64_t ev_start, int64_t ev_end) {
    return (base.start_time_ns < ev_start) && (base.end_time_ns <= ev_end && base.end_time_ns > ev_start);
  };
#endif

  std::vector<std::shared_ptr<Result>> out;
  std::vector<python_tracer::CompressedEvent> python_enters;
#if TORCH_VER >= TORCH_2_5_0
  std::vector<ProfilerStepInfo> step_info;
  long unsigned int step_idx = 0;
#endif
  for (auto& subqueue_it : sub_queues_) {
    auto& queue = *subqueue_it.second;
    auto materialize = [&](auto& events) {
      for (auto& i : events) {
        c10::time_t start_time_ns = 0;
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(i)>, ExtraFields<EventType::Backend>>) {
          start_time_ns = i.start_time_us_ * 1000;
        } else {
          start_time_ns = converter(i.start_time_);
        }
        out.emplace_back(Result::create(
            /*start_time_ns_=*/start_time_ns,
            /*start_tid_=*/queue.tid(),
            /*kineto_info_=*/queue.kineto_info(),
            /*extra_fields_=*/std::move(i)));
      }
      events.clear();
    };

    queue.torch_ops_.materialize(
        out,
#if TORCH_VER >= TORCH_2_5_0
        step_info,
#endif
        converter,
        queue.tid(),
        queue.kineto_info());
    materialize(queue.backend_events_);
    // materialize_vulkan(
    //     out, queue.vulkan_events_, converter, queue.tid(),
    //     queue.kineto_info());
    for (auto& i : queue.allocations_) {
      out.emplace_back(Result::create(
          /*start_time_ns_=*/converter(i.start_time_),
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/ExtraFields<EventType::Allocation>(i)));
    }
    queue.allocations_.clear();
    materialize(queue.ooms_);

    for (auto& i : queue.py_calls_) {
      python_enters.push_back({i.first, queue.tid(), queue.kineto_info(), converter(i.second)});
    }
  }

  if (python_tracer_) {
    std::vector<std::shared_ptr<torch::profiler::impl::Result>> ev;
    try {
      ev = python_tracer_->getEvents(converter, python_enters, static_cast<c10::time_t>(end_time_ns));
    } catch (std::exception&) {
      // Normally addKinetoEvents() below will stop the trace - but if an
      // exception happens here then the events will never be stopped and future
      // runs will be broken - so make sure to stopTrace() if we see an
      // exception.
      torch_supa::profiler::impl::kineto::stopTrace();
      throw;
    }
#if TORCH_VER >= TORCH_2_5_0
    // Placeholder for if we run out of ProfilerStep annotations
    ProfilerStepInfo defaultStep = {LLONG_MAX, LLONG_MAX, 0};
    ProfilerStepInfo step = step_idx < step_info.size() ? step_info[step_idx] : defaultStep;
#endif
    for (const auto& i : ev) {
#if TORCH_VER >= TORCH_2_5_0
      // Only adjust timestamps if experimental config is enabled
      if (config_.experimental_config.adjust_profiler_step) {
        // If event has start time after step end time we can continue to the
        // next step
        while (i->start_time_ns_ > step.end_time_ns) {
          step_idx++;
          step = step_idx < step_info.size() ? step_info[step_idx] : defaultStep;
        }
        // If Step annotation starts before event and ends before event ends
        // with intersection then we move the lefthand side of the step
        // annotation to the event start time
        if (right_intersection_only(step, i->start_time_ns_, i->endTimeNS())) {
          // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
          auto const& currStepRes = out[step.out_idx];
          currStepRes->start_time_ns_ = i->start_time_ns_ + 1;
          step_idx++;
          step = step_idx < step_info.size() ? step_info[step_idx] : defaultStep;
        }
      }
#endif
      out.push_back(i);
    }
    python_tracer_.reset();
  }

  if (config_.experimental_config.adjust_timestamps) {
    std::stable_sort(
        out.begin(), out.end(), [](const auto& a, const auto& b) { return a->start_time_ns_ < b->start_time_ns_; });
    build_tree(out);
    adjust_timestamps(out);
    for (auto& r : out) {
      r->parent_.reset();
      // Reset these so that second build_tree can happen
      r->finished_ = false;
      r->children_.clear();
    }
  }

  auto trace = addKinetoEvents(out, start_time_ns, end_time_ns, config_);

  std::stable_sort(
      out.begin(), out.end(), [](const auto& a, const auto& b) { return a->start_time_ns_ < b->start_time_ns_; });

  if (config_.report_input_shapes && config_.profile_memory) {
    calculateUniqueTensorIDs(out);
  }

  build_tree(out);
  return {out, std::move(trace)};
}

/*
implement symbols that are not exported by torch_cpu.so. keep them hidden as
well for safety.
*/
namespace linux_perf {
#define HIDDEN __attribute__((visibility("hidden")))
HIDDEN PerfEvent::~PerfEvent() {
  if (fd_ > -1) {
    close(fd_);
  }
  fd_ = -1; // poison
}

} // namespace linux_perf
} // namespace impl
} // namespace profiler

} // namespace torch
