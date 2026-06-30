/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/MapAllocator.h>
#include <torch_supa/csrc/ipc/SupaIPCTypes.h>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace torch_supa {
namespace supa {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared SUPA tensors released. See Note [Sharing SUPA tensors]";
    warned = true;
  }
}

struct SupaIPCGlobalEntities {
  // This class is used as a singleton (see supa_ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trvial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<SupaIPCRefCountersFile>> ref_counters_files_;
  std::shared_ptr<SupaIPCRefCountersFile> next_available_ref_counters_file_;
  SupaIPCSentDataLimbo SupaIPCSentDataLimbo_;
  SupaIPCGlobalEntities() {
    alive = true;
  }
  SupaIPCGlobalEntities(const SupaIPCGlobalEntities&) = delete;
  SupaIPCGlobalEntities(SupaIPCGlobalEntities&&) = delete;
  SupaIPCGlobalEntities& operator=(const SupaIPCGlobalEntities&) = delete;
  SupaIPCGlobalEntities& operator=(SupaIPCGlobalEntities&&) = delete;
  ~SupaIPCGlobalEntities() {
    SupaIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    alive = false;
  }
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ && next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool SupaIPCGlobalEntities::alive = false;
SupaIPCGlobalEntities supa_ipc_global_entities;

SupaIPCSentDataLimbo::~SupaIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool SupaIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<SupaIPCSentData>> reset_blocks;
  { // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<SupaIPCSentData>> kept_blocks;
    for (auto& sd : shared_blocks_) {
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    shared_blocks_ = std::move(kept_blocks);
  }
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void SupaIPCSentDataLimbo::add(std::unique_ptr<SupaIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > SUPA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO && !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over " << SUPA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t SupaIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void SupaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<SupaIPCSentData> sent_data(static_cast<SupaIPCSentData*>(ptr));
  if (!SupaIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    supa_ipc_global_entities.SupaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  supa_ipc_global_entities.SupaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!SupaIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(supa_ipc_global_entities.ref_counters_mutex_);
  auto& map = supa_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

} // namespace

SupaIPCSentData::SupaIPCSentData(std::string handle, uint64_t offset, uint64_t* counter_ptr, at::Device device)
    : handle_(std::move(handle)), offset_(offset), counter_ptr_(counter_ptr), device_(device) {
#if !defined(USE_ROCM)
  // SUPA have the unofficial limit on the number of recorded blocking
  // interprocess events, to prevent using of all events, we are switching to
  // StreamSync before limit reached.
  //
  //  ```python
  //  import torch
  //  a = [ torch.supa.Event(
  //      enable_timing=False, blocking=True, interprocess=True) for i in
  //      range(30000) ]
  //  [i.record() for i in a]
  //  ```
  //
  if (supa_ipc_global_entities.sync_events_used_.load() < SUPA_IPC_MAXIMUM_EVENTS_TO_USE) {
    // TODO: More efficient would be to create event inside of main thread (at
    // the moment of the queue.put). The reason this is more efficient is
    // because the main thread may have queued extra work on the stream, which
    // this event will consequently wait for (uselessly).
    supa_ipc_global_entities.sync_events_used_++;
    C10_SUPA_CHECK(
        supaEventCreateWithFlags(&event_, supaEventDisableTiming | supaEventInterprocess | supaEventBlockingSync));
    C10_SUPA_CHECK(supaEventRecord(event_, c10::supa::getCurrentSUPAStream(device.index()).stream()));
    event_sync_required_ = true;
  } else {
    auto stream = c10::supa::getCurrentSUPAStream(device.index());
    c10::supa::stream_synchronize(stream.stream());
    event_ = nullptr;
    event_sync_required_ = false;
  }
#else
  // supaIpcGetEventHandle with HIP is not supported, so we have to sync
  // stream instead of passing event
  auto stream = c10::supa::getCurrentSUPAStream(device.index());
  c10::supa::stream_synchronize(stream.stream());
  event_sync_required_ = false;
#endif
}

SupaIPCSentData::~SupaIPCSentData() {
  ReturnRefCounter(handle_, offset_);
#if !defined(USE_ROCM)
  try {
    if (event_sync_required_) {
      c10::supa::SUPAGuard device_guard(device_.index());
      C10_SUPA_CHECK(supaEventDestroy(event_));
      if (!SupaIPCGlobalEntities::alive) {
        return;
      }
      supa_ipc_global_entities.sync_events_used_--;
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (...) { /* No throw */
  }
#endif
}

uint64_t SupaIPCSentData::counter_value() const {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(supa_ipc_global_entities.ref_counters_mutex_);
    if (!supa_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(), flags, sizeof(int64_t) * SUPA_IPC_REF_COUNTER_FILE_SIZE, nullptr);
      auto rc =
          std::make_shared<SupaIPCRefCountersFile>(ref_counter_handle, SUPA_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      supa_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      supa_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  supa_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto* sent_data = new SupaIPCSentData(
      supa_ipc_global_entities.next_available_ref_counters_file_->handle(),
      supa_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      supa_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  supa_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!supa_ipc_global_entities.next_available_ref_counters_file_->have_offsets()) {
    supa_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, SupaIPCSentDataDelete, device);
}

bool SupaIPCCollect() {
  if (!SupaIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = supa_ipc_global_entities.SupaIPCSentDataLimbo_.collect();
  if (supa_ipc_global_entities.SupaIPCSentDataLimbo_.size() == 0) {
    supa_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace supa
} // namespace torch_supa

namespace c10::supa {
namespace SUPACachingAllocator {
namespace {
REGISTER_FREE_MEMORY_CALLBACK("supa_ipc_collect", SupaIPCCollectCallback)
} // namespace
} // namespace SUPACachingAllocator
} // namespace c10::supa
