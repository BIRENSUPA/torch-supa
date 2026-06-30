/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <runtime/supa_runtime_api.h>
#include <torch_supa/csrc/core/supa/SUPACachingAllocator.h>
#include <torch_supa/csrc/core/supa/SUPAException.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <cstddef>
namespace torch_supa {
namespace supa {

TORCH_SUPA_API bool SupaIPCCollect();

struct SupaIPCReceivedData final {
  SupaIPCReceivedData() = default;
  explicit SupaIPCReceivedData(std::shared_ptr<void> shared_ptr) : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct SupaIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  supaEvent_t event_; // Sync cuEventDestroy
  bool event_sync_required_;
  at::Device device_;

  SupaIPCSentData(std::string handle, uint64_t offset, uint64_t* counter_ptr, at::Device device);
  SupaIPCSentData(const SupaIPCSentData&) = delete;
  SupaIPCSentData& operator=(const SupaIPCSentData&) = delete;
  SupaIPCSentData(SupaIPCSentData&&) = delete;
  SupaIPCSentData& operator=(SupaIPCSentData&&) = delete;
  ~SupaIPCSentData();

  uint64_t counter_value() const;
  std::string handle() const {
    return handle_;
  }
  uint64_t offset() const {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

TORCH_SUPA_API at::DataPtr GetNewRefCountedSentData(void* data, at::Device device);

namespace {

inline constexpr int64_t SUPA_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t SUPA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
// This was determined empirically that SUPA (v10.1 and below) have the limit
// on the number of recorded blocking interprocess events. It is around ~22,000.
// And to give us leeway, we picked 1000 as it gives us enough events to share
// tensors effectively.
inline constexpr int64_t SUPA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct SupaIPCSentDataLimbo final {
  SupaIPCSentDataLimbo() = default;
  SupaIPCSentDataLimbo(const SupaIPCSentDataLimbo&) = delete;
  SupaIPCSentDataLimbo& operator=(const SupaIPCSentDataLimbo&) = delete;
  SupaIPCSentDataLimbo(SupaIPCSentDataLimbo&&) = delete;
  SupaIPCSentDataLimbo& operator=(SupaIPCSentDataLimbo&&) = delete;
  ~SupaIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<SupaIPCSentData> shared_block);
  uint64_t size();

 private:
  // TODO: Can be changed to FIFO in order to avoid full traverse on every
  // collect()
  std::vector<std::unique_ptr<SupaIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct SupaIPCRefCountersFile final {
  SupaIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),

        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() const {
    return next_offset_ < size_;
  }

  bool offsets_in_use() const {
    return used_slots_;
  }

  uint64_t get_offset() const {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

} // namespace
} // namespace supa
} // namespace torch_supa

namespace c10::supa {
namespace SUPACachingAllocator {
namespace {
class SupaIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch_supa::supa::SupaIPCCollect();
  }
};
} // namespace
} // namespace SUPACachingAllocator
} // namespace c10::supa
