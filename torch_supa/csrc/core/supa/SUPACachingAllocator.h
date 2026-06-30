/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#if TORCH_VER >= TORCH_2_5_0
#include <c10/core/CachingDeviceAllocator.h>
#else
#include <c10/core/Allocator.h>
#endif
#if TORCH_VER >= TORCH_2_2_0
#include <c10/util/ApproximateClock.h>
#endif
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

std::string format_size(uint64_t size);

namespace c10::supa {

using CaptureId_t = unsigned long long;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

namespace SUPACachingAllocator {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
 public:
  FreeMemoryCallback() = default;
  FreeMemoryCallback(const FreeMemoryCallback&) = delete;
  FreeMemoryCallback(FreeMemoryCallback&&) = delete;
  FreeMemoryCallback& operator=(const FreeMemoryCallback&) = delete;
  FreeMemoryCallback& operator=(FreeMemoryCallback&&) = delete;
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeSUPAMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) C10_REGISTER_CLASS(FreeSUPAMemoryCallbacksRegistry, name, __VA_ARGS__);

// (Ascend): Turn this into an honest to goodness class. I briefly attempted to
// do this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (SUPACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THNCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.
struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

#if TORCH_VER < TORCH_2_5_0
typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;
// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from SUPAMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via SUPAFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to SUPA malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to SUPA after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};
#else
using c10::CachingDeviceAllocator::DeviceStats;
#endif

typedef std::shared_ptr<c10::GatheredContext> (*CreateContextFn)(void);

// Struct containing info of an allocation block (i.e. a fractional part of a
// supaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

// Struct containing info of a memory segment (i.e. one contiguous supaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  supaStream_t stream = nullptr;
  int64_t total_size = 0;
  int64_t requested_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0};
  std::vector<BlockInfo> blocks;
  std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

struct AllocatorState {
  AllocatorState() = default;
  AllocatorState(const AllocatorState&) = delete;
  AllocatorState(AllocatorState&&) = delete;
  AllocatorState& operator=(const AllocatorState&) = delete;
  AllocatorState& operator=(AllocatorState&&) = delete;
  virtual ~AllocatorState() = default;
};

#if TORCH_VER >= TORCH_2_2_0
union trace_time_ {
  time_t t_;
  approx_time_t approx_t_;
};
#endif

struct TraceEntry {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via
                    // record_stream This event is generated when a free
                    // actually completes.
    SEGMENT_ALLOC, // a call to AclrtMalloc to get more memory from the OS
    SEGMENT_FREE, // a call to aclrtFree to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // a call to AclrtMapMem (used with expandable_segments)
    SEGMENT_UNMAP, // unmap part of a segment (used with expandable
                   // segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to
              // trace events
    OOM, // the allocator threw an OutOfMemoryError (addr_ is the amount of
         // free bytes reported by supa)
    WORKSPACE_SNAPSHOT
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      int64_t addr,
      size_t size,
      supaStream_t stream,
      MempoolId_t mempool,
#if TORCH_VER >= TORCH_2_2_0
      approx_time_t time,
#endif
      std::shared_ptr<c10::GatheredContext> context = nullptr)
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size),
        mempool_(std::move(mempool)) {
#if TORCH_VER >= TORCH_2_2_0
    time_.approx_t_ = time;
#endif
  }
  Action action_;
  int device_;
  int64_t addr_; // for OOM, this is the amount of free bytes reported by supa
  std::shared_ptr<c10::GatheredContext> context_;
  supaStream_t stream_;
  size_t size_;
  MempoolId_t mempool_;
#if TORCH_VER >= TORCH_2_2_0
  trace_time_ time_{};
#endif
};

struct AllocatorConfigInfo {
  double garbage_collection_threshold;
  size_t max_split_size;
  size_t pinned_num_register_threads;
  bool expandable_segments;
  bool release_lock_on_malloc;
  bool pinned_use_host_register;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
  AllocatorConfigInfo config_metadata{};
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<at::DataPtr> dataptrs_allocd;
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

using OutOfMemoryObserver =
    std::function<void(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free)>;

using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

struct ShareableHandle {
  ptrdiff_t offset;
  std::string handle;
};

#if TORCH_VER >= TORCH_2_9_0
class SUPAAllocator : public c10::DeviceAllocator {
#else
class SUPAAllocator : public c10::Allocator {
#endif
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, supaStream_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
#if TORCH_VER < TORCH_2_9_0
  virtual bool initialized() = 0;
#endif
  virtual double getMemoryFraction(c10::DeviceIndex device) = 0;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
#if TORCH_VER < TORCH_2_9_0
  virtual void emptyCache(MempoolId_t mempool_id = {0, 0}) = 0;
#endif
  virtual void enable(bool value) = 0;
  virtual bool isEnabled() const = 0;
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const c10::DataPtr& ptr, c10::supa::SUPAStream stream) = 0;
#if TORCH_VER <= TORCH_2_2_0
  virtual DeviceStats getDeviceStats(int device) = 0;
  virtual void resetAccumulatedStats(int device) = 0;
  virtual void resetPeakStats(int device) = 0;
#elif TORCH_VER < TORCH_2_9_0
  virtual DeviceStats getDeviceStats(c10::DeviceIndex device) = 0;
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;
#endif
#if TORCH_VER >= TORCH_2_9_0
  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    SUPAStream supa_stream = SUPAStream(stream);
    recordStream(ptr, supa_stream);
  }
#endif
  virtual SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) = 0;
  // SUPAGraph interactions
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(supaStream_t)> filter) = 0;
  virtual void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  virtual int getPoolUseCount(c10::DeviceIndex /*device*/, MempoolId_t /*mempool_id*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support getPoolUseCount. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void createOrIncrefPool(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      std::shared_ptr<SUPAAllocator> allocator = nullptr) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support createOrIncrefPool. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setUseOnOOM. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setNoSplit. "
        "If you need it, please file an issue describing your use case.");
  }

  virtual void ensureExistsAndIncrefPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support ensureExistsAndIncrefPool. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual bool checkPoolLiveAllocations(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      const std::unordered_set<void*>& /*expected_live_allocations*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual ShareableHandle shareIpcHandle(void* ptr) = 0;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // Attached AllocatorTraceTracker callbacks will be called while the
  // per-device allocator lock is held. Any additional locks taken from within
  // the callback must be proven to always have the lock order that never
  // triggers a deadlock. In particular, Python's GIL may be held when
  // calling the allocator so it is unsafe to try to acquire the GIL in this
  // callback.
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) = 0;

  virtual supaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      supaStream_t stream,
      bool p2p_enabled) = 0;
  virtual std::shared_ptr<AllocatorState> getCheckpointState(c10::DeviceIndex device, MempoolId_t id) = 0;
  virtual CheckpointDelta setCheckpointPoolState(c10::DeviceIndex device, std::shared_ptr<AllocatorState> pps) = 0;
  virtual std::string name() = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in SUPACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
C10_SUPA_API extern std::atomic<SUPAAllocator*> allocator;

inline SUPAAllocator* get() {
  return allocator.load();
}

// Called directly by clients.
inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, supaStream_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void init(int device_count) {
  return get()->init(device_count);
}

inline double getMemoryFraction(c10::DeviceIndex device) {
  return get()->getMemoryFraction(device);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

C10_SUPA_API inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  get()->emptyCache(mempool_id);
}

inline void enable(bool value) {
  return get()->enable(value);
}

inline bool isEnabled() {
  return get()->isEnabled();
}

inline void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) {
  return get()->cacheInfo(dev_id, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const c10::DataPtr& ptr, c10::supa::SUPAStream stream) {
  return get()->recordStream(ptr, stream);
}

#if TORCH_VER <= TORCH_2_2_0
inline DeviceStats getDeviceStats(int device) {
#else
inline DeviceStats getDeviceStats(c10::DeviceIndex device) {
#endif
  return get()->getDeviceStats(device);
}

#if TORCH_VER <= TORCH_2_2_0
inline void resetAccumulatedStats(int device) {
#else
inline void resetAccumulatedStats(c10::DeviceIndex device) {
#endif
  return get()->resetAccumulatedStats(device);
}

#if TORCH_VER <= TORCH_2_2_0
inline void resetPeakStats(int device) {
#else
inline void resetPeakStats(c10::DeviceIndex device) {
#endif
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) {
  return get()->snapshot(mempool_id);
}

inline std::shared_ptr<AllocatorState> getCheckpointState(c10::DeviceIndex device, MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(c10::DeviceIndex device, std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

// SUPAGraph interactions
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(supaStream_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) {
  return get()->recordHistory(enabled, context_recorder, alloc_trace_max_entries, when);
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(observer);
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}

inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::shared_ptr<SUPAAllocator> allocator_ptr = nullptr) {
  get()->createOrIncrefPool(device, mempool_id, std::move(allocator_ptr));
}
inline void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setUseOnOOM(device, mempool_id);
}
inline void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setNoSplit(device, mempool_id);
}

inline void ensureExistsAndIncrefPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->ensureExistsAndIncrefPool(device, mempool_id);
}

inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->getPoolUseCount(device, mempool_id);
}

// Not part of SUPA_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

inline ShareableHandle shareIpcHandle(void* ptr) {
  return get()->shareIpcHandle(ptr);
}

inline std::string name() {
  return get()->name();
}

inline supaError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    supaStream_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

inline void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

} // namespace SUPACachingAllocator
} // namespace c10::supa
