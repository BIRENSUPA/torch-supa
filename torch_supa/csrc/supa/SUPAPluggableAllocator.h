/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/Allocator.h>
// #include <c10/core/CachingDeviceAllocator.h>
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

#include <mutex>

namespace torch_supa::supa::SUPAPluggableAllocator {

using MallocFuncType = void*(size_t, int, supaStream_t);
using FreeFuncType = void(void*, size_t, int, supaStream_t);

using namespace c10::supa;
// A SUPAPluggableAllocatorDeleterContext object is used as the `ctx`
// argument for DataPtr. We need context because a user can use
// multiple allocators in the same PyTorch program, and
// the allocators can have different free functions, such as:
// free, supaFree, supaFreeAsync, ncclMemFree etc.
struct TORCH_SUPA_API SUPAPluggableAllocatorDeleterContext {
  explicit SUPAPluggableAllocatorDeleterContext(
      std::function<FreeFuncType> free_fn,
      void* data,
      size_t size,
      int device,
      supaStream_t stream);

  void free();

 private:
  std::function<FreeFuncType> free_fn_;
  void* data_;
  size_t size_;
  int device_;
  supaStream_t stream_;
};

using streamType = c10::supa::SUPAStream;

TORCH_SUPA_API std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> getCurrentAllocator();
TORCH_SUPA_API std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn);
TORCH_SUPA_API void changeCurrentAllocator(
    const std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator>& allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(size_t size, c10::DeviceIndex device_idx, supaStream_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  supaStream_t stream;
};

struct TORCH_SUPA_API SUPAPluggableAllocator : public c10::supa::SUPACachingAllocator::SUPAAllocator {
  SUPAPluggableAllocator(std::function<MallocFuncType> alloc_fn, std::function<FreeFuncType> free_fn);
  // #ifndef NO_SUPA_RT_HEADER
  //   using MallocFuncTypeVoid = void*(size_t, int, void*);
  //   using FreeFuncTypeVoid = void(void*, size_t, int, void*);
  //   SUPAPluggableAllocator(std::function<MallocFuncTypeVoid> alloc_fn, std::function<FreeFuncTypeVoid> free_fn);
  // #endif

  SUPAPluggableAllocator(SUPAPluggableAllocator& other);
  SUPAPluggableAllocator(SUPAPluggableAllocator&& other) = delete;
  SUPAPluggableAllocator& operator=(const SUPAPluggableAllocator& other) = delete;
  SUPAPluggableAllocator& operator=(SUPAPluggableAllocator&& other) = delete;
  ~SUPAPluggableAllocator() override = default;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(std::function<void(void* ptr, supaStream_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool(
      std::function<void(int, MempoolId_t, std::function<bool(supaStream_t)>)> capture_begin_fn);

  void set_end_allocate_to_pool_fn(std::function<void(int, MempoolId_t)> capture_about_to_end_fn);

  void set_release_pool(std::function<void(int, MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, supaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, supaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  double getMemoryFraction(c10::DeviceIndex device) override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  void emptyCache(c10::supa::MempoolId_t mempool_id = {0, 0}) override;
  void enable(bool /*value*/) override {}
  bool isEnabled() const override {
    return true;
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr& /*ptr*/, streamType stream) override;

  c10::supa::SUPACachingAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::supa::SUPACachingAllocator::SnapshotInfo snapshot(c10::supa::MempoolId_t mempool) override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(supaStream_t)> /*filter*/) override;
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) override;
  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  c10::supa::SUPACachingAllocator::ShareableHandle shareIpcHandle(void* /*ptr*/) override;
  void recordHistory(
      bool enabled,
      c10::supa::SUPACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::supa::SUPACachingAllocator::RecordContext when) override;
  void attachOutOfMemoryObserver(c10::supa::SUPACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(c10::supa::SUPACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override;
  c10::supa::SUPACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState> pps) override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override;
  supaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      supaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<MallocFuncType> alloc_fn_;
  std::function<FreeFuncType> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, supaStream_t stream)> record_stream_fn_;
  std::function<void(int, MempoolId_t, std::function<bool(supaStream_t)>)> begin_allocate_to_pool_fn_;
  std::function<void(int, MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch_supa::supa::SUPAPluggableAllocator
