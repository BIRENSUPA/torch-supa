/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <mutex>
#include <unordered_map>
#include <utility>
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

#include <torch_supa/csrc/core/supa/SUPAFunctions.h>
#include "torch_supa/csrc/supa/SUPAPluggableAllocator.h"

namespace torch_supa::supa::SUPAPluggableAllocator {

SUPAPluggableAllocatorDeleterContext::SUPAPluggableAllocatorDeleterContext(
    std::function<FreeFuncType> free_fn,
    void* data,
    size_t size,
    int device,
    supaStream_t stream)
    : free_fn_(std::move(free_fn)), data_(data), size_(size), device_(device), stream_(stream) {}

void SUPAPluggableAllocatorDeleterContext::free() {
  free_fn_(data_, size_, device_, stream_);
  delete this;
}

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata() : size(0), device_idx(-1), stream{} {}

_AllocationMetadata::_AllocationMetadata(size_t size, c10::DeviceIndex device_idx, supaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
SUPAPluggableAllocator::SUPAPluggableAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

// SUPAPluggableAllocator::SUPAPluggableAllocator(
//     std::function<MallocFuncTypeVoid> alloc_fn,
//     std::function<FreeFuncTypeVoid> free_fn) {
//   auto streamalloc_fn = [alloc_fn](size_t size, int device, void* stream) -> void* {
//     return alloc_fn(size, device, reinterpret_cast<supaStream_t>(stream));
//   };
//   auto streamfree_fn = [free_fn](void* ptr, size_t size, int device, void* stream) {
//     free_fn(ptr, size, device, reinterpret_cast<supaStream_t>(stream));
//   };
//   alloc_fn_ = std::move(streamalloc_fn);
//   free_fn_ = std::move(streamfree_fn);
// }

SUPAPluggableAllocator::SUPAPluggableAllocator(SUPAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void SUPAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void SUPAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void SUPAPluggableAllocator::set_memory_fraction_fn(std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void SUPAPluggableAllocator::set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void SUPAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, supaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void SUPAPluggableAllocator::set_begin_allocate_to_pool(
    std::function<void(int, MempoolId_t, std::function<bool(supaStream_t)>)> capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void SUPAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void SUPAPluggableAllocator::set_release_pool(std::function<void(int, MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* SUPAPluggableAllocator::malloc(size_t size, c10::DeviceIndex device, supaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr SUPAPluggableAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));
  supaStream_t stream = c10::supa::getCurrentSUPAStream(device);
  void* r = this->malloc(size, device, stream);
  auto* ctx = new SUPAPluggableAllocatorDeleterContext(free_fn_, r, size, device, stream);
  c10::DataPtr data_ptr = {r, ctx, raw_deleter(), c10::Device(c10::kPrivateUse1, device)};
  return data_ptr;
}

c10::DeleterFnPtr SUPAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* SUPAPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));
  supaStream_t stream = c10::supa::getCurrentSUPAStream(device);
  return malloc(nbytes, device, stream);
}

void* SUPAPluggableAllocator::raw_alloc_with_stream(size_t nbytes, supaStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void SUPAPluggableAllocator::raw_delete(void* ptr) {
  supaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void SUPAPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool SUPAPluggableAllocator::initialized() {
  return initialized_;
}

double SUPAPluggableAllocator::getMemoryFraction(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support getMemoryFraction. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::setMemoryFraction(double fraction, c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void SUPAPluggableAllocator::emptyCache(/*unused*/ c10::supa::MempoolId_t mempool_id) {
  if (reset_fn_) {
    return reset_fn_();
  }
}

void SUPAPluggableAllocator::cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* SUPAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  }
  return ptr;
}

void SUPAPluggableAllocator::recordStream(const c10::DataPtr& ptr, streamType stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::supa::SUPACachingAllocator::DeviceStats SUPAPluggableAllocator::getDeviceStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::supa::SUPACachingAllocator::SnapshotInfo SUPAPluggableAllocator::snapshot(c10::supa::MempoolId_t mempool_id) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

c10::supa::SUPACachingAllocator::ShareableHandle SUPAPluggableAllocator::shareIpcHandle(void* ptr) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> SUPAPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// SUPAGraph interactions
void SUPAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(supaStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void SUPAPluggableAllocator::endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void SUPAPluggableAllocator::releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void SUPAPluggableAllocator::recordHistory(
    bool enabled,
    c10::supa::SUPACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::supa::SUPACachingAllocator::RecordContext when) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::attachOutOfMemoryObserver(c10::supa::SUPACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::attachAllocatorTraceTracker(
    c10::supa::SUPACachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState> SUPAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    at::supa::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::supa::SUPACachingAllocator::CheckpointDelta SUPAPluggableAllocator::setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "SUPAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void SUPAPluggableAllocator::enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  c10::supa::SUPAGuard device_guard(dev);
  supaError_t err = supaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == supaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)supaGetLastError();
  } else {
    C10_SUPA_CHECK(err);
  }
}

supaError_t SUPAPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    supaStream_t stream,
    bool p2p_enabled) {
  return supaMemcpyAsync(dst, src, count, supaMemcpyDeviceToDevice, stream);
}

std::string SUPAPluggableAllocator::name() {
  return "pluggable";
}

void SUPAPluggableAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  C10_SUPA_CHECK(supaMemcpy(dest, src, count, supaMemcpyKind::supaMemcpyDeviceToDevice));
}

std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> current_custom_allocator;

std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn) {
  std::shared_ptr<SUPAPluggableAllocator> allocator(
      new SUPAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(const std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator>& allocator) {
  TORCH_CHECK(
      !c10::supa::SUPACachingAllocator::allocator.load()->initialized(), "Can't swap an already initialized allocator");
  c10::supa::SUPACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ctx) {
  reinterpret_cast<SUPAPluggableAllocatorDeleterContext*>(ctx)->free();
}

} // namespace torch_supa::supa::SUPAPluggableAllocator
