/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/MemPool.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"

namespace c10::supa {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10::supa::MemPool().
//
// uuid_ is incremented when SUPAGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or SUPAGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    std::shared_ptr<SUPACachingAllocator::SUPAAllocator> allocator,
    bool is_user_created,
    bool use_on_oom,
    bool no_split)
    : allocator_(allocator.get()), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::supa::current_device();
  SUPACachingAllocator::createOrIncrefPool(device_, id_, std::move(allocator));
  if (use_on_oom) {
    SUPACachingAllocator::setUseOnOOM(device_, id_);
  }
  if (no_split) {
    SUPACachingAllocator::setNoSplit(device_, id_);
  }
}

MemPool::~MemPool() {
  // TORCH_INTERNAL_ASSERT(use_count() == 1);
  // We used to assert that TORCH_INTERNAL_ASSERT(use_count() == 1);
  // However, this assertion is not true if a memory pool is shared
  // with a supa graph. That SUPAGraph will increase the use count
  // until it is reset.
  SUPACachingAllocator::releasePool(device_, id_);
  auto ctx = MemPoolContext(this);
  c10::supa::SUPACachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id() {
  return id_;
}

SUPACachingAllocator::SUPAAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return SUPACachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() const {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

// Note that active_mempool_ is a global variable here
// and not inside MemPoolContext class, because in windows we
// can't use __declspec(dllexport) and __declspec(thread)
// together: https://stackoverflow.com/a/50967977
static thread_local MemPool* active_mempool_ = nullptr;

MemPoolContext::MemPoolContext(MemPool* mempool) : prev_mempool_(active_mempool_) {
  active_mempool_ = mempool;
}

MemPoolContext::~MemPoolContext() {
  active_mempool_ = prev_mempool_;
}

MemPool* MemPoolContext::getActiveMemPool() {
  return active_mempool_;
}

} // namespace c10::supa
