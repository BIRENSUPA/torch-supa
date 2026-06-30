/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/Allocator.h>
#include <torch_supa/csrc/core/supa/SUPACachingAllocator.h>

#include <memory>

namespace c10::supa {

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the SUPACachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct C10_SUPA_API MemPool {
  MemPool(
      std::shared_ptr<SUPACachingAllocator::SUPAAllocator> allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  SUPACachingAllocator::SUPAAllocator* allocator();
  int use_count();
  c10::DeviceIndex device() const;
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  SUPACachingAllocator::SUPAAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};

// MemPoolContext holds the currently active pool and stashes the previous
// pool. On deletion it makes the previous pool active.
struct C10_SUPA_API MemPoolContext {
  MemPoolContext(MemPool* mempool);
  MemPoolContext(const MemPoolContext&) = delete;
  MemPoolContext(MemPoolContext&&) = delete;
  MemPoolContext& operator=(const MemPoolContext&) = delete;
  MemPoolContext& operator=(MemPoolContext&&) = delete;

  ~MemPoolContext();

  // getActiveMemPool() can be used to get the currently active pool.
  // For instance: in SUPACachingAllocator, we can route allocations
  // to a user provided allocator, by doing:
  //
  //  auto active_pool = MemPoolContext::getActiveMemPool();
  //  if (active_pool && active_pool->allocator()) {
  //    ptr = active_pool->allocator()->raw_alloc(size);
  //  }
  //
  static MemPool* getActiveMemPool();

 private:
  MemPool* prev_mempool_;
};

} // namespace c10::supa