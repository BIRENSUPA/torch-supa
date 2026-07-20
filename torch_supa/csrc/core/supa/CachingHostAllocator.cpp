/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <c10/core/thread_pool.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/llvmMathExtras.h>
#include <supa_runtime.h>

#include "torch_supa/csrc/core/supa/CachingHostAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAAllocatorConfig.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAEvent.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace at::supa {
namespace {

struct BlockSize {
  size_t size_{0};
  void* ptr_{nullptr};
};

struct Block {
  size_t size_{0};
  void* ptr_{nullptr};

  std::mutex mutex_;
  bool allocated_{false};
  size_t event_count_{0};
  ska::flat_hash_set<c10::supa::SUPAStream> streams_;
};

// Note: supaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling supaEventCreate/supaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<c10::supa::SUPAEvent, std::function<void(c10::supa::SUPAEvent*)>>;
  EventPool() : pools_(c10::supa::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](c10::supa::SUPAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<c10::supa::SUPAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(std::make_unique<c10::supa::SUPAEvent>(supaEventDisableTiming).release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<c10::supa::SUPAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// Used for heterogenous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  // Transparent overloads
  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }
  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class SUPAHostAllocator {
 public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events();

    // First, try to allocate from the free list
    {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      auto it = free_list_.lower_bound(BlockSize{size, nullptr});
      if (it != free_list_.end()) {
        auto* block = *it;
        block->allocated_ = true;
        free_list_.erase(it);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }
    }
    // Then, create a new block.
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index = c10::supa::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    size_t roundSize = llvm::PowerOf2Ceil(size);
    void* ptr = nullptr;
    if (c10::supa::SUPACachingAllocator::SUPAAllocatorConfig::pinned_use_supa_host_register()) {
      allocWithSupaHostRegister(&ptr, roundSize);
    } else {
      // Use supaHostAlloc for allocating pinned memory (global lock in driver)
      C10_SUPA_CHECK(supaHostAlloc(&ptr, roundSize, supaHostAllocDefault));
    }

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* block = new Block();
    block->size_ = roundSize;
    block->ptr_ = ptr;
    block->allocated_ = true;

    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      blocks_.insert(block);
      ptr_to_block_.insert({block->ptr_, block});
    }
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc,
    // and thus we do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<Block*>(ctx);

    c10::optional<std::vector<EventPool::Event>> events;
    ska::flat_hash_set<c10::supa::SUPAStream> streams;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        block->event_count_ += block->streams_.size();
        // Move out streams to avoid holding the mutex during event recording
        streams = std::move(block->streams_);
        block->streams_.clear();
      }
    }

    // Event recording must be done outside the mutex to avoid potential
    // deadlocks (e.g., when Python GIL is involved)
    for (auto stream : streams) {
      record_stream(events, stream);
    }

    if (!events) {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      std::lock_guard<std::mutex> g(supa_events_mutex_);
      for (auto&& event : *events) {
        supa_events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, c10::Stream s) {
    auto stream = c10::supa::SUPAStream(s);
    auto* block = reinterpret_cast<Block*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Release cached events from the event pool.
    event_pool_.empty_cache();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutex and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> gf(free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

    std::vector<Block*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      if (c10::supa::SUPACachingAllocator::SUPAAllocatorConfig::pinned_use_supa_host_register()) {
        void* ptr = block->ptr_;
        C10_SUPA_CHECK(supaHostUnregister(ptr));
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
        std::free(ptr);
      } else {
        C10_SUPA_CHECK(supaFreeHost(block->ptr_));
      }
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete block;
    }
  }

  bool isPinnedPtr(const void* ptr) {
    // First check if driver is broken/missing, in which case PyTorch CPU
    // functionalities should still work, we should report `false` here.
    if (!c10::supa::is_available()) {
      return false;
    }
    // supaPointerGetAttributes grabs context on the current device, so we set
    // device to one that already has context, if exists.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index = c10::supa::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
    }
    supaPointerAttributes attr{};
    // We do not believe that SUPA needs mutable access to the data
    // here.
    supaError_t err = supaPointerGetAttributes(&attr, ptr);
    // HIP throws hipErrorUnknown here
    if (err != supaSuccess) {
      (void)supaGetLastError(); // clear HIP error
      return false;
    }
    return attr.type == supaMemoryTypeHost;
  }

  void copy_data(void* dest, const void* src, std::size_t count) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for SUPAHostAllocator");
  }

 private:
  void process_events() {
    process_events_for_specific_size(-1);
  }

  void process_events_for_specific_size(int64_t size) {
    size_t event_count = 0;
    size_t max_events = 0;
    {
      std::lock_guard<std::mutex> g(supa_events_mutex_);
      max_events = supa_events_.size();
    }

    while (true) {
      // Avoid calling supaEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<EventPool::Event, Block*>> processed;

      {
        std::lock_guard<std::mutex> g(supa_events_mutex_);
        if (!supa_events_.empty()) {
          processed = std::move(supa_events_.back());
          supa_events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      if (size != -1) {
        if (event_count++ > max_events) {
          {
            std::lock_guard<std::mutex> g(supa_events_mutex_);
            supa_events_.push_front(std::move(*processed));
          }
          return;
        }
        if (size != static_cast<int64_t>(processed->second->size_)) {
          // if we are processing a specific size, and the size of the block
          // doesn't match, we can't use it.
          {
            std::lock_guard<std::mutex> g(supa_events_mutex_);
            supa_events_.push_front(std::move(*processed));
          }
          continue;
        }
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        if (!query_event(event)) {
          // push the event onto the back if it's not ready.
          {
            std::lock_guard<std::mutex> g(supa_events_mutex_);
            if (size == -1) {
              supa_events_.push_back(std::move(*processed));
              return;
            }
            supa_events_.push_front(std::move(*processed));
            continue;
          }
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        std::lock_guard<std::mutex> g(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

  void record_stream(std::optional<std::vector<EventPool::Event>>& events, c10::supa::SUPAStream stream) {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  static bool query_event(EventPool::Event& event) {
    supaError_t err = supaEventQuery(*event);
    if (err == supaErrorNotReady) {
      (void)supaGetLastError(); // clear supa error
      return false;
    }
    if (err != supaSuccess) {
      C10_SUPA_CHECK(err);
    }
    return true;
  }

  static EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  static TaskThreadPool* getThreadPool() {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::supa::SUPACachingAllocator::SUPAAllocatorConfig::pinned_max_register_threads()));
    return pool;
  }

  static void mapPagesForRegister(const void* ptr, size_t size, size_t i, size_t numThreads, size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart = (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      memset((void*)p, 0, 1);
    }
  }

  static void registerPages(const void* ptr, size_t size) {
    C10_SUPA_CHECK(supaHostRegister((void*)ptr, (size_t)size, supaHostRegisterDefault));

    // If host and device pointer don't match, give a warning and exit
    void* devptr = nullptr;
    C10_SUPA_CHECK(supaHostGetDevicePointer(&devptr, (void*)ptr, 0));
    TORCH_CHECK(
        (void*)devptr == (void*)ptr,
        "Host and device pointer dont match with supaHostRegister. "
        "Please dont use this feature by setting "
        "PYTORCH_SUPA_ALLOC_CONF=use_supa_host_register:False (default)",
        "");
  }

  inline void allocWithSupaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // supaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the supa global lock.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)
    *ptr = std::malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::supa::SUPACachingAllocator::SUPAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this, i, ptr, roundSize, numMapThreads, pageSize, &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using supaHostRegister
    registerPages(*ptr, roundSize);
  }

  EventPool event_pool_;

  alignas(64) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded
  // scenarios.
  alignas(64) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in
  // microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(64) std::mutex supa_events_mutex_;
  std::deque<std::pair<EventPool::Event, Block*>> supa_events_;
};

SUPAHostAllocator& getSUPAHostAllocator() {
  // leak and don't worry about shutdown
  static SUPAHostAllocator allocator;
  return allocator;
}

} // anonymous namespace

static void SUPAHostAllocatorDeleter(void* ctx) {
  getSUPAHostAllocator().free(ctx);
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, c10::supa::SUPAStream stream) {
  return getSUPAHostAllocator().record_event(ptr, ctx, stream);
}

bool CachingHostAllocator_isPinned(const void* ptr) {
  return getSUPAHostAllocator().isPinnedPtr(ptr);
}

// Releases cached pinned memory allocations via supaHostFree
void CachingHostAllocator_emptyCache() {
  getSUPAHostAllocator().empty_cache();
}

struct SUPAHostAllocatorWrapper final : public HostAllocator {
#if TORCH_VER >= TORCH_2_3_0
  at::DataPtr allocate(size_t size) override
#else
  at::DataPtr allocate(size_t size) const override
#endif
  {
    auto ptr_and_ctx = getSUPAHostAllocator().allocate(size);
    return {ptr_and_ctx.first, ptr_and_ctx.second, &SUPAHostAllocatorDeleter, at::DeviceType::CPU};
  }

  void free(void* ctx) {
    getSUPAHostAllocator().free(ctx);
  }

  bool record_event(void* ptr, void* ctx, c10::Stream s) override {
    return getSUPAHostAllocator().record_event(ptr, ctx, s);
  }

  void empty_cache() override {
    getSUPAHostAllocator().empty_cache();
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    getSUPAHostAllocator().copy_data(dest, src, count);
  }
};

static SUPAHostAllocatorWrapper supa_host_allocator;

HostAllocator* getCachingHostAllocator() {
  return &supa_host_allocator;
}

} // namespace at::supa
