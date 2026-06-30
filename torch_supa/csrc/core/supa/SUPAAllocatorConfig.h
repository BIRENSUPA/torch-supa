/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>

#include <c10/util/Exception.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>

namespace c10::supa::SUPACachingAllocator {

static constexpr size_t kMinBlockSize = 512; // all sizes are rounded to at least 512 bytes
static constexpr size_t kSmallSize = 1024 * 1024; // largest "small"cd .. allocation is 1 MiB
static constexpr size_t kSmallBuffer = 2 * 1024 * 1024UL; // "small" allocations are packed in 2 MiB blocks
static constexpr size_t kLargeBuffer = 20 * 1024 * 1024UL; // "large" allocations may be packed in 20 MiB blocks
static constexpr size_t kMinLargeAlloc = 10 * 1024 * 1024; // allocations between 1 and 10 MiB may use kLargeBuffer
static constexpr size_t kRoundLarge = 2 * 1024 * 1024UL; // round up large allocations to 2 MiB
static constexpr size_t kUmdPage = 32 * 1024 * 1024UL;
static constexpr size_t k4KAligned = 4 * 1024;
static constexpr size_t k512BAligned = 512;
static constexpr size_t k8KAligned = 8 * 1024UL;
static constexpr size_t k256KAligned = 256 * 1024;

enum class Expandable_Segments_Handle_Type : int {
  UNSPECIFIED = 0,
  POSIX_FD = 1,
  FABRIC_HANDLE = 2,
};

class C10_SUPA_API SUPAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
    return instance().m_expandable_segments;
  }

  static bool release_lock_on_supamalloc() {
    return instance().m_release_lock_on_supamalloc;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_supa_host_register() {
    return instance().m_pinned_use_supa_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static size_t pinned_reserve_segment_size_mb() {
    return instance().m_pinned_reserve_segment_size_mb;
  }

  static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

  static bool pinned_use_background_threads() {
    return instance().m_pinned_use_background_threads;
  }

  static SUPAAllocatorConfig& instance() {
    static SUPAAllocatorConfig* s_instance = ([]() {
      auto* inst = new SUPAAllocatorConfig(); // NOLINT(cppcoreguidelines-owning-memory)
      const char* env = getenv("PYTORCH_SUPA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  std::atomic<size_t> m_max_split_size;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<size_t> m_pinned_num_register_threads;
  std::atomic<size_t> m_pinned_reserve_segment_size_mb;
  std::atomic<bool> m_expandable_segments;
  std::atomic<bool> m_release_lock_on_supamalloc;
  std::atomic<bool> m_pinned_use_supa_host_register;
  std::atomic<bool> m_pinned_use_background_threads;
  bool set_expandable_segments_flag = false;

  SUPAAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_pinned_num_register_threads(1),
        m_pinned_reserve_segment_size_mb(0),
        m_expandable_segments(false),
        m_release_lock_on_supamalloc(false),
        m_pinned_use_supa_host_register(false),
        m_pinned_use_background_threads(false) {}

  static void lexArgs(const char* env, std::vector<std::string>& config);
  static void consumeToken(const std::vector<std::string>& config, size_t i, char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(const std::vector<std::string>& config, size_t i);
  size_t parsePinnedUseSupaHostRegister(const std::vector<std::string>& config, size_t i);
  size_t parsePinnedNumRegisterThreads(const std::vector<std::string>& config, size_t i);
};

// General caching allocator utilities
C10_SUPA_API void setAllocatorSettings(const std::string& env);

} // namespace c10::supa::SUPACachingAllocator
