/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "torch_supa/csrc/utils/EnvConfig.h"

#pragma once

namespace torch_supa::utils {

constexpr std::size_t k_msg_size = 128;

class RingBufferForShm;

class SpinLock {
 public:
  SpinLock() = default;
  SpinLock(const SpinLock&) = delete;
  SpinLock(SpinLock&&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;
  SpinLock& operator=(SpinLock&&) = delete;
  ~SpinLock() = default;

  void lock() {
    while (flag.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() {
    flag.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
 public:
  explicit SpinLockGuard(SpinLock& spinLock) : lock(spinLock) {
    lock.lock();
  }

  ~SpinLockGuard() {
    lock.unlock();
  }

  SpinLockGuard(const SpinLockGuard&) = delete;
  SpinLockGuard(SpinLockGuard&&) = delete;
  SpinLockGuard& operator=(const SpinLockGuard&) = delete;
  SpinLockGuard& operator=(SpinLockGuard&&) = delete;

 private:
  SpinLock& lock;
};

class RealtimeLogger {
 public:
  RealtimeLogger();
  RealtimeLogger(const RealtimeLogger&) = delete;
  RealtimeLogger(RealtimeLogger&&) = delete;
  RealtimeLogger& operator=(const RealtimeLogger&) = delete;
  RealtimeLogger& operator=(RealtimeLogger&&) = delete;
  ~RealtimeLogger();

  void log(const std::string& msg);

 private:
  void _log_impl(const std::string& msg);
  void flushLogsToDisk(); // write the log to the disk
  static void RegisterSignalHandlers();

  int shm_fd_ = -1;
  void* shm_addr_ = nullptr;
  std::unique_ptr<RingBufferForShm> ring_buffer_for_shm_;
  size_t buffer_index_ = 0;

  std::thread realtime_log_thread_;
  std::vector<std::pair<time_t, std::string>> log_msgs_;
  bool stop_ = false;

  std::size_t log_size_ = 0;
  std::size_t buffer_length_ = 0;
  SpinLock spinlock_;

  std::string shm_name_;
  pid_t pid_ = 0;
};
} // namespace torch_supa::utils
