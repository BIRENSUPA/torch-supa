/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <algorithm>
#include <chrono>

#include "torch_supa/csrc/utils/logger/LoggerInitializer.h"
#include "torch_supa/csrc/utils/logger/RealtimeLogger.h"
#include "torch_supa/csrc/utils/logger/RingBuffer.h"

namespace torch_supa::utils {

#define SHM_NAME "/torch_supa_log_"
#define DUMP_PATH "./logs/realtime_log_"

time_t GetTimeStampSec() {
  timeval ts{};
  gettimeofday(&ts, nullptr);
  return ts.tv_sec;
}

RealtimeLogger::RealtimeLogger()
    : pid_(getpid()), log_size_(EnvConfig::GetRealtimeLogSize()), buffer_length_(log_size_ / k_msg_size) {
  // generate logs directory
  std::string log_dir = "./logs/";
  mkdir(log_dir.c_str(), 0755);

  // generate log files for different proc
  shm_name_ = SHM_NAME + std::to_string(pid_);
  shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR | O_CREAT, 0666);
  if (shm_fd_ == -1) {
    std::cerr << "Failed to open shared memory." << std::endl;
    return;
  }

  RegisterSignalHandlers();

  if (ftruncate(shm_fd_, static_cast<off_t>(log_size_)) == -1) {
    std::cerr << "Failed to truncate shared memory." << std::endl;
    return;
  }

  shm_addr_ = mmap(nullptr, log_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (shm_addr_ == MAP_FAILED) {
    std::cerr << "Failed to map shared memory." << std::endl;
    return;
  }

  ring_buffer_for_shm_ = std::make_unique<RingBufferForShm>(k_msg_size, buffer_length_, shm_addr_);

  realtime_log_thread_ = std::thread([&] {
    static constexpr std::size_t ts_size = 21;
    char timestamp[ts_size] = {0};
    while (!stop_) {
      SpinLockGuard guard(spinlock_);
      if (!log_msgs_.empty()) {
        auto& ts = log_msgs_[0].first;
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S ", std::localtime(&ts));
        _log_impl(std::string(timestamp) + log_msgs_[0].second);
        log_msgs_.erase(log_msgs_.begin());
      }
    }
  });
}

RealtimeLogger::~RealtimeLogger() {
  stop_ = true;
  realtime_log_thread_.join();

  // save logs to disk
  flushLogsToDisk();

  munmap(shm_addr_, log_size_);
  close(shm_fd_);
  shm_unlink(shm_name_.c_str());
}

void RealtimeLogger::log(const std::string& msg) {
  SpinLockGuard guard(spinlock_);
  log_msgs_.emplace_back(GetTimeStampSec(), msg);
}

void RealtimeLogger::_log_impl(const std::string& msg) {
  std::string truncated_msg = msg;

  // index
  char index[5];
  std::snprintf(index, sizeof(index), "%-4ld", buffer_index_++);
  if (buffer_index_ >= buffer_length_) {
    buffer_index_ = 0;
  }
  truncated_msg = std::string(index) + " " + truncated_msg;

  std::size_t remain_len = truncated_msg.size();
  while (remain_len > 0) {
    if (remain_len < k_msg_size) {
      truncated_msg.resize(k_msg_size, ' ');
      truncated_msg.at(k_msg_size - 1) = '\n';
      ring_buffer_for_shm_->push(const_cast<char*>(truncated_msg.c_str()));
      remain_len = 0;
    } else {
      std::string sub_msg = truncated_msg.substr(0, k_msg_size);
      truncated_msg = truncated_msg.substr(k_msg_size);
      ring_buffer_for_shm_->push(const_cast<char*>(sub_msg.c_str()));
      remain_len -= k_msg_size;
    }
  }
}

void RealtimeLogger::flushLogsToDisk() {
  // dump logs to ./logs/realtime_log_<pid>.log
  std::string log_file = DUMP_PATH + std::to_string(pid_) + ".log";
  std::ofstream file(log_file, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "[RealtimeLogger] Failed to open log file: " << log_file << std::endl;
    return;
  }

  std::string current_log;

  // read messages from the ring buffer
  auto callback = [&](const char* msg_data) {
    std::string line(msg_data, k_msg_size);

    line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return ch != ' '; }).base(), line.end());

    if (line.empty()) {
      return;
    }

    // check if it is new log (start with index)
    if (line.length() >= 5 && std::isdigit(line[0])) {
      if (!current_log.empty()) {
        file << current_log;
        if (current_log.back() != '\n') {
          file << '\n';
        }
        current_log.clear();
      }
      current_log = line.substr(5);
    } else {
      current_log += line;
    }
  };

  // dump messages data from ring buffer for shm
  ring_buffer_for_shm_->dump(callback);

  if (!current_log.empty()) {
    file << current_log;
    if (current_log.back() != '\n') {
      file << '\n';
    }
  }

  file.close();
}

void RealtimeLogger::RegisterSignalHandlers() {
  struct sigaction sa {};
  sa.sa_handler = [](int sig) {
    auto* logger = RealtimeLoggerInitializer::getLogger();
    if (logger) {
      logger->flushLogsToDisk();
      if (!logger->shm_name_.empty()) {
        shm_unlink(logger->shm_name_.c_str());
      }
    }
    signal(sig, SIG_DFL);
    raise(sig);
  };
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  // handle signals
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGQUIT, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
}

} // namespace torch_supa::utils
