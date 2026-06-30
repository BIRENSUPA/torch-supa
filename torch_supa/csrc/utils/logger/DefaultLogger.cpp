/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <sys/stat.h>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <mutex>

#include "fmt/chrono.h"
#include "glog/logging.h"
#include "torch_supa/csrc/utils/EnvConfig.h"
#include "torch_supa/csrc/utils/logger/DefaultLogger.h"
#include "torch_supa/csrc/utils/logger/RingBuffer.h"

using namespace std::chrono_literals;
using namespace std::string_literals;

namespace torch_supa {
namespace utils {

void BaseLogger::setSeverity(Severity severity) noexcept {
  severity_ = severity;
}

bool BaseLogger::isSeverityEnabled(Severity severity) noexcept {
  return (severity <= severity_);
}

static inline std::string LoggerPrefix() {
  const std::chrono::time_point<std::chrono::high_resolution_clock> time = std::chrono::high_resolution_clock::now();
  std::time_t t = std::chrono::high_resolution_clock::to_time_t(time);
  auto fraction = time - std::chrono::time_point_cast<std::chrono::seconds>(time);

  return fmt::format("{:%Y-%m-%dT%H:%M:%S}.{:09}Z FRAMEWORK", fmt::gmtime(t), static_cast<int32_t>(fraction / 1ns));
}

void StdLogger::log(Severity severity, const std::string& msg) noexcept {
  std::cout << LoggerPrefix() << msg << std::endl;
}

GlogLogger::GlogLogger() {
  auto log_dir = "./logs/"s;
  if (const auto* env_log_dir = EnvConfig::GetLogDir(); (env_log_dir != nullptr) && (strlen(env_log_dir) > 0)) {
    log_dir = env_log_dir;
  }

  (void)mkdir(log_dir.c_str(), 0755);
  google::InitGoogleLogging("torch_supa");
  std::string log_destination = log_dir + "/torch_supa-"s;
  for (int severity = 0; severity < google::NUM_SEVERITIES; ++severity) {
    google::SetLogDestination(severity, log_destination.c_str());
    google::SetLogFilenameExtension(".log");
  }

  FLAGS_timestamp_in_logfile_name = true;
  FLAGS_logfile_mode = 0664;
  FLAGS_log_prefix = false;
  FLAGS_max_log_size = 1024; // Set max log file size
  FLAGS_stop_logging_if_full_disk = true; // If disk is full
}

GlogLogger::~GlogLogger() {
  google::ShutdownGoogleLogging();
}

void GlogLogger::log(Severity severity, const std::string& msg) noexcept {
  static auto flush_log_instantly = EnvConfig::IsEnableFlushLogInstantly();
  static std::once_flag once_flag_instant_flushing;
  std::call_once(once_flag_instant_flushing, []() {
    LOG(INFO) << "Log instant flushing is " << (flush_log_instantly ? "" : "not ") << "on for glog";
  });

  auto prefix = LoggerPrefix();

  switch (severity) {
    case Severity::Emerg:
    case Severity::Alert:
    case Severity::Critical:
      LOG(FATAL) << prefix << msg;
      break;
    case Severity::Error:
      LOG(ERROR) << prefix << msg;
      break;
    case Severity::Warning:
      LOG(WARNING) << prefix << msg;
      break;
    default:
      LOG(INFO) << prefix << msg;
      break;
  }

  if (flush_log_instantly) {
    // flush log instantly in debug mode to avoid log missing if the process exits unexpectly
    flush();
  }
}

// Special debug mode logging macros only have an effect in debug mode
// and are compiled away to nothing for non-debug mode compiles.
void GlogLogger::dlog(Severity severity, const std::string& msg) noexcept {
  auto prefix = LoggerPrefix();
  switch (severity) {
    case Severity::Emerg:
    case Severity::Alert:
    case Severity::Critical:
      DLOG(FATAL) << prefix << msg;
      break;
    case Severity::Error:
      DLOG(ERROR) << prefix << msg;
      break;
    case Severity::Warning:
      DLOG(WARNING) << prefix << msg;
      break;
    default:
      DLOG(INFO) << prefix << msg;
      break;
  }
  flush();
}

void GlogLogger::flush() {
  google::FlushLogFiles(google::FATAL);
  google::FlushLogFiles(google::ERROR);
  google::FlushLogFiles(google::WARNING);
  google::FlushLogFiles(google::INFO);
}

AsyncLogger::AsyncLogger() {
  std::size_t queue_size = EnvConfig::GetAsyncLoggerQueueSize();
  LOG(INFO) << "Logger is running asynchronously with queue size: " << queue_size;
  ring_buffer_ = std::make_unique<RingBuffer<std::pair<Severity, std::string>>>(queue_size);

  log_thread_ = std::thread([&] {
    std::pair<Severity, std::string> item;
    while (!stop_) {
      if (ring_buffer_->pop(item)) {
        real_logger_.log(item.first, item.second);
      } else {
        std::this_thread::yield();
      }
    }

    real_logger_.flush();
  });
}

AsyncLogger::~AsyncLogger() {
  stop_ = true;
  log_thread_.join();
}

void AsyncLogger::log(Severity severity, const std::string& msg) noexcept {
  ring_buffer_->push(std::make_pair(severity, msg));
}

} // namespace utils
} // namespace torch_supa
