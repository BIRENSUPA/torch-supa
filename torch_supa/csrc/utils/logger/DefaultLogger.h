/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <string>
#include <thread>
#include <utility>

#include "torch_supa/csrc/utils/logger/Logger.h"

namespace torch_supa {
namespace utils {

template <typename T>
class RingBuffer;

class BaseLogger : public ILogger {
 public:
  BaseLogger() = default;
  BaseLogger(const BaseLogger&) = delete;
  BaseLogger& operator=(const BaseLogger&) = delete;
  BaseLogger(BaseLogger&&) = delete;
  BaseLogger& operator=(BaseLogger&&) = delete;
  ~BaseLogger() override = default;

  void setSeverity(Severity severity) noexcept override;

  bool isSeverityEnabled(Severity severity) noexcept override;

 protected:
#ifdef NDEBUG
  Severity severity_ = Severity::Notice;
#else
  Severity severity_ = Severity::Debug;
#endif
};

class StdLogger : public BaseLogger {
 public:
  void log(Severity severity, const std::string& msg) noexcept override;
  void dlog(Severity severity, const std::string& msg) noexcept override {
#ifndef NDEBUG
    log(severity, msg);
#endif
  };
};

class GlogLogger : public BaseLogger {
 public:
  GlogLogger();
  GlogLogger(const GlogLogger&) = delete;
  GlogLogger& operator=(const GlogLogger&) = delete;
  GlogLogger(GlogLogger&&) = delete;
  GlogLogger& operator=(GlogLogger&&) = delete;

  ~GlogLogger() override;

  void log(Severity severity, const std::string& msg) noexcept override;
  void dlog(Severity severity, const std::string& msg) noexcept override;

  // glog only
  static void flush();
};

class AsyncLogger : public BaseLogger {
 public:
  AsyncLogger();
  AsyncLogger(const AsyncLogger&) = delete;
  AsyncLogger& operator=(const AsyncLogger&) = delete;
  AsyncLogger(AsyncLogger&&) = delete;
  AsyncLogger& operator=(AsyncLogger&&) = delete;
  ~AsyncLogger() override;

  void log(Severity severity, const std::string& msg) noexcept override;
  void dlog(Severity severity, const std::string& msg) noexcept override {
#ifndef NDEBUG
    log(severity, msg);
#endif
  };

 private:
  GlogLogger real_logger_;
  std::thread log_thread_;
  std::unique_ptr<RingBuffer<std::pair<Severity, std::string>>> ring_buffer_;
  bool stop_ = false;
};

} // namespace utils
} // namespace torch_supa
