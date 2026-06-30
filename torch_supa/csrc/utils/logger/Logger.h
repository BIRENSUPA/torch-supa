/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>

#include "fmt/format.h"
#include "torch_supa/csrc/utils/logger/LoggerInitializer.h"

#define TORCH_SUPA_S_EMERG torch_supa::utils::ILogger::Severity::Emerg
#define TORCH_SUPA_S_ALERT torch_supa::utils::ILogger::Severity::Alert
#define TORCH_SUPA_S_CRIT torch_supa::utils::ILogger::Severity::Critical
#define TORCH_SUPA_S_ERR torch_supa::utils::ILogger::Severity::Error
#define TORCH_SUPA_S_WARNING torch_supa::utils::ILogger::Severity::Warning
#define TORCH_SUPA_S_NOTICE torch_supa::utils::ILogger::Severity::Notice
#define TORCH_SUPA_S_INFO torch_supa::utils::ILogger::Severity::Info
#define TORCH_SUPA_S_DEBUG torch_supa::utils::ILogger::Severity::Debug
#define TORCH_SUPA_S_VERBOSE torch_supa::utils::ILogger::Severity::Verbose

#define TORCH_SUPA_ANY_SEVERITY_WITH_TAG(S, SS, T, format, ...)                                       \
  do {                                                                                                \
    auto logger = torch_supa::utils::LoggerInitializer::getLogger();                                  \
    if (logger && logger->isSeverityEnabled(S)) {                                                     \
      thread_local auto k_tid = syscall(SYS_gettid);                                                  \
      torch_supa::utils::InternalLog(                                                                 \
          logger, S, " [" T ":" __FILE__ ":{}] [" SS "] {} " format, __LINE__, k_tid, ##__VA_ARGS__); \
    }                                                                                                 \
    if (torch_supa::utils::EnvConfig::IsEnableRealtimeLog() && S <= TORCH_SUPA_S_DEBUG) {             \
      torch_supa::utils::RealtimeLog(format, ##__VA_ARGS__);                                          \
    }                                                                                                 \
  } while (0)

// dlog
#define TORCH_SUPA_ANY_SEVERITY_WITH_DTAG(S, SS, T, format, ...)                                      \
  do {                                                                                                \
    auto logger = torch_supa::utils::LoggerInitializer::getLogger();                                  \
    if (logger && logger->isSeverityEnabled(S)) {                                                     \
      thread_local auto k_tid = syscall(SYS_gettid);                                                  \
      torch_supa::utils::InternalDLog(                                                                \
          logger, S, " [" T ":" __FILE__ ":{}] [" SS "] {} " format, __LINE__, k_tid, ##__VA_ARGS__); \
    }                                                                                                 \
  } while (0)

#define TORCH_SUPA_ANY_SEVERITY_WITH_TAG_PURE(S, SS, T, format, ...)                                   \
  do {                                                                                                 \
    auto logger = torch_supa::utils::LoggerInitializer::getLogger();                                   \
    if (logger && logger->isSeverityEnabled(S)) {                                                      \
      thread_local auto k_tid = syscall(SYS_gettid);                                                   \
      torch_supa::utils::InternalLog(logger, S, " [" T "] [" SS "] {} " format, k_tid, ##__VA_ARGS__); \
    }                                                                                                  \
  } while (0)

// Use following macro to define TORCH_SUPA_TAG in source code
// #undef TORCH_SUPA_TAG
// #define TORCH_SUPA_TAG "default"
//
// If you need to log in header file (e.g. template implementation)
// Define TORCH_SUPA_TAG at the beginning and undef it at the end
// or use TORCH_SUPA_ANY_SEVERITY_WITH_TAG directly.

// For easy use with default tag
#define TORCH_SUPA_TAG "torch_supa"

#define TORCH_SUPA_ANY_SEVERITY(S, SS, format, ...) \
  TORCH_SUPA_ANY_SEVERITY_WITH_TAG(S, SS, TORCH_SUPA_TAG, format, ##__VA_ARGS__)

#define TORCH_SUPA_EMERG(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_EMERG, "EMERG", format, ##__VA_ARGS__)
#define TORCH_SUPA_ALERT(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_ALERT, "ALERT", format, ##__VA_ARGS__)
#define TORCH_SUPA_CRIT(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_CRIT, "CRITICAL", format, ##__VA_ARGS__)
#define TORCH_SUPA_ERROR(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_ERR, "ERROR", format, ##__VA_ARGS__)
#define TORCH_SUPA_WARN(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_WARNING, "WARN", format, ##__VA_ARGS__)
#define TORCH_SUPA_NOTICE(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_NOTICE, "NOTICE", format, ##__VA_ARGS__)
#define TORCH_SUPA_INFO(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_INFO, "INFO", format, ##__VA_ARGS__)
#define TORCH_SUPA_DEBUG(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_DEBUG, "DEBUG", format, ##__VA_ARGS__)
#define TORCH_SUPA_VERBOSE(format, ...) TORCH_SUPA_ANY_SEVERITY(TORCH_SUPA_S_VERBOSE, "VERBOSE", format, ##__VA_ARGS__)

#define TORCH_SUPA_EXCEPITON_ERROR(format, ...) \
  TORCH_SUPA_ANY_SEVERITY_WITH_TAG_PURE(TORCH_SUPA_S_ERR, "ERROR", TORCH_SUPA_TAG, format, ##__VA_ARGS__)

#define TORCH_SUPA_ANY_SEVERITY_DLOG(S, SS, format, ...) \
  TORCH_SUPA_ANY_SEVERITY_WITH_DTAG(S, SS, TORCH_SUPA_TAG, format, ##__VA_ARGS__)

#define TORCH_SUPA_DEMERG(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_EMERG, "EMERG", format, ##__VA_ARGS__)
#define TORCH_SUPA_DALERT(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_ALERT, "ALERT", format, ##__VA_ARGS__)
#define TORCH_SUPA_DCRIT(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_CRIT, "CRITICAL", format, ##__VA_ARGS__)
#define TORCH_SUPA_DERROR(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_ERR, "ERROR", format, ##__VA_ARGS__)
#define TORCH_SUPA_DWARN(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_WARNING, "WARN", format, ##__VA_ARGS__)
#define TORCH_SUPA_DNOTICE(format, ...) \
  TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_NOTICE, "NOTICE", format, ##__VA_ARGS__)
#define TORCH_SUPA_DINFO(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_INFO, "INFO", format, ##__VA_ARGS__)
#define TORCH_SUPA_DDEBUG(format, ...) TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_DEBUG, "DEBUG", format, ##__VA_ARGS__)
#define TORCH_SUPA_DVERBOSE(format, ...) \
  TORCH_SUPA_ANY_SEVERITY_DLOG(TORCH_SUPA_S_VERBOSE, "VERBOSE", format, ##__VA_ARGS__)

namespace torch_supa {
namespace utils {

class ILogger {
 public:
  enum class Severity : int32_t {
    Emerg = 0, /* system is unusable */
    Alert, /* action must be taken immediately */
    Critical, /* critical conditions */
    Error, /* error conditions */
    Warning, /* warning conditions */
    Notice, /* normal but significant conditions */
    Info, /* informational */
    Debug, /* debug-level messages */
    Verbose, /* very redundant messages */
  };

 protected:
  ILogger() = default;

 public:
  ILogger(const ILogger&) = delete;
  ILogger(ILogger&&) = delete;
  ILogger& operator=(const ILogger&) = delete;
  ILogger& operator=(ILogger&&) = delete;
  virtual ~ILogger() = default;
  virtual void log(Severity severity, const std::string& msg) noexcept = 0;
  virtual void dlog(Severity severity, const std::string& msg) noexcept = 0;
  virtual void setSeverity(Severity severity) noexcept = 0;
  virtual bool isSeverityEnabled(Severity severity) noexcept = 0;
};

template <typename... Args>
void InternalLog(ILogger* logger, ILogger::Severity severity, const char* format, Args... args) {
  logger->log(severity, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void InternalDLog(ILogger* logger, ILogger::Severity severity, const char* format, Args... args) {
  logger->dlog(severity, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void RealtimeLog(const char* format, Args... args) {
  auto* realtime_logger = torch_supa::utils::RealtimeLoggerInitializer::getLogger();
  realtime_logger->log(fmt::format(format, std::forward<Args>(args)...));
}

} // namespace utils
} // namespace torch_supa
