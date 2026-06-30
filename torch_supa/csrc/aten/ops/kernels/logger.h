/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <array>

namespace at::supa {

constexpr const char *kREPO_COLOR = "\x1b[38;5;127m";  // magenta
constexpr const char *kTIMESTAMP_COLOR = "\x1B[0;38m"; // black
constexpr const char *kFILENAME_COLOR = "\x1B[1;35m";  // bold magenta
constexpr const char *kLINE_COLOR = "\x1B[38;5;40m";   // bold green
constexpr const char *kFUNCTION_COLOR = "\x1B[0;35m";  // magenta
constexpr const char *kDEBUG_COLOR = "\x1B[0;34m";     // blue
constexpr const char *kINFO_COLOR = "\x1B[0;32m";      // green
constexpr const char *kWARNING_COLOR = "\x1B[0;33m";   // yellow
constexpr const char *kERROR_COLOR = "\x1B[1;31m";     // red
constexpr const char *kRESET_COLOR = "\x1B[0m";        // reset

enum class SupaKernelLogLevel : uint8_t {
  LOG_DEBUG = 0,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_MAX,
};

class SupaKernelLog {
public:
  static void LogPrintf(const char *moduleColor, const char *moduleName, SupaKernelLogLevel level,
                        const char *file, int line, const char *func, const char *color,
                        const char *format, ...);
  static SupaKernelLogLevel GetLogLevel() noexcept { return log_level_; }
  static void SetLogLevel(SupaKernelLogLevel level) { log_level_ = level; }
  static void Reset(bool canLog) { can_log_ = canLog; }
  static bool CheckCanLog(SupaKernelLogLevel level);

private:
  static const char *RetrieveLogLevelName(SupaKernelLogLevel level) noexcept;
  static const char *GetTimeStamp();

  static inline SupaKernelLogLevel log_level_{SupaKernelLogLevel::LOG_DEBUG};
  static inline bool can_log_{false};
  static inline std::array<char, 1024> log_buffer_;
  static inline std::array<char, 100> timestamp_;
};
} // namespace at::supa

#define LOG_PRINT(level, color, format, ...)                                                       \
  do {                                                                                             \
    if (at::supa::SupaKernelLog::CheckCanLog(at::supa::SupaKernelLogLevel::level)) {               \
      at::supa::SupaKernelLog::LogPrintf(at::supa::kREPO_COLOR, "SUPA Kernel",                     \
                                         at::supa::SupaKernelLogLevel::level, __FILE__, __LINE__,  \
                                         __FUNCTION__, at::supa::color, format, ##__VA_ARGS__);    \
    }                                                                                              \
  } while (false)

#define SUPA_KERNEL_ERROR(format, ...) LOG_PRINT(LOG_ERROR, kERROR_COLOR, format, ##__VA_ARGS__)
#define SUPA_KERNEL_WARN(format, ...) LOG_PRINT(LOG_WARNING, kWARNING_COLOR, format, ##__VA_ARGS__)
#define SUPA_KERNEL_INFO(format, ...) LOG_PRINT(LOG_INFO, kINFO_COLOR, format, ##__VA_ARGS__)
#define SUPA_KERNEL_DEBUG(format, ...) LOG_PRINT(LOG_DEBUG, kDEBUG_COLOR, format, ##__VA_ARGS__)

#define SUPA_KERNEL_CHECK_TRUE(x, format, ...)                                                     \
  do {                                                                                             \
    if (!(x)) {                                                                                    \
      SUPA_KERNEL_ERROR(format, __VA_ARGS__);                                                      \
    }                                                                                              \
  } while (0)
