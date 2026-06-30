/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright (c) 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <cstdlib>

namespace torch_supa::utils {

// For variant diagnostic features:
// Global static/dynamic configuration from environment var
// Global volatile configuration accessed runtime by get/set interfaces
namespace EnvConfig {

class ConfigState {
public:
  ConfigState();
  // Global static configuration from environment var
  bool debug_ = false;
#ifdef _DEBUG
  bool should_flush_log_instantly = true;
  bool should_enable_signal_handling = true;
#else
  bool should_flush_log_instantly = false;
  bool should_enable_signal_handling = false;
#endif
};

ConfigState &Instance();

bool getEnv(const char *env_name, bool default_value);
int getEnv(const char *env_name, int default_value);
double getEnv(const char *env_name, double default_value);
const char *getEnv(const char *env_name, const char *default_value);

// static:    value got from environment once and never changed.
#define DEFINE_STATIC_ENV_VAR(_ENV, _VAR, _T, _DFT)                            \
  class Env_##_VAR {                                                           \
  public:                                                                      \
    static inline _T value =                                                   \
        (#_ENV[0] == '\0') ? static_cast<_T>(_DFT)                             \
                           : getEnv("BRTB_" #_ENV, static_cast<_T>(_DFT));     \
  };                                                                           \
                                                                               \
  static inline _T Get##_VAR() { return Env_##_VAR::value; }                   \
                                                                               \
  static inline bool Is##_VAR() { return Get##_VAR(); }

// clang-format off

// Global static configuration from environment var
DEFINE_STATIC_ENV_VAR(LOG_LEVEL, LogLevel, const char*, nullptr)
DEFINE_STATIC_ENV_VAR(LOG_BACKEND, LogBackend, const char*, nullptr)
DEFINE_STATIC_ENV_VAR(LOG_DIR, LogDir, const char*, nullptr)
DEFINE_STATIC_ENV_VAR(ENABLE_REALTIME_LOG, EnableRealtimeLog, bool, false)
DEFINE_STATIC_ENV_VAR(REALTIME_LOG_SIZE, RealtimeLogSize, int, 12800)
DEFINE_STATIC_ENV_VAR(ENABLE_SIGNAL_HANDLING, EnableSignalHandling, bool, Instance().should_enable_signal_handling)
// instantly flush log switch, debug mode is on; otherwise off
DEFINE_STATIC_ENV_VAR(ENABLE_FLUSH_LOG_INSTANTLY, EnableFlushLogInstantly, bool, Instance().should_flush_log_instantly)
DEFINE_STATIC_ENV_VAR(ASYNC_LOGGER_QUEUE_SIZE, AsyncLoggerQueueSize, int, 1048576)
DEFINE_STATIC_ENV_VAR(SUBLAS_PREFERRED_BACKEND, SublasPreferredBackend, const char*, "Sublas")
DEFINE_STATIC_ENV_VAR(ENABLE_NATIVE_OP, EnableNativeOP, bool, false)
DEFINE_STATIC_ENV_VAR(ENABLE_DTYPE_DEMOTION, EnableDtypeDemotion, bool, false)
// clang-format on

#undef DEFINE_STATIC_ENV_VAR
} // namespace EnvConfig

} // namespace torch_supa::utils
