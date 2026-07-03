/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <stdlib.h>
#include <cstring>
#include <mutex>

#include "torch_supa/csrc/utils/EnvConfig.h"

namespace torch_supa {
namespace utils {
namespace EnvConfig {

static inline bool GetEnvBool(const char* env_name, bool default_value = false) {
  // Treat common textual false values as false. Perf/native-op switches are
  // often exported as "true"/"false" by test harnesses, not only "1"/"0".
  auto* env = std::getenv(env_name);
  if (!env || env[0] == '\0') {
    return default_value;
  }
  if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "False") == 0 ||
      std::strcmp(env, "FALSE") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "Off") == 0 ||
      std::strcmp(env, "OFF") == 0) {
    return false;
  }
  return true;
}

static inline int GetEnvInt(const char* env_name, int default_value = 0) {
  // return false when empty or is '0'
  auto* env = std::getenv(env_name);
  if (!env || env[0] == '\0') {
    return default_value;
  }

  return std::atoi(env);
}

static inline double GetEnvDouble(const char* env_name, double default_value = 0.0) {
  // return 0.0 when empty or is '0.0'
  auto* env = std::getenv(env_name);
  if (!env || env[0] == '\0') {
    return default_value;
  }

  return std::atof(env);
}

static inline const char* GetEnvStr(const char* env_name) {
  return std::getenv(env_name);
}

bool getEnv(const char* env_name, bool default_value) {
  return GetEnvBool(env_name, default_value);
}

int getEnv(const char* env_name, int default_value) {
  return GetEnvInt(env_name, default_value);
}

double getEnv(const char* env_name, double default_value) {
  return GetEnvDouble(env_name, default_value);
}

const char* getEnv(const char* env_name, const char* default_value) {
  auto* env = std::getenv(env_name);
  if (!env) {
    return default_value;
  }

  return env;
}

ConfigState::ConfigState() : debug_(GetEnvBool("DEBUG")) {}

ConfigState& Instance() {
  static ConfigState k_instance;
  return k_instance;
}

} // namespace EnvConfig
} // namespace utils
} // namespace torch_supa
