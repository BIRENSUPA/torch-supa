/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include "torch_supa/csrc/utils/logger/RealtimeLogger.h"
#pragma once

namespace torch_supa {
namespace utils {

class ILogger;
class RealtimeLogger;

// Eager Singleton initializer
class LoggerInitializer {
 public:
  static ILogger* getLogger();
};

class RealtimeLoggerInitializer {
  public:
   static RealtimeLogger* getLogger();
 };

} // namespace utils
} // namespace torch_supa
