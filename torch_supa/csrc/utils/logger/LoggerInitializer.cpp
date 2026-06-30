/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <iostream>
#include <mutex>
#include <thread>

#include "torch_supa/csrc/utils/EnvConfig.h"
#include "torch_supa/csrc/utils/logger/DefaultLogger.h"
#include "torch_supa/csrc/utils/logger/Logger.h"
#include "torch_supa/csrc/utils/logger/LoggerInitializer.h"

namespace torch_supa {
namespace utils {

// Used only for logger env var detection, all string is translated to lower case
// and compare
static inline bool EqualIgnoreCase(const std::string& lhs, const std::string& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return (strncasecmp(lhs.c_str(), rhs.c_str(), lhs.size()) == 0);
}

ILogger* LoggerInitializer::getLogger() {
  static std::unique_ptr<ILogger> k_logger;
  static std::once_flag k_once_flag;

  std::call_once(k_once_flag, [&logger = k_logger]() {
    // Set logger backend
    const auto* logger_backend = EnvConfig::GetLogBackend();
    if (!logger_backend) {
      // Default use Glog
      logger = std::make_unique<GlogLogger>();
    } else if (EqualIgnoreCase(logger_backend, "empty")) {
      // No log backend
      logger.reset();
    } else if (EqualIgnoreCase(logger_backend, "glog")) {
      logger = std::make_unique<GlogLogger>();
    } else if (EqualIgnoreCase(logger_backend, "stdout")) {
      logger = std::make_unique<StdLogger>();
    } else if (EqualIgnoreCase(logger_backend, "async")) {
      logger = std::make_unique<AsyncLogger>();
    } else {
      // Default use Glog if backend setting is invalid
      logger = std::make_unique<GlogLogger>();
      std::cout << fmt::format("Warning: Invalid logger backend: {}", logger_backend) << std::endl;
    }

    // Set log level by env var if exists. We do not allow omit log
    // level higher than warning. we can set backend to 'empty' if we
    // not want any log
    if (logger) {
      const auto* logger_level = EnvConfig::GetLogLevel();
      if (!logger_level) {
        // Default do nothing if no logger level specified or no log backend
      } else if (EqualIgnoreCase(logger_level, "Debug")) {
        logger->setSeverity(ILogger::Severity::Debug);
      } else if (EqualIgnoreCase(logger_level, "Verbose")) {
        logger->setSeverity(ILogger::Severity::Verbose);
      } else if (EqualIgnoreCase(logger_level, "Warning")) {
        logger->setSeverity(ILogger::Severity::Warning);
      } else if (EqualIgnoreCase(logger_level, "Notice")) {
        logger->setSeverity(ILogger::Severity::Notice);
      } else if (EqualIgnoreCase(logger_level, "Info")) {
        logger->setSeverity(ILogger::Severity::Info);
      } else {
        // Do nothing if logger level setting is invalid
        std::cout << fmt::format("Warning: Invalid logger level: {}", logger_level) << std::endl;
      }
    }
  });

  return k_logger.get();
}

RealtimeLogger* RealtimeLoggerInitializer::getLogger() {
  static std::unique_ptr<RealtimeLogger> k_logger;
  static std::once_flag k_once_flag;

  std::call_once(k_once_flag, [&logger = k_logger]() { logger = std::make_unique<RealtimeLogger>(); });

  return k_logger.get();
}

} // namespace utils
} // namespace torch_supa
