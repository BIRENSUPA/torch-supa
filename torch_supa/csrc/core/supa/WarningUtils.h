/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/util/Exception.h>

namespace torch_supa::warning {

// Suppresses dispatcher warnings while intentionally overriding registrations.
class IgnoreWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {}
};

inline c10::WarningHandler* getIgnoreHandler() {
  static IgnoreWarningHandler handler_ = IgnoreWarningHandler();
  return &handler_;
}

// Generates unique helper symbol names for each macro expansion site.
#define TORCH_SUPA_CONCAT_INNER(x, y) x##y
#define TORCH_SUPA_CONCAT(x, y) TORCH_SUPA_CONCAT_INNER(x, y)
// Temporarily installs the ignore handler around a registration block.
// The name prefix avoids duplicate symbols across translation units.
#define WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(name, enable, registration_body)                     \
  static int TORCH_SUPA_CONCAT(name, _enter_warning)() {                                             \
    if (enable) {                                                                                  \
      c10::WarningUtils::set_warning_handler(torch_supa::warning::getIgnoreHandler());               \
    }                                                                                              \
    return 1;                                                                                      \
  }                                                                                                \
  static int TORCH_SUPA_CONCAT(name, _temp_enter_warning) = TORCH_SUPA_CONCAT(name, _enter_warning)(); \
  registration_body                                                                                \
  static int TORCH_SUPA_CONCAT(name, _exit_warning)() {                                              \
    if (enable) {                                                                                  \
      c10::WarningUtils::set_warning_handler(nullptr);                                             \
    }                                                                                              \
    return 1;                                                                                      \
  }                                                                                                \
  static int TORCH_SUPA_CONCAT(name, _temp_exit_warning) = TORCH_SUPA_CONCAT(name, _exit_warning)();

} // namespace torch_supa::warning
