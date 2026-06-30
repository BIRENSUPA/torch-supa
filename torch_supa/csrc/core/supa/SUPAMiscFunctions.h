/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
// this file is to avoid circular dependency between SUPAFunctions.h and
// SUPAExceptions.h

#include "torch_supa/csrc/core/supa/SUPAMacros.h"

#include <mutex>

namespace c10::supa {
C10_SUPA_API const char* get_supa_check_suffix() noexcept;
C10_SUPA_API std::mutex* getFreeMutex();
} // namespace c10::supa
