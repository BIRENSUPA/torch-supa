/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <time.h>
#include <cstdint>

namespace at {
namespace supa {
// enqueues a kernel that spins for the specified number of cycles
void sleep(int64_t cycles);

} // namespace supa
} // namespace at