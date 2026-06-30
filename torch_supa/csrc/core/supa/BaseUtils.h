/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once
#include <algorithm>
#include <numeric>

namespace c10::supa {

inline uint64_t PowerOf2Ceil(uint64_t a) noexcept {
  if (!a) {
    return 0;
  }

  a |= (a >> 1);
  a |= (a >> 2);
  a |= (a >> 4);
  a |= (a >> 8);
  a |= (a >> 16);
  a |= (a >> 32);

  return a + 1;
}

/// Return true if the argument is a power of two > 0 (64 bit edition.)
constexpr inline bool isPowerOf2_64(uint64_t Value) {
  return Value && !(Value & (Value - 1));
}

} // namespace c10::supa
