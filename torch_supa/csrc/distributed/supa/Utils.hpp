/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

// This file contains utility functions common for SUPA, which can be used by
// ProcessGroupBCCL or SymmetricMemory.

namespace c10d::supa {

bool deviceSupportsMulticast(int device_idx);

} // namespace c10d::supa
