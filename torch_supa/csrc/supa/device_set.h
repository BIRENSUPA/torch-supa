/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <bitset>
#include <cstddef>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace torch_supa {

using device_set = std::bitset<C10_COMPILE_TIME_MAX_SUPA_GPUS>;

} // namespace torch_supa
