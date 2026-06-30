/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <supa_driver.h>
#include <cstddef>
#include <cstdint>

#include "torch_supa/csrc/core/supa/TorchVersion.h"

namespace c10d::supa::symmetric_memory {

constexpr int max_supa_p2p_domain_size = 72;
// Maximum number of channels
constexpr int symm_max_nblocks = 32;

#if TORCH_VER < TORCH_2_9_0
constexpr size_t signal_pad_size = 2048;
#endif
#if TORCH_VER >= TORCH_2_9_0 && TORCH_VER < TORCH_2_10_0
constexpr size_t signal_pad_size = symm_max_nblocks * max_supa_p2p_domain_size * sizeof(uint32_t);
#endif

// Maximally, a rank will need to sync with all other ranks, over all
// channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
// Default signal pad size, can be overridden via set_signal_pad_size().
constexpr size_t default_signal_pad_size = symm_max_nblocks * max_supa_p2p_domain_size * sizeof(uint32_t);

using HandleType = SUmemGenericAllocationHandle;

} // namespace c10d::supa::symmetric_memory
