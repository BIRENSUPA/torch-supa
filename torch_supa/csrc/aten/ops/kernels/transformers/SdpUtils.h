/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/macros/Macros.h>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace sdp {

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug);
SDPBackend select_sdp_backend(sdp_params const& kernel_params);
C10_SUPA_EXPORT bool is_flash_attention_available();
C10_SUPA_EXPORT bool can_use_flash_attention(sdp_params const& params, bool debug);
C10_SUPA_EXPORT bool can_use_mem_efficient_attention(sdp_params const& params, bool debug);
C10_SUPA_EXPORT bool can_use_cudnn_attention(sdp_params const& params, bool debug);

} // namespace sdp
