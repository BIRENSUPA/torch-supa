/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace at::native {
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
__flash_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& _seqused_k,
    const std::optional<Tensor>& _alibi_slopes);
} // namespace at::native
