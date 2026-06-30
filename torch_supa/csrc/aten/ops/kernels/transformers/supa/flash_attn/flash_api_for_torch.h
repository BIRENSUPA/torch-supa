/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace pytorch_flash {

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
    float p_dropout,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool return_softmax,
    std::optional<at::Generator> gen_);

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& out, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>& dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>& dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
    float p_dropout, // probability to drop
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool deterministic);

} // namespace pytorch_flash
