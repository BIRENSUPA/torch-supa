/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright © 2020~2024 Shanghai Biren Technology Co., Ltd.
 * All rights reserved.
 */

#ifdef USE_FLASH_ATTENTION

#include <torch/nn/functional.h>
#include <torch/python.h>
#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
// #include <ATen/cuda/detail/UnpackRaw.cuh>
#include "flash_api.h"
#include "flash_api_for_torch.h"
// #include <sute/numeric/numeric_types.hpp>
#include "torch_supa/csrc/utils/Utils.h"
// TODO: numeric_types.hpp/type_traits.hpp is not compatible with g++ in torch_supa
//       the declaration of bfloat16_t/half_t is a temperary bypass
#include <static_switch.h>
#include "flash.h"

namespace pytorch_flash {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {
  std::optional<at::Tensor> dummy_opt = std::nullopt;
  std::optional<const at::Tensor> dummy_opt2 = std::nullopt;
  auto results = ::mha_fwd(
      q,
      k,
      v,
      out_,
      std::nullopt,
      dummy_opt,
      dummy_opt2,
      p_dropout,
      softmax_scale,
      is_causal,
      window_size_left,
      window_size_right,
      false,
      std::nullopt);
  //   auto [output, q_padded, k_padded, v_padded, out_padded, logsumexp, p]
  // ( output, q_padded, k_padded, v_padded, logsumexp, philox_seed, philox_offset, debug_attn_mask)
  return std::make_tuple(
      std::move(results[0]),
      std::move(results[1]),
      std::move(results[2]),
      std::move(results[3]),
      std::move(results[5]),
      std::move(at::Tensor()),
      std::move(at::Tensor()),
      std::move(at::Tensor()));
}

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
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool deterministic) {
  const int64_t max_seqlen_batch_q = q.size(1);
  const int64_t max_seqlen_batch_k = k.size(1);

  std::optional<at::Tensor> cumulative_sequence_length_q{std::nullopt};
  std::optional<at::Tensor> cumulative_sequence_length_k{std::nullopt};

  std::optional<at::Generator> gen{std::nullopt};
  std::optional<at::Tensor> rng_state{std::nullopt};

  //   auto [dQuery, dKey, dValue, dSoftmax, dq_accum]
  auto results = ::mha_bwd(
      dout,
      q,
      k,
      v,
      out,
      softmax_lse,
      dq_,
      dk_,
      dv_,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      p_dropout,
      softmax_scale,
      is_causal,
      window_size_left,
      window_size_right,
      alibi_slopes_,
      deterministic,
      gen,
      rng_state);

  return std::make_tuple(std::move(results[0]), std::move(results[1]), std::move(results[2]), std::move(results[3]));
}
} // namespace pytorch_flash

#endif
