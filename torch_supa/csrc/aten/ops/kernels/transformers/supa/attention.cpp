/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/attention.h"
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "torch_supa/csrc/aten/ops/kernels/transformers/sdp_utils_cpp.h"
#ifdef USE_FLASH_ATTENTION
#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/flash_attn/flash_api_for_torch.h"
#endif
#include <torch/csrc/autograd/custom_function.h>
namespace at {

namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> __flash_attention_forward(
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
    const std::optional<Tensor>& _alibi_slopes) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  std::optional<Tensor> out = at::empty_like(query);
  std::optional<at::Tensor> block_table = std::nullopt; // we are not using the block table yet
  std::optional<Tensor> alibi_slopes = _alibi_slopes;

  const auto non_null_window_left = window_size_left.has_value() ? window_size_left.value() : -1;
  const auto non_null_window_right = window_size_right.has_value() ? window_size_right.value() : (is_causal ? 0 : -1);

  Tensor output;
  Tensor q_padded;
  Tensor k_padded;
  Tensor v_padded;
  Tensor logsumexp;
  Tensor philox_seed;
  Tensor philox_offset;
  Tensor debug_attn_mask;
  std::tie(output, q_padded, k_padded, v_padded, logsumexp, philox_seed, philox_offset, debug_attn_mask) =
      pytorch_flash::mha_fwd(
          query,
          key,
          value,
          out,
          alibi_slopes,
          static_cast<float>(dropout_p),
          static_cast<float>(softmax_scale),
          is_causal,
          static_cast<int>(non_null_window_left),
          static_cast<int>(non_null_window_right),
          return_debug_mask, /*return_softmax (this is used for testing)*/
          std::nullopt);

  debug_attn_mask = return_debug_mask ? debug_attn_mask : at::empty({0}, query.options());
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
}

} // namespace native
} // namespace at
