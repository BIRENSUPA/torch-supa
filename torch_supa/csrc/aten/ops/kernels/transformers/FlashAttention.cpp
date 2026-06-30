/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/custom_function.h>
#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"

#include <limits>
#include <utility>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"

#include "torch_supa/csrc/aten/ops/kernels/transformers/sdp_utils_cpp.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/attention.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/attention_backward.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/sudnn_attn/sudnn_mha.h"

namespace at::supa {

using namespace at::native;

namespace {
template <int alignment_size, bool slice>
at::Tensor pad_last_dim(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  if (last_dim_size % alignment_size == 0) {
    return attn_bias;
  }
  auto pad_count = alignment_size - (last_dim_size % alignment_size);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  if (slice) {
    return padded_bias.slice_symint(-1, 0, last_dim_size);
  }
  return padded_bias;
}

inline c10::SymFloat calculate_scale(const at::Tensor& query, std::optional<double> scale) {
  const auto softmax_scale =
      scale.has_value() ? scale.value() : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

at::Tensor post_process_flash_output(at::Tensor out, c10::SymInt const& og_size) {
  if (!out.is_nested() && out.sym_size(-1) != og_size) {
    out = out.slice_symint(-1, 0, og_size);
  }
  return out;
}

} // namespace

// clang-format off
std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor>
SUPANativeFunctions::_scaled_dot_product_flash_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // clang-format on
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(max_seqlen_batch_k == max_seqlen_batch_v, "Key and Value must have the same sequence length");

  auto [output, logsumexp, philox_seed, philox_offset, debug_attn_mask] = at::native::__flash_attention_forward(
      q_t,
      k_t,
      v_t,
      std::nullopt,
      std::nullopt,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      is_causal,
      return_debug_mask,
      softmax_scale,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt);

  // Reshape output to convert nnz to batch_size and seq_len
  Tensor attention = output.transpose(1, 2);
  return std::make_tuple(
      attention,
      logsumexp,
      Tensor(),
      Tensor(),
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> SUPANativeFunctions::_scaled_dot_product_flash_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  Tensor dout = grad_out_.transpose(1, 2);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  Tensor out_t = out.transpose(1, 2);

  auto [grad_q, grad_k, grad_v] =
      at::native::__flash_attention_backward(dout, q_t, k_t, v_t, out_t, logsumexp, dropout_p, is_causal, scale);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> SUPANativeFunctions::_scaled_dot_product_efficient_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(max_seqlen_batch_k == max_seqlen_batch_v, "Key and Value must have the same sequence length");

  auto [output, logsumexp, philox_seed, philox_offset, debug_attn_mask] = at::native::__flash_attention_forward(
      q_t,
      k_t,
      v_t,
      std::nullopt,
      std::nullopt,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      is_causal,
      false,
      softmax_scale,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt);
  Tensor attention = output.transpose(1, 2);
  return std::make_tuple(std::move(attention), std::move(logsumexp), std::move(philox_seed), std::move(philox_offset));
}

// clang-format off
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SUPANativeFunctions::_scaled_dot_product_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {
  // clang-format on
  Tensor dout = grad_out_.transpose(1, 2);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  Tensor out_t = out.transpose(1, 2);

  auto [grad_q, grad_k, grad_v] =
      at::native::__flash_attention_backward(dout, q_t, k_t, v_t, out_t, logsumexp, dropout_p, is_causal, scale);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), at::Tensor());
}

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor> _cudnn_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_bias,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_kv,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_kv,
    bool compute_logsumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const bool is_nested = cumulative_sequence_length_q.has_value();
  if (!is_nested) {
    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim_qk = query.size(3);
    const int64_t head_dim_v = value.size(3);
    auto attn_bias_ = attn_bias;
    if (attn_bias_.has_value()) {
      const auto bias_dim = attn_bias_.value().dim();
      if (bias_dim == 2) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else if (bias_dim == 3) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else {
        TORCH_CHECK(
            bias_dim == 4,
            "SuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ",
            attn_bias_.value().dim(),
            "D");
        attn_bias_ = attn_bias_.value().expand(
            {batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_kv});
      }
    }

    at::Tensor attention;
    at::Tensor log_sumexp;
    at::Tensor sudnn_seed;
    at::Tensor sudnn_offset;
    sudnn_seed = at::empty({}, at::dtype(at::kLong).device(query.device()));
    sudnn_offset = at::empty({}, at::dtype(at::kLong).device(query.device()));

    const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

    PhiloxSupaState philox_state;
    if (use_dropout) {
      auto* gen =
          at::get_generator_or_default<SUPAGeneratorImpl>(std::nullopt, at::supa::detail::getDefaultSUPAGenerator());
      std::lock_guard<std::mutex> lock(gen->mutex_);
      philox_state = gen->philox_supa_state(batch_size * num_heads * max_seqlen_batch_q * max_seqlen_batch_kv);
      int64_t* seed_ptr = static_cast<int64_t*>(sudnn_seed.data_ptr());
      int64_t* offset_ptr = static_cast<int64_t*>(sudnn_offset.data_ptr());
      if (philox_state.captured_) {
        *seed_ptr = static_cast<int64_t>(*philox_state.seed_.ptr);
        *offset_ptr =
            static_cast<int64_t>(*(philox_state.offset_.ptr) + static_cast<int64_t>(philox_state.offset_intragraph_));
      } else {
        *seed_ptr = static_cast<int64_t>(philox_state.seed_.val);
        *offset_ptr = static_cast<int64_t>(philox_state.offset_.val);
      }
    }

    const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();

    run_sudnn_SDP_fprop(
        batch_size,
        num_heads,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        head_dim_qk,
        head_dim_v,
        static_cast<float>(softmax_scale),
        compute_logsumexp,
        is_causal,
        dropout_p,
        query,
        key,
        value,
        attn_bias_,
        log_sumexp,
        attention,
        sudnn_seed,
        sudnn_offset);

    return std::make_tuple(
        std::move(attention),
        std::move(log_sumexp),
        Tensor(),
        Tensor(),
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        std::move(sudnn_seed),
        std::move(sudnn_offset),
        Tensor());
  }
  const int64_t batch_size = cumulative_sequence_length_q.value().size(0) - 1;
  const int64_t num_heads_q = query.size(-2);
  const int64_t num_heads_k = key.size(-2);
  const int64_t num_heads_v = value.size(-2);
  const int64_t head_dim_qk = query.size(-1);
  const int64_t head_dim_v = value.size(-1);
  auto attn_bias_ = attn_bias;
  if (attn_bias_.has_value()) {
    const auto bias_dim = attn_bias_.value().dim();
    if (bias_dim == 2) {
      attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
    } else if (bias_dim == 3) {
      attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
    } else {
      attn_bias_ =
          attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_kv});
      TORCH_CHECK(
          bias_dim == 4, "SuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
    }
  }

  at::Tensor attention;
  at::Tensor log_sumexp;
  at::Tensor sudnn_seed;
  at::Tensor sudnn_offset;
  sudnn_seed = at::empty({}, at::dtype(at::kLong).device(query.device()));
  sudnn_offset = at::empty({}, at::dtype(at::kLong).device(query.device()));

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  PhiloxSupaState philox_state;
  if (use_dropout) {
    auto* gen =
        at::get_generator_or_default<SUPAGeneratorImpl>(std::nullopt, at::supa::detail::getDefaultSUPAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox_state = gen->philox_supa_state(batch_size * num_heads_q * max_seqlen_batch_q * max_seqlen_batch_kv);
    int64_t* seed_ptr = static_cast<int64_t*>(sudnn_seed.data_ptr());
    int64_t* offset_ptr = static_cast<int64_t*>(sudnn_offset.data_ptr());
    if (philox_state.captured_) {
      *seed_ptr = static_cast<int64_t>(*philox_state.seed_.ptr);
      *offset_ptr =
          static_cast<int64_t>(*(philox_state.offset_.ptr) + static_cast<int64_t>(philox_state.offset_intragraph_));
    } else {
      *seed_ptr = static_cast<int64_t>(philox_state.seed_.val);
      *offset_ptr = static_cast<int64_t>(philox_state.offset_.val);
    }
  }

  const auto softmax_scale = sdp::calculate_scale(query, scale).as_float_unchecked();

  run_sudnn_SDP_fprop_nestedtensor(
      batch_size,
      num_heads_q,
      num_heads_k,
      num_heads_v,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      head_dim_qk,
      head_dim_v,
      static_cast<float>(softmax_scale),
      compute_logsumexp,
      is_causal,
      dropout_p,
      cumulative_sequence_length_q.value(),
      cumulative_sequence_length_kv.value(),
      query,
      key,
      value,
      attn_bias_,
      log_sumexp,
      attention,
      sudnn_seed,
      sudnn_offset);
  return std::make_tuple(
      std::move(attention),
      std::move(log_sumexp),
      cumulative_sequence_length_q.value(),
      cumulative_sequence_length_kv.value(),
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      std::move(sudnn_seed),
      std::move(sudnn_offset),
      Tensor());
}

// clang-format off
std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor> 
SUPANativeFunctions::_scaled_dot_product_cudnn_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_bias,
    bool compute_logsumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // clang-format on
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);

  return at::supa::_cudnn_attention_forward(
      query,
      key,
      value,
      attn_bias,
      std::nullopt,
      std::nullopt,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      compute_logsumexp,
      dropout_p,
      is_causal,
      return_debug_mask,
      scale);
}

std::tuple<Tensor, Tensor, Tensor> SUPANativeFunctions::_scaled_dot_product_cudnn_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& attn_bias,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "suDNN Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    }
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);

  std::optional<Tensor> attn_bias_;
  if (attn_bias.defined()) {
    attn_bias_ = attn_bias;
  }
  if (attn_bias_.has_value()) {
    const auto bias_dim = attn_bias_.value().dim();
    if (bias_dim == 2) {
      attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
    } else if (bias_dim == 3) {
      attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
    } else {
      TORCH_CHECK(
          bias_dim == 4, "SuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
      attn_bias_ =
          attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_k});
    }
  }

  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  auto dq = at::empty_like(query);
  auto dk = at::empty_like(key);
  auto dv = at::empty_like(value);
  run_sudnn_SDP_bprop(
      batch_size,
      num_heads,
      max_q,
      max_k,
      head_dim_qk,
      head_dim_v,
      static_cast<float>(softmax_scale),
      is_causal,
      static_cast<float>(dropout_p),
      query,
      key,
      value,
      attn_bias_,
      out,
      grad_out,
      logsumexp.unsqueeze(-1),
      dq,
      dk,
      dv,
      philox_seed,
      philox_offset);
  return std::make_tuple(std::move(dq), std::move(dk), std::move(dv));
}

} // namespace at::supa
