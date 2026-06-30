/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <cstdint>

#include <ATen/SDPBackend.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/transformers/attention.h>
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/SdpUtils.h"

namespace at::supa {

int64_t SUPANativeFunctions::_fused_sdp_choice(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = sdp::select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

namespace {
// copy from pytorch/aten/src/ATen/native/transformers/attention.cpp
inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK(
      query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
      "Expected query, key, and value to have the same dtype, but got query.dtype: ",
      query_.dtype(),
      " key.dtype: ",
      key.dtype(),
      " and value.dtype: ",
      value.dtype(),
      " instead.");
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got query.device: ",
      query_.device(),
      " key.device: ",
      key.device(),
      " and value.device: ",
      value.device(),
      " instead.");
  TORCH_CHECK(
      query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
      "Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: ",
      query_.dim(),
      " key.dim: ",
      key.dim(),
      " and value.dim: ",
      value.dim(),
      " instead.");
  if (attn_mask_.has_value()) {
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(
        mask_dtype == at::kBool || mask_dtype == at::kFloat || mask_dtype == query_.dtype(),
        "Expected attn_mask dtype to be bool or float or to match query dtype, but got attn_mask.dtype: ",
        mask_dtype,
        " and  query.dtype: ",
        query_.dtype(),
        " instead.");
    TORCH_CHECK(
        !query_.is_nested() && !key.is_nested(),
        "Scaled_dot_product_attention: Nested tensors for query / key are not supported "
        "when an explicit attn_mask is set");
  }
}

bool should_compute_logsumexp(const Tensor& query, const Tensor& key, const Tensor& value) {
  const bool any_inputs_require_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

template <int alignment>
bool aligned_tensor(const at::Tensor& tensor) {
  for (const auto i : c10::irange(tensor.dim() - 1)) {
    auto stride = tensor.sym_stride(i).maybe_as_int();
    // If the stride is unknown at compilation time, assume it is unaligned
    // and always pad it. This is helpful to avoid unnecessary guards.
    if (!stride) {
      return false;
    }

    if ((*stride) % alignment != 0) {
      return false;
    }
  }
  return tensor.sym_stride(-1) == 1;
}

template <int alignment>
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

at::Tensor preprocess_mask(
    const at::Tensor& mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  constexpr int mem_eff_alignment = 8;
  at::Tensor result_mask = mask;
  if (!aligned_tensor<mem_eff_alignment>(mask)) {
    result_mask = pad_bias<mem_eff_alignment>(mask);
  }
  return result_mask.expand_symint({query.sym_size(0), query.sym_size(1), query.sym_size(2), key.sym_size(2)});
}

std::optional<Tensor> convert_boolean_attn_mask_(
    const std::optional<Tensor>& attn_mask,
    caffe2::TypeMeta dtype,
    double neg_inf) {
  // Pass through
  if (!attn_mask.has_value()) {
    return std::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    return at::where(
        *attn_mask, 0.0, at::scalar_tensor(neg_inf, at::TensorOptions().dtype(dtype).device(attn_mask->device())));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

std::optional<Tensor> convert_boolean_attn_mask(const std::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  return convert_boolean_attn_mask_(attn_mask, dtype, -std::numeric_limits<double>::infinity());
}

} // namespace

// is_flash_attention_available
Tensor SUPANativeFunctions::scaled_dot_product_attention(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  using sdp::SDPBackend;
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  choice_int = _fused_sdp_choice(query_, key, value, attn_mask_, dropout_p, is_causal, scale, enable_gqa);
  const auto backend = static_cast<SDPBackend>(choice_int);
  auto attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
  switch (backend) {
    case SDPBackend::cudnn_attention: {
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      auto out_lse_softmax = at::_scaled_dot_product_cudnn_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case SDPBackend::flash_attention: {
      // remove useless preprocessing code
      auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
          query_, key, value, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case SDPBackend::efficient_attention: {
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      if (attn_mask.has_value()) {
        attn_mask.value() = preprocess_mask(attn_mask.value(), query_, key, value);
      }
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, scale);
      return std::get<0>(out_and_lse);
    }
    case SDPBackend::overrideable: {
      auto out_lse_softmax = at::_scaled_dot_product_fused_attention_overrideable(
          query_, key, value, attn_mask, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case SDPBackend::math: {
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          std::nullopt, /*dropout_mask*/
          scale,
          enable_gqa));
    }
    default:
      TORCH_CHECK(false, "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

} // namespace at::supa
