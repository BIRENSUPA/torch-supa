/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/attention_backward.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/sdp_utils_cpp.h"
#ifdef USE_FLASH_ATTENTION
#include "torch_supa/csrc/aten/ops/kernels/transformers/supa/flash_attn/flash_api_for_torch.h"
#endif
namespace at::native {

std::tuple<Tensor, Tensor, Tensor> __flash_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

  const int non_null_window_left = -1;
  const int non_null_window_right = is_causal ? 0 : -1;

  std::optional<at::Tensor> dq{std::nullopt};
  std::optional<at::Tensor> dk{std::nullopt};
  std::optional<at::Tensor> dv{std::nullopt};

  //  The kernel computes irregardless we will drop for this functions return
  Tensor grad_softmax;

  // Currently unused args:
  std::optional<at::Tensor> alibi_slopes{std::nullopt};

  bool determinisitic{false};
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      determinisitic = true;
    }
  }

  auto [dQuery, dKey, dValue, dSoftmax] = pytorch_flash::mha_bwd(
      contiguous_grad_out,
      query,
      key,
      value,
      contiguous_out,
      logsumexp,
      dq,
      dk,
      dv,
      alibi_slopes,
      static_cast<float>(dropout_p),
      static_cast<float>(softmax_scale),
      is_causal,
      non_null_window_left,
      non_null_window_right,
      determinisitic);
  return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
}
} // namespace at::native