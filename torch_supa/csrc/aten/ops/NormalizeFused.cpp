/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

namespace at::supa {

void normalize_fused_kernel_launch(const float* input, float* output, int rows, int cols, float eps, int block_size);

namespace {

constexpr int kMaxNormalizeBlockSize = 256;

int select_block_size(int rows, int cols) {
  if (cols <= 128) {
    if (rows <= 8) {
      return 64;
    }
    if (rows <= 32) {
      return 128;
    }
    return kMaxNormalizeBlockSize;
  }
  return kMaxNormalizeBlockSize;
}

} // namespace

at::Tensor SUPANativeFunctions::normalize_fused(const at::Tensor& input, double eps) {
  torch_supa::profiler::SUPARecordFunction recorder;
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "normalize_fused expects a SUPA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "normalize_fused expects float32 input");
  TORCH_CHECK(input.dim() >= 1, "normalize_fused expects at least 1D input");
  TORCH_CHECK(input.is_contiguous(), "normalize_fused expects contiguous input");

  auto cols = static_cast<int>(input.size(input.dim() - 1));
  auto rows = static_cast<int>(input.numel() / cols);
  auto output = at::empty_like(input);
  normalize_fused_kernel_launch(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      cols,
      static_cast<float>(eps),
      select_block_size(rows, cols));
  return output;
}

} // namespace at::supa
