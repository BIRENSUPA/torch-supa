/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

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
    std::optional<double> scale);
}