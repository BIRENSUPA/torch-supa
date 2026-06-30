/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

namespace torch_supa {
void to_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
void supa_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
} // namespace torch_supa
