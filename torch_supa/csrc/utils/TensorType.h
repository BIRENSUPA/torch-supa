/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <torch/csrc/utils/tensor_new.h>

namespace torch_supa {
namespace utils {

// Initializes the Python tensor type objects: torch.supa.FloatTensor,
// torch.supa.DoubleTensor, etc. and binds them in their containing modules.
void initializePythonBindings();

} // namespace utils
} // namespace torch_supa
