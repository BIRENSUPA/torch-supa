/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/native/ResizeCommon.h>

#include "torch_supa/csrc/aten/common/EmptyTensor.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace at::supa {

TORCH_SUPA_API void resize_bytes_supa(StorageImpl* storage, size_t size_bytes);

} // namespace at::supa
