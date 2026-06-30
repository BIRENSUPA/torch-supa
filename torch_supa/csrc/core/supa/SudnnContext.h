/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <sudnn/sudnn.h>

#include <torch_supa/csrc/core/supa/SUPAMacros.h>

namespace at::supa {

C10_SUPA_EXPORT sudnnHandle_t getSudnnHandle();

} // namespace at::supa
