/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <c10/core/TensorOptions.h>
#include <pybind11/pybind11.h>

#include <supa_runtime.h>

#include "torch_supa/csrc/utils/Utils.h"

namespace torch_supa {
namespace utils {

void supa_lazy_init();

void supa_set_run_yet_variable_to_false();

bool is_call_from_python();

inline void maybe_initialize_supa(const at::TensorOptions& options) {
  if (torch_supa::utils::is_supa(options) && is_call_from_python()) {
    torch_supa::utils::supa_lazy_init();
  }
}

inline void maybe_initialize_supa(const at::Device& device) {
  if (torch_supa::utils::is_supa(device) && is_call_from_python()) {
    torch_supa::utils::supa_lazy_init();
  }
}

inline void maybe_initialize_supa(const c10::optional<at::Device>& device) {
  if (!device.has_value()) {
    return;
  }
  maybe_initialize_supa(device.value());
}

void lazyInitSUPA();
} // namespace utils
} // namespace torch_supa
