/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/util/Optional.h>
#include <string>
#include <torch/csrc/Export.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"

namespace torch_supa::supa {

// C++-only versions of these, for python use
// those defined in supa/Module.cpp which also record python state.
TORCH_SUPA_API void
_record_memory_history(bool enabled, bool record_context = true,
                       int64_t trace_alloc_max_entries = 1,
                       bool trace_alloc_record_context = false,
                       bool record_cpp_context = false);

TORCH_SUPA_API void
_record_memory_history(c10::optional<std::string> enabled = "all",
                       c10::optional<std::string> context = "all",
                       const std::string &stacks = "all",
                       size_t max_entries = SIZE_MAX);

TORCH_SUPA_API std::string _memory_snapshot_pickled();

} // namespace torch_supa::supa