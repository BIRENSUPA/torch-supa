/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once
#include <mutex>
#include <shared_mutex>

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/core/Device.h>

namespace torch_supa {
namespace utils {

inline bool is_supa(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }

  return tensor.device().is_privateuseone();
}

inline bool is_supa(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}

inline bool is_supa(const at::Device& device) {
  return device.is_privateuseone();
}

static std::shared_mutex env_mutex;

// Reads an environment variable and returns the content if it is set
inline std::optional<std::string> get_env(const char* name) noexcept {
  std::shared_lock lk(env_mutex);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto* envar = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  if (envar != nullptr) {
    return std::string(envar);
  }
  return std::nullopt;
}

} // namespace utils
} // namespace torch_supa
