/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include "torch_supa/csrc/utils/Utils.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <fcntl.h>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace c10d::supa {

// A struct to hold the latest status of the process group.
struct ProcessGroupStatus {
  // the sequential number of the last collective enqueued into workMetaList_
  // This is useful for indentifying a rank that has not join a collective
  // initialized to be -1 to indicate no collective has been enqueued
  int64_t lastEnqueuedSeq{-1};
  // the sequential number of the last collective started as the kernel
  int64_t lastStartedSeq{-1};
  // the sequential number of the last collective completed marked by
  // the watchdog thread
  // initialized to be -1 to indicate no collective has been completed
  int64_t lastCompletedSeq{-1};

  // the name of the last collective enqueued into workMetaList_
  std::string lastEnqueuedWorkName;
  // the name of the last collective started as the kernel
  std::string lastStartedWorkName;
  // the name of the last collective completed
  std::string lastCompletedWorkName;

  // the sizes of the last work enqueued
  size_t lastEnqueuedNumelIn{0};
  size_t lastEnqueuedNumelOut{0};
  // the sizes of the last work completed
  size_t lastCompletedNumelIn{0};
  size_t lastCompletedNumelOut{0};
  // the sizes of the last work started
  size_t lastStartedNumelIn{0};
  size_t lastStartedNumelOut{0};
};

// TODO: support different types of failures/errors
enum class WorkResult : std::uint8_t {
  SUCCESS = 0,
  TIMEOUT = 1,
  COMM_ERROR = 2,
  UNKNOWN = 100,
};

inline std::vector<at::Tensor> getTensorShapes(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> shapeTensors;
  shapeTensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // Use `at::tensor()` to copy the data underlying `sizes()` since it may be
    // released elsewhere.
    at::Tensor shapesTensor = at::tensor(tensor.sizes(), at::TensorOptions().dtype(at::kLong));
    shapeTensors.emplace_back(std::move(shapesTensor));
  }
  return shapeTensors;
}

inline size_t getTensorsNumel(const std::vector<at::Tensor>& tensors) {
  size_t numel = 0;
  for (const auto& tensor : tensors) {
    numel += tensor.numel();
  }
  return numel;
}

#if TORCH_VER < TORCH_2_4_0
#define WARN_ENV_VAR_ONCE(deprecated_env, new_env) \
  TORCH_WARN_ONCE("Environment variable " + deprecated_env + " is deprecated; use " + new_env + " instead");
#endif

inline std::string getCvarString(const std::vector<std::string>& env, const char* def) {
  const char* ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    const char* val = std::getenv(env[i].c_str());
    if (val == nullptr) {
      continue;
    }
    if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]);
    }

    ret = val;
  }

  return ret;
}

inline int getCvarInt(const std::vector<std::string>& env, int def) {
  int ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    char* val = std::getenv(env[i].c_str());
    if (val == nullptr) {
      continue;
    }
    if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]);
    }

    try {
      ret = std::stoi(val);
    } catch (std::exception&) {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
    }
  }

  return ret;
}

inline bool getCvarBool(const std::vector<std::string>& env, bool def) {
  bool ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    char* val_ = std::getenv(env[i].c_str());
    if (val_ == nullptr) {
      continue;
    }
    if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]);
    }

    std::string val = std::string(val_);
    for (auto& x : val) {
      // NOLINTNEXTLINE(*-narrowing-conversions)
      x = std::tolower(x);
    }

    if (val == "y" || val == "yes" || val == "1" || val == "t" || val == "true") {
      ret = true;
    } else if (val == "n" || val == "no" || val == "0" || val == "f" || val == "false") {
      ret = false;
    } else {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
      return ret;
    }
  }

  return ret;
}

/***
 *  Trace Util
 */
inline std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) { result.insert(result.end(), data, data + size); };
    torch::jit::Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string ranks_str(const std::vector<uint64_t>& ranks) {
  std::string str;
  for (const auto& rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return c10::str("[", str, "]");
}

} // namespace c10d::supa
