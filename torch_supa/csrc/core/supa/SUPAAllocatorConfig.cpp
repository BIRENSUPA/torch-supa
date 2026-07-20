/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPAAllocatorConfig.h"
#include "torch_supa/csrc/utils/logger/Logger.h"

#include <c10/util/llvmMathExtras.h>

namespace c10::supa::SUPACachingAllocator {

size_t SUPAAllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start = 63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
  const size_t interval_end = 63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
  TORCH_CHECK((interval_end - interval_start == kRoundUpPowerOfTwoIntervals), "kRoundUpPowerOfTwoIntervals mismatch");

  int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

  index = std::max(0, index);
  index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
  return instance().m_roundup_power2_divisions[index];
}

void SUPAAllocatorConfig::lexArgs(const char* env, std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void SUPAAllocatorConfig::consumeToken(const std::vector<std::string>& config, size_t i, const char c) {
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c), "Error parsing CachingAllocator settings, expected ", c, "");
}

size_t SUPAAllocatorConfig::parseMaxSplitSize(const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / (1024 * 1024),
        "");
    val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t SUPAAllocatorConfig::parseMaxNonSplitRoundingSize(const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_non_split_rounding_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_non_split_rounding_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_non_split_rounding_mb value", "");
  }
  return i;
}

size_t SUPAAllocatorConfig::parseGarbageCollectionThreshold(const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

void SUPAAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_supaMallocAsync = false;
  bool used_native_specific_option = false;
  if (env == nullptr) {
    return;
  }
  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "max_non_split_rounding_mb") {
      i = parseMaxNonSplitRoundingSize(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "expandable_segments") {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() && (std::string_view(config[i]) == "True" || std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for expandable_segments");
      m_expandable_segments = (config[i] == "True");
    } else if (
        // ROCm build's hipify step will change "supa" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        config_item_view == "release_lock_on_hipmalloc" || config_item_view == "release_lock_on_supamalloc" ||
        config_item_view == "release_lock_on_cudamalloc") {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() && (std::string_view(config[i]) == "True" || std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for "
          "release_lock_on_supamalloc");
      m_release_lock_on_supamalloc = (config_item_view == "True");
    } else if (
        // ROCm build's hipify step will change "cuda" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        config_item_view == "pinned_use_hip_host_register" || config_item_view == "pinned_use_supa_host_register" ||
        config_item_view == "pinned_use_cuda_host_register") {
      i = parsePinnedUseSupaHostRegister(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_num_register_threads") {
      i = parsePinnedNumRegisterThreads(config, i);
      used_native_specific_option = true;
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_supaMallocAsync && used_native_specific_option) {
    TORCH_SUPA_WARN(
        "backend:supaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

size_t SUPAAllocatorConfig::parsePinnedUseSupaHostRegister(const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for "
        "pinned_use_supa_host_register");
    m_pinned_use_supa_host_register = (config[i] == "True");
  } else {
    TORCH_CHECK(false, "Error, expecting pinned_use_supa_host_register value", "");
  }
  return i;
}

size_t SUPAAllocatorConfig::parsePinnedNumRegisterThreads(const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val2 = stoi(config[i]);
    TORCH_CHECK(llvm::isPowerOf2_64(val2), "Number of register threads has to be power of 2 ", "");
    auto maxThreads = SUPAAllocatorConfig::pinned_max_register_threads();
    TORCH_CHECK(
        val2 <= maxThreads,
        "Number of register threads should be less than or equal to " + std::to_string(maxThreads),
        "");
    m_pinned_num_register_threads = val2;
  } else {
    TORCH_CHECK(false, "Error, expecting pinned_num_register_threads value", "");
  }
  return i;
}

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  SUPACachingAllocator::SUPAAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace c10::supa::SUPACachingAllocator
