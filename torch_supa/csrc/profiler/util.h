/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "record_function_wrapper.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace torch_supa::profiler::impl {

struct TORCH_SUPA_API SaveBcclMetaConfig {
  bool truncate;
  bool introspectMetadata;
  bool introspectInputs;
  bool introspectOutputs;

  // Default constructor with default values
  SaveBcclMetaConfig() : truncate(true), introspectMetadata(true), introspectInputs(false), introspectOutputs(false) {}

  SaveBcclMetaConfig(bool truncate, bool introspectMetadata, bool introspectInputs, bool introspectOutputs)
      : truncate(truncate),
        introspectMetadata(introspectMetadata),
        introspectInputs(introspectInputs),
        introspectOutputs(introspectOutputs) {}
};

std::unordered_map<std::string, std::string> TORCH_API
saveBcclMeta(const at::RecordFunction& fn, const SaveBcclMetaConfig& config = SaveBcclMetaConfig());

} // namespace torch_supa::profiler::impl
