/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "torch_supa/csrc/profiler/util.h"

namespace torch_supa::profiler::impl {

std::unordered_map<std::string, std::string> saveBcclMeta(
    // @lint-ignore CLANGTIDY
    const at::RecordFunction& fn,
    // @lint-ignore CLANGTIDY
    const SaveBcclMetaConfig& config) {
  std::unordered_map<std::string, std::string> map;
#ifdef USE_DISTRIBUTED
  auto debugInfo =
      dynamic_cast<ParamCommsDebugInfo*>(c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));

  if (config.introspectMetadata) {
    if (debugInfo == nullptr) {
      LOG(WARNING) << "ParamCommsDebugInfo not available for function: " << fn.name();
      return map;
    }
    auto& collective_name = debugInfo->getCollectiveName();
    map.emplace(kCommsName, fmt::format("\"{}\"", collective_name));
    map.emplace(kDtype, fmt::format("\"{}\"", c10::toString(debugInfo->getDType())));
    map.emplace(kInMsgNelems, std::to_string(debugInfo->getInMessageNelems()));
    map.emplace(kOutMsgNelems, std::to_string(debugInfo->getOutMessageNelems()));

    auto& inSplitSizes = debugInfo->getInputSplitSizes();
    map.emplace(kInSplit, format_list(inSplitSizes, config.truncate));

    auto& outSplitSizes = debugInfo->getOutputSplitSizes();
    map.emplace(kOutSplit, format_list(outSplitSizes, config.truncate));

    auto globalRankStart = debugInfo->getGlobalRankStart();
    if (globalRankStart >= 0) {
      map.emplace(kGlobalRankStart, std::to_string(globalRankStart));
    }
    auto globalRankStride = debugInfo->getGlobalRankStride();
    if (globalRankStride > 0) {
      map.emplace(kGlobalRankStride, std::to_string(globalRankStride));
    }
    map.emplace(kGroupSize, std::to_string(debugInfo->getWorldSize()));
    auto& group_name = debugInfo->getProcessGroupName();
    if (!group_name.empty()) {
      map.emplace(kProcessGroupName, fmt::format("\"{}\"", group_name));
    }
    auto& group_desc = debugInfo->getProcessGroupDesc();
    if (!group_desc.empty()) {
      map.emplace(kProcessGroupDesc, fmt::format("\"{}\"", group_desc));
    }
    auto& groupRanks = debugInfo->getGroupRanks();
    map.emplace(kGroupRanks, format_list(groupRanks, config.truncate));

    auto rank = debugInfo->getRank();
    map.emplace(kRank, std::to_string(rank));
    int nRanks = static_cast<int>(groupRanks.size());
    if (collective_name == "send") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pDst, std::to_string(groupRanks[rank]));
      }
    } else if (collective_name == "recv") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pSrc, std::to_string(groupRanks[rank]));
      }
    }
  }

  if (get_record_tensor_addrs_enabled()) {
    std::vector<std::string> addressList;
    if (config.introspectInputs) {
      auto num_inputs = fn.num_inputs();
      const auto inputs = fn.inputs();
      if (checkFunctionInputsForLogging(fn)) {
        // need to account for Stack mode where the inputs are at the end.
        size_t input_start = inputs.size() - num_inputs;
        for (const auto i : c10::irange(input_start, inputs.size())) {
          const c10::IValue& val = inputs[i];
          auto [is_list, result] = findStartAddrForTensors(val);
          if (is_list) {
            auto const& list_result = std::get<std::vector<int>>(result);
            addressList.push_back(format_list(list_result, config.truncate, false));
          } else {
            auto scalar_result = std::get<int>(result);
            addressList.push_back(std::to_string(scalar_result));
          }
          // today we record a lot of metadata in record_param_comms that shows
          // up as inputs. here we only need the addresses of the first inputs,
          // which are the real tensor inputs to the collective call. So let's
          // break out of the loop here.
          break;
        }
        map.emplace(kInTensorsStart, format_list(addressList, false));
        addressList.clear();
      }
    }
    if (config.introspectOutputs) {
      const auto outputs = fn.outputs();
      auto num_outputs = fn.num_outputs();
      if (checkFunctionOutputsForLogging(fn)) {
        // need to account for Stack mode where the outputs are at the end.
        size_t output_start = outputs.size() - num_outputs;
        for (const auto i : c10::irange(output_start, outputs.size())) {
          const c10::IValue& val = outputs[i];
          auto [is_list, result] = findStartAddrForTensors(val);
          if (is_list) {
            auto const& list_result = std::get<std::vector<int>>(result);
            addressList.push_back(format_list(list_result, config.truncate, false));
          } else {
            auto scalar_result = std::get<int>(result);
            addressList.push_back(std::to_string(scalar_result));
          }
        }
        map.emplace(kOutTensorsStart, format_list(addressList, false));
        addressList.clear();
      }
    }
  }
#endif // USE_DISTRIBUTED
  return map;
}

} // namespace torch_supa::profiler::impl
