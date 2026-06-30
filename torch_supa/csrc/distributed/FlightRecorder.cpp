/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
// TODO: Make Fligth Recorder device agnostic
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#include <supa_runtime.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#if TORCH_VER >= TORCH_2_5_0
#include <c10/util/WaitCounter.h>
#endif

#include "FlightRecorder.hpp"
// #include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d::supa {
#if 0
control_plane::RegisterHandler dumpHandler{
    "dump_bccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setContent(
          dump_bccl_trace(
              processedParams[includeCollectivesStr],
              processedParams[includeStackTracesStr],
              processedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHandler{
    "dump_bccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setStatus(200);
      res.setContent(
          dump_bccl_trace_json(
              processedParams[includeCollectivesStr],
              processedParams[onlyActiveStr]),
          "application/json");
    }};
#endif
void DebugInfoWriter::write(const std::string& trace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing BCCLPG debug info: " << filename_;
    return;
  }

  file.write(trace.data(), static_cast<std::streamsize>(trace.size()));
  if (!file) {
    LOG(ERROR) << "Error opening file for writing BCCLPG debug info: " << filename_;
    return;
  }
  LOG(INFO) << "Finished writing BCCLPG debug info to " << filename_;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    std::string fileNamePrefix =
        getCvarString({"BCCL_DEBUG_INFO_TEMP_FILE", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/bccl_trace_rank_");
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(new DebugInfoWriter(fileNamePrefix, rank));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  TORCH_CHECK_WITH(DistBackendError, hasWriterRegistered_.load() == false, "debugInfoWriter already registered");
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

// Returns the traceback of current entry, in string form.
// Note: `getTraceback` invokes `torch::symbolize`, which may need to acquire
// the GIL. If you don't want to block the current thread or take the risk of a
// GIL deadlock, you can use an asynchronous calling mechanism like std::async.
std::string FlightRecorder::Entry::getTraceback() {
  torch::CapturedTraceback* traceback = traceback_.get();
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({traceback});
  // We use 0 because we only have one traceback here.
  const auto& s_tb = s_tbs.tracebacks.at(0);
  std::stringstream oss;
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    oss << "#" << idx << " " << frame.funcname << " from " << frame.filename << ":" << frame.lineno << '\n';
  }
  /* Resulted format is like:
    #0 all_reduce from pytorch/torch/distributed/distributed_c10d.py:2696
    #1 wrapper from pytorch/torch/distributed/c10d_logger.py:83
    #2 bar from /home/user/repro.py:15
    #3 foo from /home/user/repro.py:24
    #4 main from /home/user/repro.py:34
    #5 <module> from /home/user/repro.py:40
  */
  return oss.str();
}

std::optional<size_t> FlightRecorder::record(
    size_t pg_id,
    const std::tuple<std::string, std::string>& pg_name,
    size_t collective_seq_id,
    size_t p2p_seq_id,
    size_t op_id,
    std::string profiling_name,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    Event* start,
    Event* end,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<ProcessGroupStatus> pg_status,
    bool isP2P) {
  if (!enabled_) {
    return std::nullopt;
  }
  if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
    // Current pg_status is not in FR.
    all_pg_status_[pg_id] = std::move(pg_status);
  }
  auto traceback = torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
  std::lock_guard<std::mutex> guard(mutex_);

  auto te = Entry{
      id_,
      pg_id,
      pg_name,
      collective_seq_id,
      p2p_seq_id,
      op_id,
      std::move(profiling_name),
      std::move(traceback),
      start,
      end,
      c10::getTime(),
      timeout_ms.count(),
      isP2P,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      {},
      {},
      {},
      {},
      {},
      false};

  for (const auto& input : inputs) {
    c10::IntArrayRef sizes = input.sizes();
    te.input_dtypes_.push_back(input.dtype().toScalarType());
    te.input_dims_.push_back(static_cast<int64_t>(sizes.size()));
    te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
  }

  for (const auto& output : outputs) {
    c10::IntArrayRef sizes = output.sizes();
    te.output_dtypes_.push_back(output.dtype().toScalarType());
    te.output_dims_.push_back(static_cast<int64_t>(sizes.size()));
    te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
  }

  if (entries_.size() < max_entries_) {
    entries_.emplace_back(std::move(te));
  } else {
    entries_[next_++] = std::move(te);
    if (next_ == max_entries_) {
      next_ = 0;
    }
  }
  return id_++;
}

void FlightRecorder::record_pg_ranks(const std::tuple<std::string, std::string>& pg_name, std::vector<uint64_t> ranks) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  pg_name_to_ranks_[pg_name] = std::move(ranks);
}

void FlightRecorder::update_state(Entry& r) {
  if (r.start_ != nullptr) {
    bool started = r.start_->query();
    if (started && !r.time_discovered_started_) {
      r.time_discovered_started_ = c10::getTime();
    }
  }
  if (r.end_ != nullptr) {
    bool completed = r.end_->query();
    if (completed && !r.time_discovered_completed_) {
      r.time_discovered_completed_ = c10::getTime();
    }
  }
}

std::vector<FlightRecorder::Entry> FlightRecorder::dump_entries() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<Entry> result;
  result.reserve(entries_.size());
  result.insert(result.end(), entries_.begin() + static_cast<std::ptrdiff_t>(next_), entries_.end());
  result.insert(result.end(), entries_.begin(), entries_.begin() + static_cast<std::ptrdiff_t>(next_));
  // query any remaining events
  for (auto& r : result) {
    update_state(r);
    r.start_ = r.end_ = nullptr;
  }
  return result;
}

// Returns the entry with the given id, if it exists. Otherwise, returns
// std::nullopt.
std::optional<FlightRecorder::Entry> FlightRecorder::getEntry(std::optional<size_t> id) {
  if (!enabled_ || !id) {
    return std::nullopt;
  }

  std::unique_lock<std::mutex> guard(mutex_);
  Entry entry = entries_.at(*id % max_entries_);
  if (entry.id_ == *id) {
    return entry;
  }
  return std::nullopt;
}

void FlightRecorder::retire_id(std::optional<size_t> id, bool compute_duration) {
  if (!enabled_ || !id) {
    return;
  }

  bool can_compute_duration = false;
  Event* startEvent = nullptr;
  Event* endEvent = nullptr;
  std::optional<float> duration = std::nullopt;

  std::unique_lock<std::mutex> guard(mutex_);

  Entry* entry = &entries_.at(*id % max_entries_);
  if (entry->id_ == *id) {
    update_state(*entry);

    if (compute_duration) {
      can_compute_duration = entry->time_discovered_completed_.has_value() && entry->start_ && entry->end_;
      startEvent = entry->start_;
      endEvent = entry->end_;
    }
    entry->retired_ = true;
    entry->start_ = entry->end_ = nullptr;
  }

  if (can_compute_duration) {
    // Compute duration without without holding the lock, because
    // cudaEventDuration() can hang, and we need to acquire the lock before we
    // can dump(), which we never want to block.
    guard.unlock();
    duration = getDurationFromEvent(*startEvent, *endEvent);
    guard.lock();

    // Refresh the entry pointer, see if the entry has been overwritten
    entry = &entries_.at(*id % max_entries_);
    if (entry->id_ != *id) {
      LOG(INFO) << "retire_id abandoned for id " << *id << ", event was overwritten while waiting to compute duration.";
      return;
    }
    if (duration.has_value()) {
      entry->duration_ = duration;
    }
  }
}

c10::List<c10::IValue> FlightRecorder::getCollectiveTrace(bool includeStacktraces, bool onlyActive) {
  auto entries = new_list();
  // Entries are returned in the order they were recorded
  auto result = dump_entries();
  std::vector<torch::CapturedTraceback*> tracebacks;
  torch::SymbolizedTracebacks stracebacks;
  std::vector<c10::IValue> all_frames;
  if (includeStacktraces) {
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    stracebacks = torch::symbolize(tracebacks);
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_key, f.funcname);
      d.insert(filename_key, f.filename);
      d.insert(line_key, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }
  }
  for (auto i : c10::irange(result.size())) {
    auto dict = new_dict();
    auto& e = result.at(i);
    // Skip completed events
    if (onlyActive && e.time_discovered_completed_.has_value()) {
      continue;
    }
    if (includeStacktraces) {
      auto& tb = stracebacks.tracebacks.at(i);
      auto frames = new_list();
      for (auto frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_key, frames);
    }

    dict.insert(record_id_key, int64_t(e.id_));
    dict.insert(pg_id_key, int64_t(e.pg_id_));
    dict.insert(pg_name_key, e.pg_name_);
    dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
    dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
    dict.insert(op_id_key, int64_t(e.op_id_));
    dict.insert(profiling_name_key, e.profiling_name_);
    dict.insert(time_created_key, int64_t(e.time_created_));
    if (e.duration_) {
      dict.insert(duration_key, *e.duration_);
    }

    auto* it = e.sizes_.begin();
    auto read_sizes = [&](const c10::SmallVector<int64_t, 4>& dims) {
      auto sizes = new_list();
      for (auto dim : dims) {
        auto arg_sizes = new_list();
        for ([[maybe_unused]] auto i : c10::irange(dim)) {
          arg_sizes.push_back(*it++);
        }
        sizes.push_back(arg_sizes);
      }
      return sizes;
    };

    dict.insert(input_sizes_key, read_sizes(e.input_dims_));
    std::vector<std::string> input_dtypes_strs;
    input_dtypes_strs.reserve(e.input_dtypes_.size());
    for (const auto& input_dtype : e.input_dtypes_) {
      input_dtypes_strs.emplace_back(c10::toString(input_dtype));
    }
    dict.insert(input_dtypes_key, input_dtypes_strs);
    dict.insert(output_sizes_key, read_sizes(e.output_dims_));
    std::vector<std::string> output_dtypes_strs;
    output_dtypes_strs.reserve(e.output_dtypes_.size());
    for (const auto& output_dtype : e.output_dtypes_) {
      output_dtypes_strs.emplace_back(c10::toString(output_dtype));
    }
    dict.insert(output_dtypes_key, output_dtypes_strs);
    if (e.time_discovered_completed_.has_value()) {
      dict.insert(state_key, completed_state);
    } else if (e.time_discovered_started_.has_value()) {
      dict.insert(state_key, started_state);
    } else {
      dict.insert(state_key, scheduled_state);
    }

    dict.insert(
        time_discovered_started_key,
        e.time_discovered_started_.has_value() ? int64_t(*e.time_discovered_started_) : c10::IValue());
    dict.insert(
        time_discovered_completed_key,
        e.time_discovered_completed_.has_value() ? int64_t(*e.time_discovered_completed_) : c10::IValue());
    dict.insert(retired_key, e.retired_);
    dict.insert(timeout_key, e.timeout_ms_);
    dict.insert(is_p2p_key, e.isP2P_);

    entries.push_back(dict);
  }
  return entries;
}

c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgConfig() {
  auto pg_config = new_dict();
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = new_dict();
    pg_info.insert("name", std::get<0>(pg_name));
    pg_info.insert("desc", std::get<1>(pg_name));
    pg_info.insert("ranks", ranks_str(ranks));
    pg_config.insert(std::get<0>(pg_name), pg_info);
  }
  return pg_config;
}

std::map<std::string, std::map<std::string, std::string>> FlightRecorder::getPgConfigJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = std::map<std::string, std::string>();
    pg_info["name"] = std::get<0>(pg_name);
    pg_info["desc"] = std::get<1>(pg_name);
    pg_info["ranks"] = ranks_str(ranks);
    result.emplace(std::get<0>(pg_name), pg_info);
  }
  return result;
}

c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgStatus() {
  auto all_pg_status = new_dict();
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = new_dict();
    pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
    pg_status.insert("last_started_collective", status->lastStartedSeq);
    pg_status.insert("last_completed_collective", status->lastCompletedSeq);
    all_pg_status.insert(std::to_string(pg_id), pg_status);
  }
  return all_pg_status;
}

std::map<std::string, std::map<std::string, std::string>> FlightRecorder::getPgStatusJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = std::map<std::string, std::string>();
    pg_status["last_enqueued_collective"] = std::to_string(status->lastEnqueuedSeq);
    pg_status["last_started_collective"] = std::to_string(status->lastStartedSeq);
    pg_status["last_completed_collective"] = std::to_string(status->lastCompletedSeq);
    result[std::to_string(pg_id)] = pg_status;
  }
  return result;
}

static inline void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::string& value,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\" : \"" << value << "\"";
}

static inline void key_value(std::stringstream& ss, const std::string& key, int64_t value, char prefix = ',') {
  ss << prefix << '\"' << key << "\" : " << value;
}

static inline void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::map<std::string, std::string>& value_map,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\": {";
  prefix = ' ';
  for (auto it = value_map.begin(); it != value_map.end(); ++it) {
    key_value(ss, it->first, it->second, prefix);
    prefix = ',';
  }
  ss << '}';
}

static inline void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::tuple<std::string, std::string>& value_tuple,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\": [\"" << std::get<0>(value_tuple) << "\",\"" << std::get<1>(value_tuple) << "\"]";
}

static inline void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::vector<std::string>& value_list,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\": [";
  prefix = ' ';
  for (const auto& value : value_list) {
    ss << prefix << '\"' << value << "\"";
    prefix = ',';
  }
  ss << "]";
}

static inline void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::list<std::list<int64_t>>& values,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\": [";
  prefix = ' ';
  for (const auto& value : values) {
    ss << prefix << '[';
    char prefix2 = ' ';
    for (const auto& value2 : value) {
      ss << prefix2 << value2;
      prefix2 = ',';
    }
    ss << "]";
    prefix = ',';
  }
  ss << "]";
}

static void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::map<std::string, std::map<std::string, std::string>>& value_map,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\" : {";
  prefix = ' ';
  for (auto it = value_map.begin(); it != value_map.end(); ++it) {
    key_value(ss, it->first, it->second, prefix);
    prefix = ',';
  }
  ss << "}";
}

static void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::unordered_map<std::basic_string<char>, std::basic_string<char>>& value_map,
    char prefix = ',') {
  ss << '\"' << key << "\" : {";
  for (auto it = value_map.begin(); it != value_map.end(); ++it) {
    key_value(ss, it->first, it->second);
  }
  ss << "}";
}
static void key_value(
    std::stringstream& ss,
    const std::string& key,
    const std::unordered_map<
        std::basic_string<char>,
        std::unordered_map<std::basic_string<char>, std::basic_string<char>>>& value_map,
    char prefix = ',') {
  ss << prefix << '\"' << key << "\" : {";
  prefix = ' ';
  for (auto it = value_map.begin(); it != value_map.end(); ++it) {
    key_value(ss, it->first, it->second, prefix);
    prefix = ',';
  }
  ss << "}";
}

std::string FlightRecorder::dump_json(
    const std::optional<std::unordered_map<std::string, std::unordered_map<std::string, std::string>>>& bcclDumpMap,
    bool includeCollectives,
    bool onlyActive) {
  std::stringstream ss;
  ss << '{';
  key_value(ss, version_key_str, version_val_str, ' ');
  key_value(ss, pg_config_key_str, getPgConfigJson());
  key_value(ss, pg_status_key_str, getPgStatusJson());

  // collective trace
  if (includeCollectives) {
    auto entries_ = dump_entries();
    char prefix = ' ';
    if (!entries_.empty()) {
      ss << ",\"" << entries_key_str << "\":[";
      for (auto& e : entries_) {
        if (onlyActive && e.time_discovered_completed_.has_value()) {
          continue;
        }
        ss << prefix << '{';
        prefix = ',';
        key_value(ss, record_id_key_str, int64_t(e.id_), ' ');
        key_value(ss, pg_id_key_str, int64_t(e.pg_id_));
        key_value(ss, pg_name_key_str, e.pg_name_);
        key_value(ss, collective_seq_id_key_str, int64_t(e.collective_seq_id_));
        key_value(ss, p2p_seq_id_key_str, int64_t(e.p2p_seq_id_));
        key_value(ss, op_id_key_str, int64_t(e.op_id_));
        key_value(ss, profiling_name_key_str, e.profiling_name_);
        key_value(ss, time_created_key_str, int64_t(e.time_created_));
        if (e.duration_) {
          key_value(ss, duration_key_str, static_cast<int64_t>(*e.duration_));
        }

        auto* it = e.sizes_.begin();
        auto read_sizes = [&](const c10::SmallVector<int64_t, 4>& dims) {
          auto sizes = std::list<std::list<int64_t>>();
          for (auto dim : dims) {
            auto arg_sizes = std::list<int64_t>();
            for (auto i : c10::irange(dim)) {
              (void)i;
              arg_sizes.push_back(*it++);
            }
            sizes.push_back(arg_sizes);
          }
          return sizes;
        };
        key_value(ss, input_sizes_key_str, read_sizes(e.input_dims_));

        std::vector<std::string> input_dtypes_strs;
        input_dtypes_strs.reserve(e.input_dtypes_.size());
        for (const auto& input_dtype : e.input_dtypes_) {
          input_dtypes_strs.emplace_back(c10::toString(input_dtype));
        }
        key_value(ss, input_dtypes_key_str, input_dtypes_strs);
        key_value(ss, output_sizes_key_str, read_sizes(e.output_dims_));
        std::vector<std::string> output_dtypes_strs;
        output_dtypes_strs.reserve(e.output_dtypes_.size());
        for (const auto& output_dtype : e.output_dtypes_) {
          output_dtypes_strs.emplace_back(c10::toString(output_dtype));
        }
        key_value(ss, output_dtypes_key_str, output_dtypes_strs);
        if (e.time_discovered_completed_.has_value()) {
          key_value(ss, state_key_str, completed_state_str);
        } else if (e.time_discovered_started_.has_value()) {
          key_value(ss, state_key_str, started_state_str);
        } else {
          key_value(ss, state_key_str, scheduled_state_str);
        }
        key_value(
            ss,
            time_discovered_started_key_str,
            e.time_discovered_started_.has_value() ? int64_t(*e.time_discovered_started_) : 0);
        key_value(
            ss,
            time_discovered_completed_key_str,
            e.time_discovered_completed_.has_value() ? int64_t(*e.time_discovered_completed_) : 0);
        key_value(ss, retired_key_str, e.retired_);
        key_value(ss, timeout_key_str, e.timeout_ms_);
        key_value(ss, is_p2p_key_str, e.isP2P_);
        ss << '}';
      }
      ss << ']';
    }

    if (bcclDumpMap.has_value()) {
      key_value(ss, bccl_comm_key_str, bcclDumpMap.value());
    }
  }
  ss << "}";
  return ss.str();
}

std::string FlightRecorder::dump(
    const std::optional<std::unordered_map<std::string, std::unordered_map<std::string, std::string>>>& bcclDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
#if TORCH_VER >= TORCH_2_5_0
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
#endif
  auto result = new_dict();
  // common values
  result.insert(version_key, version_val);
  result.insert(pg_config_key, getPgConfig());
  result.insert(pg_status_key, getPgStatus());

  // collective trace
  if (includeCollectives) {
    result.insert(entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
  }
  // convert bcclDumpMap into a dictionary
  auto per_comm_dict = new_dict();
  if (bcclDumpMap.has_value()) {
    for (const auto& [bcclId, bcclDump] : bcclDumpMap.value()) {
      auto inner_dict = new_dict();
      for (const auto& [key, value] : bcclDump) {
        inner_dict.insert(key, value);
      }
      per_comm_dict.insert(bcclId, inner_dict);
    }
  }
  if (!per_comm_dict.empty()) {
    result.insert(bccl_comm_key, per_comm_dict);
  }
  return pickle_str(result);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

float getDurationFromEvent(c10::supa::SUPAEvent& bcclStartEvent, c10::supa::SUPAEvent& bcclEndEvent) {
  TORCH_CHECK(bcclEndEvent.query(), "getDuration can only be called after work is succeeded.")
  return bcclStartEvent.elapsed_time(bcclEndEvent);
}

} // namespace c10d::supa
