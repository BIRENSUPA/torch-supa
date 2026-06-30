/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

// TODO: Make Fligth Recorder device agnostic
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>

#include <ATen/ATen.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/logger.hpp>
#endif
#include <torch/csrc/distributed/c10d/TraceUtils.h>

#include "torch_supa/csrc/core/supa/SUPAEvent.h"
#include "torch_supa/csrc/distributed/Utils.hpp"

namespace c10d::supa {

#define DEFINE_CONSTANT(name, value) \
  static c10::IValue name = value;   \
  static std::string name##_str = value;
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
// Also update both JSON and Pickle dumps to make use of the newly defined
// field(s).
DEFINE_CONSTANT(version_val, "2.4")
DEFINE_CONSTANT(entries_key, "entries")
DEFINE_CONSTANT(bccl_comm_key, "bccl_comm_state")
DEFINE_CONSTANT(version_key, "version")
DEFINE_CONSTANT(pg_config_key, "pg_config")
DEFINE_CONSTANT(pg_status_key, "pg_status")
DEFINE_CONSTANT(record_id_key, "record_id")
DEFINE_CONSTANT(pg_id_key, "pg_id")
DEFINE_CONSTANT(pg_name_key, "process_group")
DEFINE_CONSTANT(collective_seq_id_key, "collective_seq_id")
DEFINE_CONSTANT(p2p_seq_id_key, "p2p_seq_id")
DEFINE_CONSTANT(is_p2p_key, "is_p2p")
DEFINE_CONSTANT(op_id_key, "op_id")
DEFINE_CONSTANT(profiling_name_key, "profiling_name")
DEFINE_CONSTANT(input_sizes_key, "input_sizes")
DEFINE_CONSTANT(input_dtypes_key, "input_dtypes")
DEFINE_CONSTANT(output_sizes_key, "output_sizes")
DEFINE_CONSTANT(output_dtypes_key, "output_dtypes")
DEFINE_CONSTANT(time_created_key, "time_created_ns")
DEFINE_CONSTANT(duration_key, "duration_ms")
DEFINE_CONSTANT(timeout_key, "timeout_ms")
DEFINE_CONSTANT(frames_key, "frames")
DEFINE_CONSTANT(state_key, "state")
DEFINE_CONSTANT(line_key, "line")
DEFINE_CONSTANT(name_key, "name")
DEFINE_CONSTANT(filename_key, "filename")
DEFINE_CONSTANT(retired_key, "retired")
DEFINE_CONSTANT(time_discovered_started_key, "time_discovered_started_ns")
DEFINE_CONSTANT(time_discovered_completed_key, "time_discovered_completed_ns")
DEFINE_CONSTANT(completed_state, "completed")
DEFINE_CONSTANT(scheduled_state, "scheduled")
DEFINE_CONSTANT(started_state, "started")
#undef DEFINE_CONSTANT

// Write NCCL debug info to local disk or any storage users define.
// There are some constrains we set for the debug info writer:
// 1. The writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recommended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class TORCH_API DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter() = default;
  DebugInfoWriter(const DebugInfoWriter&) = delete;
  DebugInfoWriter& operator=(const DebugInfoWriter&) = delete;
  DebugInfoWriter(DebugInfoWriter&&) = delete;
  DebugInfoWriter& operator=(DebugInfoWriter&&) = delete;
  virtual void write(const std::string& trace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  DebugInfoWriter(const std::string& namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);
  }
  std::string filename_;

 private:
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};

/* Helper used by work::getDuration() and flight recorder */
float getDurationFromEvent(c10::supa::SUPAEvent& bcclStartEvent, c10::supa::SUPAEvent& bcclEndEvent);

struct FlightRecorder {
  static FlightRecorder* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static FlightRecorder* instance = new FlightRecorder(); // NOLINT(cppcoreguidelines-owning-memory)
    return instance;
  }
  FlightRecorder() {
    max_entries_ = getCvarInt({"BCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool({"BCCL_TRACE_CPP_STACK", "TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  using Event = c10::supa::SUPAEvent;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    std::tuple<std::string, std::string> pg_name_; // <group_name, group_desc>

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    Event *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    int64_t time_created_;

    // configured timeout for this entry
    int64_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on CUDA APIs.
    std::optional<int64_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<int64_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int64_t, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int64_t, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out

    // Returns the traceback of current entry, in string form.
    std::string getTraceback();
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_ = {};
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>> pg_name_to_ranks_ = {};

  std::optional<size_t> record(
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
      bool isP2P);

  void record_pg_ranks(const std::tuple<std::string, std::string>& pg_name, std::vector<uint64_t> ranks);

  static void update_state(Entry& r);

  std::vector<Entry> dump_entries();

  // Returns the entry with the given id, if it exists. Otherwise, returns
  // std::nullopt.
  std::optional<Entry> getEntry(std::optional<size_t> id);

  /*
  Mark an Event as completed and free its events.
  This is called by the watchdog thread, and is asynchronous from the
  perspective of the main thread.
  compute_duration defaults to true since retire_id is only called in the
  watchdog thread, which is currently a place we call cuda APIs which may hang,
  but care should be taken to avoid computing duration in any function that must
  never hang. (timing must also be enabled for compute_duration - see
  TORCH_NCCL_ENABLE_TIMING).
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true);

  c10::List<c10::IValue> getCollectiveTrace(bool includeStacktraces, bool onlyActive);

  // dump pg_entries
  c10::Dict<c10::IValue, c10::IValue> getPgConfig();

  std::map<std::string, std::map<std::string, std::string>> getPgConfigJson();

  // dump pg_status
  c10::Dict<c10::IValue, c10::IValue> getPgStatus();

  std::map<std::string, std::map<std::string, std::string>> getPgStatusJson();

  std::string dump_json(
      const std::optional<std::unordered_map<std::string, std::unordered_map<std::string, std::string>>>& bcclDumpMap,
      bool includeCollectives,
      bool onlyActive);

  // dump all collectives + bcclDumpMap
  std::string dump(
      const std::optional<std::unordered_map<std::string, std::unordered_map<std::string, std::string>>>& bcclDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);
};

} // namespace c10d::supa
