/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>

#include <torch_supa/csrc/core/supa/SUPAStream.h>

namespace at {

struct Generator;
struct SUPAGeneratorImpl;
struct SUPAGeneratorState;

namespace supa {

using CaptureId_t = unsigned long long;
using CaptureStatus = supaStreamCaptureStatus;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// Standalone way to get a unique mempool id usable as a pool=... argument
// to SUPAGraph::capture_begin
TORCH_SUPA_API MempoolId_t graph_pool_handle();

struct TORCH_SUPA_API SUPAGraph {
  SUPAGraph();
  SUPAGraph(const SUPAGraph&) = delete;
  SUPAGraph& operator=(const SUPAGraph&) = delete;
  SUPAGraph(SUPAGraph&&) = delete;
  SUPAGraph& operator=(SUPAGraph&&) = delete;
  ~SUPAGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  // See Note [Explicit Registration of Generators to the SUPA Graph]
  void register_generator_state(c10::intrusive_ptr<at::SUPAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(MempoolId_t pool = {0, 0}, supaStreamCaptureMode capture_mode = supaStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

 protected:
  supaGraph_t graph_ = nullptr;
  supaGraphExec_t graph_exec_ = nullptr;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if supaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by
  // supaGraphInstantiate to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if supaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // the ID assigned by supa during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from
  // SUPACachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  c10::supa::SUPAStream capture_stream_;

  // multiple generator states and their wholegraph_increments in this graph
  // that are managed by the SUPA Graph
  ska::flat_hash_map<c10::intrusive_ptr<at::SUPAGeneratorState>, uint64_t> captured_generator_states_;

  // Device where capture occurred. Right now, for simplicity, we require all
  // ops in a capture to run on the same device, but this is a limitation of
  // SUPAGraph, not SUPA itself.  We can straightforwardly modify SUPAGraph to
  // support multi-device captures if needed. init capture_dev_ as
  // UNDEFINED_DEVICE to check that it stores the real device id in the
  // destructor
  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};
};

} // namespace supa
} // namespace at
