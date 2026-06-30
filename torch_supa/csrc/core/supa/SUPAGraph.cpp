/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Functions.h>

#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/core/supa/MemPool.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGraph.h"
#include "torch_supa/csrc/core/supa/impl/SUPAGuardImpl.h"

#include <chrono>
#include <cstddef>
#include <thread>

namespace at::supa {

static bool _supa_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // from supaStreamGetCaptureInfo id_s in capture_begin.
  return c10::supa::MemPool::graph_pool_handle();
}

/**
 * Note [SUPA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native SUPA ops (like RNG ops
 * with CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [SUPA Graph-safe RNG states] in SUPAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write
 * capture bindings naively in an extension, they likely won't interact with the
 * native ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with SUPA graph capture] in SUPACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> SUPAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL
// watchdog so that they can be resolved before the capture begins. Note that
// event queries are not allowed during a graph capture in the default capture
// mode.
void SUPAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void SUPAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(
      pending_event_queries > 0,
      "Attempted to decrement the number of outstanding "
      "events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int SUPAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

SUPAGraph::SUPAGraph()
    // SUPAStreams may not be default-constructed.
    : capture_stream_(c10::supa::getCurrentSUPAStream()) {}

void SUPAGraph::register_generator_state(c10::intrusive_ptr<at::SUPAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void SUPAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<SUPAGeneratorImpl> supa_gen =
      dynamic_intrusive_pointer_cast<SUPAGeneratorImpl>(generator.getIntrusivePtr());
  supa_gen->register_graph(this);
}

void SUPAGraph::capture_begin(MempoolId_t pool /*=0*/, supaStreamCaptureMode capture_mode) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This SUPAGraph instance already owns a captured graph. "
      "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<SUPAGeneratorImpl>(c10::nullopt, supa::detail::getDefaultSUPAGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = c10::supa::getCurrentSUPAStream();

  TORCH_CHECK(
      stream != c10::supa::getDefaultSUPAStream(),
      "SUPA graphs must be captured on a non-default stream. "
      "(However, after capture, it's ok to replay them on the "
      "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::supa::current_device();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using
    // is_user_created=false. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = c10::supa::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // beginAllocateStreamToPool is now called before supaStreamBeginCapture to
  // prevent an autograd thread's free() call triggering an invalid
  // supaEventRecord in the caching allocator due to the capture status being
  // updated _after_ a capture had already started.
  c10::supa::SUPACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](supaStream_t stream) {
    supaStreamCaptureStatus status{};
    CaptureId_t stream_capture_id = 0;
    C10_SUPA_CHECK(supaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
    return status == supaStreamCaptureStatus::supaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });

  // At this point, any NCCL watchdogs should be aware that we are in capture
  // mode and therefore should not enqueue any additional work that could be
  // event-queried. We still must wait on any existing work that has not been
  // cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE(
        "Waiting for pending NCCL work to finish before starting "
        "graph capture.");
    std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // supaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe SUPA API calls during capture.
  C10_SUPA_CHECK(supaStreamBeginCapture(capture_stream_, capture_mode));

  supaStreamCaptureStatus status{};
  C10_SUPA_CHECK(supaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == supaStreamCaptureStatus::supaStreamCaptureStatusActive);
}

void SUPAGraph::capture_end() {
  auto stream = c10::supa::getCurrentSUPAStream();

  TORCH_CHECK(stream == capture_stream_, "Capture must end on the same stream it began on.");

  C10_SUPA_CHECK(supaStreamEndCapture(capture_stream_, &graph_));

  c10::supa::SUPACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
  has_graph_ = true;

  C10_SUPA_CHECK(supaGraphInstantiateWithFlags(&graph_exec_, graph_, supaGraphInstantiateFlagAutoFreeOnLaunch));

  has_graph_exec_ = true;

  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numSUPAGraphNodes = 0;
  C10_SUPA_CHECK(supaGraphGetNodes(graph_, nullptr, &numSUPAGraphNodes));
  if (numSUPAGraphNodes == 0) {
    TORCH_WARN(
        "The SUPA Graph is empty. This usually means that the graph was ",
        "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_supa_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    C10_SUPA_CHECK(supaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN(
        "DEBUG: TORCH_SUPAGRAPHS_DEBUG_PATH detected. graph_ will not "
        "be freed until debug_dump is called.");
  }
}

void SUPAGraph::replay() {
  TORCH_CHECK(has_graph_exec_, "Called SUPAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  C10_SUPA_CHECK(supaGraphLaunch(graph_exec_, c10::supa::getCurrentSUPAStream()));
}

void SUPAGraph::enable_debug_mode() {
  _supa_graphs_debug = true;
}

void SUPAGraph::debug_dump(const std::string& debug_path) {
  if (_supa_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling supaGraphDebugDotPrint() with ", debug_path);
      C10_SUPA_CHECK_WARN(supaGraphDebugDotPrint(
          graph_,
          debug_path.c_str(),
          supaGraphDebugDotFlagsVerbose)); // most verbose output
      C10_SUPA_CHECK(supaGraphDestroy(graph_));
      has_graph_ = false;
    }
  } else {
    TORCH_WARN(
        "SUPA Graphs debug not enabled, set with "
        "torch._C._supa_enable_graphs_debug_mode");
  }
}

void SUPAGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this
  // SUPAGraph, the generator, and the allocator could end up in all kinds of
  // weird states depending where failure occurred. If the user catches the
  // failure exception in a script, or is running in REPL or (god forbid) a
  // Jupyter notebook, I don't see an easy way for reset() to gracefully fix all
  // such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::supa::SUPACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_SUPA_CHECK_WARN(supaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_SUPA_CHECK_WARN(supaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory
// pool as this graph.
MempoolId_t SUPAGraph::pool() {
  TORCH_CHECK(has_graph_exec_, "Called SUPAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

SUPAGraph::~SUPAGraph() {
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

} // namespace at::supa