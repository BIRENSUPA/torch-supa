/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/GeneratorImpl.h>
#include <limits>
#include <unordered_set>

#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please note that most
 * of the places where we have inserted locking for generators are historically
 * based, and we haven't actually checked that everything is truly thread safe
 * (and it probably isn't). Please use the public mutex_ when using any methods
 * from these classes, except for the read-only methods.
 */

namespace at {

namespace supa {
struct SUPAGraph;
}
/**
 * Note [SUPA Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 *
 * A SUPA graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic below records
 * the total of all offset increments that occur in the graphed
 * region, and records the final total as the offset for the
 * entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device's SUPA generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxSupaStates with the uint64_t offset pulled
 * directly from the global state, PhiloxSupaState instead
 * holds a pointer to one-element stream-local int64_t device tensor
 * holding an initial offset value, and a uint64_t holding an
 * intra-graph offset. (The intra-graph offset starts from zero
 * when capture begins.)  In each consumer kernel,
 * at::supa::philox::unpack computes the offset to use for this kernel
 * as intra-graph offset + *initial offset.
 *
 * When the graph reruns, the logic that reruns it first
 * fill_s the initial offset tensor with this device's
 * SUPA generator's current offset.
 *
 * The control flow above ensures graphed execution is bitwise
 * identical to eager execution as long as RNG ops are enqueued
 * from a single thread, even if RNG ops and graphs containing
 * RNG ops are enqueued and run simultaneously on multiple streams.
 *
 * Usage:
 * ~~~~~~
 * PhiloxSupaState is NOT recommended for use in kernels. Because ops
 * in torch_supa are also used by DIOPI, which should not rely on Torch.
 * SUPAGeneratorImpl inherit from c10::GeneratorImpl.
 *
 * torch_supa uses philox_engine_inputs in frontend to get the pair of
 * seed and philox_offset, and update current philox_offset.
 *
 * Example (see e.g. torch_supa/csrc/aten/operators/pt_frontend_op/Random.cpp):
 *
 * #include <mutex>
 * #include "torch_supa/csrc/aten/core/SUPAGeneratorImpl.h"
 *
 * at::Tensor& SUPANativeFunctions::OP(..., c10::optional<at::Generator> gen_) {
 *    int64_t counter_offset = 256;  // should be caculated by nelem, block_size
 * or other args std::pair<uint64_t, uint64_t> rng_engine_inputs;
 *    {
 *        // See Note [Acquire lock when using random generators]
 *        std::lock_guard<std::mutex> lock(gen->mutex_);
 *        rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
 *    }
 *
 *    return PtOpFrontend("op_name")
 *        .setAttr("seed", std::get<0>(rng_engine_inputs))
 *        .setAttr("offset", std::get<1>(rng_engine_inputs))
 * }
 *
 */

// Stores state values. Passed as a kernel argument. See "Usage:" above.
struct PhiloxSupaState {
  PhiloxSupaState() : seed_{}, offset_{} {}
  PhiloxSupaState(const PhiloxSupaState&) = default;
  PhiloxSupaState& operator=(const PhiloxSupaState&) = default;
  PhiloxSupaState(PhiloxSupaState&&) = default;
  PhiloxSupaState& operator=(PhiloxSupaState&&) = default;
  ~PhiloxSupaState() = default;
  // Called if graph capture is not underway
  PhiloxSupaState(uint64_t seed, uint64_t offset) : seed_{}, offset_{} {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxSupaState(int64_t* seed, int64_t* offset_extragraph, uint32_t offset_intragraph)
      : seed_{}, offset_{}, offset_intragraph_(offset_intragraph), captured_(true) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
  }

  // Public members, directly accessible by at::Supa::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

struct SUPAGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
  uint32_t offset_intragraph_;
  bool capturing_{};
  std::unordered_set<supa::SUPAGraph*> registered_graphs_;
  at::TensorBase seed_extragraph_{};
  at::TensorBase offset_extragraph_{};

  SUPAGeneratorState(
      uint64_t seed = default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0,
      uint32_t offset_intragraph = 0)
      : seed_(seed), philox_offset_per_thread_(philox_offset_per_thread), offset_intragraph_(offset_intragraph) {}

  void increase(uint64_t increment);

  void register_graph(supa::SUPAGraph* graph);
  void unregister_graph(supa::SUPAGraph* graph);

  void capture_prologue();
  // capture_epilogue returns the wholegraph_increment
  uint64_t capture_epilogue();
  void replay_prologue(uint64_t wholegraph_increment);
  c10::intrusive_ptr<SUPAGeneratorState> clone();
};

struct TORCH_SUPA_API SUPAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  SUPAGeneratorImpl(DeviceIndex device_index = -1);
  SUPAGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<SUPAGeneratorState> state_);
  SUPAGeneratorImpl(const SUPAGeneratorImpl&) = delete;
  SUPAGeneratorImpl& operator=(const SUPAGeneratorImpl&) = delete;
  SUPAGeneratorImpl(SUPAGeneratorImpl&&) = delete;
  SUPAGeneratorImpl& operator=(SUPAGeneratorImpl&&) = delete;
  ~SUPAGeneratorImpl() override = default;

  // SUPAGeneratorImpl methods
  std::shared_ptr<SUPAGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
#if TORCH_VER >= TORCH_2_4_0
  void graphsafe_set_state(const c10::intrusive_ptr<GeneratorImpl>& state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;
#endif
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;

  void register_graph(supa::SUPAGraph* graph);
  void unregister_graph(supa::SUPAGraph* graph);

  // Generates a PhiloxSupaState with a specified increment, and increment
  // current state
  PhiloxSupaState philox_supa_state(uint64_t increment);

  bool reset_rnn_state() {
    return !no_reset_rnn_state_.test_and_set();
  }

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_supa_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static c10::DeviceType device_type();

 private:
  SUPAGeneratorImpl* clone_impl() const override;

  c10::intrusive_ptr<SUPAGeneratorState> state_;
  std::atomic_flag no_reset_rnn_state_{};
};

namespace supa::detail {

TORCH_SUPA_API const Generator& getDefaultSUPAGenerator(DeviceIndex device_index = -1);
TORCH_SUPA_API Generator createSUPAGenerator(DeviceIndex device_index = -1);

} // namespace supa::detail
} // namespace at
