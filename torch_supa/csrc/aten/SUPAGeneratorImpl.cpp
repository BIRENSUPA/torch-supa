/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2023-2025 Shanghai Biren Technology Co., Ltd.
 * All rights reserved.
 */

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/StreamGuard.h>

#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGraph.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"

namespace at {
namespace supa::detail {

namespace {

// Ensures we only call supaGetDeviceCount only once.
std::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
int64_t num_gpus;

// Ensures default_gens_supa is initialized once.
std::deque<std::once_flag> supa_gens_init_flag;

// Default, global SUPA generators, one per GPU.
std::vector<Generator> default_gens_supa;

/*
 * Populates the global variables related to SUPA generators
 * Warning: this function must only be called once!
 */
void initSUPAGenVector() {
  num_gpus = c10::supa::device_count();
  supa_gens_init_flag.resize(num_gpus);
  default_gens_supa.resize(num_gpus);
}

} // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultSUPAGenerator gets the default generator for a particular
 * supa device.
 */
const Generator& getDefaultSUPAGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initSUPAGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::supa::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  std::call_once(supa_gens_init_flag[idx], [&] {
    default_gens_supa[idx] = make_generator<SUPAGeneratorImpl>(idx);
    default_gens_supa[idx].seed();
  });
  return default_gens_supa[idx];
}

/**
 * Utility to create a SUPAGeneratorImpl. Returns a shared_ptr
 */
Generator createSUPAGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initSUPAGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::supa::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = make_generator<SUPAGeneratorImpl>(idx);
  auto* supa_gen = check_generator<SUPAGeneratorImpl>(gen);
  supa_gen->set_current_seed(default_rng_seed_val);
  supa_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace supa::detail

/**
 * Creates a clone of this SUPA Generator State.
 */
c10::intrusive_ptr<SUPAGeneratorState> SUPAGeneratorState::clone() {
  return make_intrusive<SUPAGeneratorState>(seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void SUPAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;
  // Handling different behaviors based on whether capturing is active.
  if (c10::supa::currentStreamCaptureStatus() != c10::supa::CaptureStatus::supaStreamCaptureStatusNone) {
    // Ensures that the state is actually capturing.
    TORCH_CHECK(capturing_, "Attempt to increase offset for a SUPA generator not in capture mode.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    // Ensures the increment does not cause overflow.
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    // Checks that the increment is expected outside graph capturing.
    TORCH_CHECK(!capturing_, "Offset increment outside graph capture encountered unexpectedly.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(philox_offset_per_thread_ % 4 == 0, "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Registers this state to a SUPA graph to manage within the graph.
 */
void SUPAGeneratorState::register_graph(supa::SUPAGraph* graph) {
  // Ensures that the RNG state is not currently being captured.
  c10::supa::assertNotCapturing("Cannot register the state during capturing stage.");

  // If this is the first graph to be registered, allocate memory for the seed
  // and offset on the GPU.
  if (registered_graphs_.empty()) {
    auto options = at::TensorOptions().device(at::kPrivateUse1).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  // Insert the graph into the set of registered graphs if it's not already
  // registered.
  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

/**
 * Unregisters a SUPA graph from the RNG state.
 */
void SUPAGeneratorState::unregister_graph(supa::SUPAGraph* graph) {
  // Verify the graph was previously registered.
  TORCH_CHECK(
      registered_graphs_.find(graph) != registered_graphs_.end(), "The graph should be registered to the state");

  // Remove the graph from the set of registered graphs.
  registered_graphs_.erase(graph);

  // If no more graphs are registered, deallocate the GPU memory for the seed
  // and offset.
  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Surand philox does allow offsets that aren't a multiple of 4.
 * But jit kernels don't use surand, they use a custom "Philox" class (see
 * torch/csrc/jit/tensorexpr/supa_random.h or
 * torch/csrc/jit/codegen/supa/runtime/random_numbers.cu).
 * The "Philox" constructor computes offset/4 (a uint64_t division) to locate
 * its internal start in its virtual bitstream viewed as 128-bit chunks, then,
 * when called in a thread, returns one 32-bit chunk at a time from that start
 * in the bitstream. In other words, if the incoming offset is not a multiple of
 * 4, each thread might repeat some previously-generated 32-bit values in the
 * bitstream. See https://github.com/pytorch/pytorch/pull/50169.
 */

/**
 * SUPAGeneratorImpl class implementation
 */
SUPAGeneratorImpl::SUPAGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl{
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          DispatchKeySet(c10::DispatchKey::PrivateUse1)} {
  c10::supa::assertNotCapturing("Cannot construct a new SUPAGeneratorImpl");
  state_ = make_intrusive<SUPAGeneratorState>();
  no_reset_rnn_state_.clear();
}

SUPAGeneratorImpl::SUPAGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<SUPAGeneratorState> state)
    : c10::
          GeneratorImpl{Device(c10::DeviceType::PrivateUse1, device_index), DispatchKeySet(c10::DispatchKey::PrivateUse1)},
      state_(std::move(state)) {
  no_reset_rnn_state_.clear();
}

/**
 * Sets the seed to be used by surandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void SUPAGeneratorImpl::set_current_seed(uint64_t seed) {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::set_current_seed");
  state_->seed_ = seed;
  state_->philox_offset_per_thread_ = 0;
  no_reset_rnn_state_.clear();
}

#define CAPTURE_DEFAULT_GENS_MSG                                               \
  "In regions captured by SUPA graphs, you may only use the default SUPA RNG " \
  "generator on the device that's current when capture begins. "               \
  "If you need a non-default (user-supplied) generator, or a generator on "    \
  "another "                                                                   \
  "device, please file an issue."

/**
 * Gets the current seed of SUPAGeneratorImpl.
 */
uint64_t SUPAGeneratorImpl::current_seed() const {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::current_seed");
  // Debatable if current_seed() should be allowed in captured regions.
  // Conservatively disallow it for now.
  return state_->seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and SUPA
 */
uint64_t SUPAGeneratorImpl::seed() {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::seed");
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the current internal state of SUPAGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> SUPAGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::get_state");
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto* rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(this->philox_offset_per_thread()); // Note that old THCGeneratorState had
                                                                        // offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of SUPAGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of SUPAGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void SUPAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed = 0;
  const auto* new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

#if TORCH_VER >= TORCH_2_4_0
/**
 * Sets the generator's current state to
 * This function allows switching between different registered states of
 * the generator.
 */
void SUPAGeneratorImpl::graphsafe_set_state(const c10::intrusive_ptr<GeneratorImpl>& gen) {
  c10::intrusive_ptr<SUPAGeneratorImpl> supa_gen = dynamic_intrusive_pointer_cast<SUPAGeneratorImpl>(gen);
  TORCH_CHECK(supa_gen, "Expected a SUPA Generator");
  state_ = supa_gen->state_;
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> SUPAGeneratorImpl::graphsafe_get_state() const {
  auto gen = make_intrusive<SUPAGeneratorImpl>(device().index(), state_);
  return gen;
}
#endif

/**
 * Sets the philox_offset_per_thread_ to be used by surandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void SUPAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of SUPAGeneratorImpl.
 */
uint64_t SUPAGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

/**
 * Registers this state to a SUPA graph to manage within the graph.
 */
void SUPAGeneratorImpl::register_graph(supa::SUPAGraph* graph) {
  graph->register_generator_state(state_);
  state_->register_graph(graph);
}

/**
 * Unregisters a SUPA graph from the RNG state.
 */
void SUPAGeneratorImpl::unregister_graph(supa::SUPAGraph* graph) {
  state_->unregister_graph(graph);
}

/**
 * Performs the prologue steps for capturing a SUPA graph state.
 * This method is intended to reset graph-related state variables before
 * capturing begins.
 */
void SUPAGeneratorState::capture_prologue() {
  capturing_ = true;
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(0));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t SUPAGeneratorState::capture_epilogue() {
  capturing_ = false;
  return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void SUPAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
  // Ensures the generator is not in capturing mode.
  c10::supa::assertNotCapturing("Cannot prepare for replay during capturing stage.");
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
  // Applies the total increment achieved during previous captures to update the
  // offset.
  increase(wholegraph_increment);
}

/**
 * Gets the seed and philox offset value to be used in
 * surandStatePhilox4_32_10, in an opaque PhiloxSupaState that's safe
 * and can be used non-divergently in callers whether SUPA graph
 * capture is underway or not.  See
 * Note [SUPA Graph-safe RNG states]
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of surand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of surand() calls. Increment
 * value > the number of surand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
PhiloxSupaState SUPAGeneratorImpl::philox_supa_state(uint64_t increment) {
  if (c10::supa::currentStreamCaptureStatus() != c10::supa::CaptureStatus::supaStreamCaptureStatusNone) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxSupaState(
        state_->seed_extragraph_.data_ptr<int64_t>(), state_->offset_extragraph_.data_ptr<int64_t>(), offset);
  }
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return PhiloxSupaState(state_->seed_, offset);
}

/**
 * Sets the offset to be used by surandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void SUPAGeneratorImpl::set_offset(uint64_t offset) {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::set_offset");
  set_philox_offset_per_thread(offset);
  no_reset_rnn_state_.clear();
}

/**
 * Gets the current offset of SUPAGeneratorImpl.
 */
uint64_t SUPAGeneratorImpl::get_offset() const {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::get_offset");
  // Debatable if get_offset() should be allowed in captured regions.
  // Conservatively disallow it for now.
  return state_->philox_offset_per_thread_;
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_supa_state.
 *
 * See Note [Acquire lock when using random generators]
 */
std::pair<uint64_t, uint64_t> SUPAGeneratorImpl::philox_engine_inputs(uint64_t increment) {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
}

/*
 * Gets the DeviceType of SUPAGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType SUPAGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<SUPAGeneratorImpl> SUPAGeneratorImpl::clone() const {
  return std::shared_ptr<SUPAGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
SUPAGeneratorImpl* SUPAGeneratorImpl::clone_impl() const {
  c10::supa::assertNotCapturing("Cannot call SUPAGeneratorImpl::clone_impl");
  auto* gen = new SUPAGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}

// this is used to register generator
at::Generator make_supa_generator(c10::DeviceIndex device_index) {
  c10::supa::assertNotCapturing("Not support Generator while in capture mode");
  return at::make_generator<SUPAGeneratorImpl>(device_index);
}

REGISTER_GENERATOR_PRIVATEUSE1(make_supa_generator)

} // namespace at
