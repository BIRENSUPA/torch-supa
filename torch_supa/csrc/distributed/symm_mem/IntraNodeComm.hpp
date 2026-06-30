/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#elif TORCH_VER >= TORCH_2_5_0
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#endif

namespace c10d::supa::intra_node_comm {

using namespace c10d::symmetric_memory;

constexpr size_t kMaxDevices = 8;
constexpr size_t kDefaultBufferSize = 10ULL * 1024 * 1024;

using BlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;
using HybridCubeMesh = std::array<std::array<int, 4>, kMaxDevices>;

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
};

// NOTE: this class will be be removed soon in favor of SymmetricMemory
class TORCH_SUPA_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = std::nullopt);

  ~IntraNodeComm() override;
  IntraNodeComm(const IntraNodeComm&) = delete;
  IntraNodeComm& operator=(const IntraNodeComm&) = delete;
  IntraNodeComm(IntraNodeComm&&) = delete;
  IntraNodeComm& operator=(IntraNodeComm&&) = delete;

  static bool isEnabled();

  /**
   * Performs rendezvous.
   * If rendezvous fails, the IntraNodeComm object will be in an invalid
   * state and it is the caller's responsibility to dispose it.
   */
  bool rendezvous();

  /**
   * Selects a AllReduceAlgo that we think will outperform bccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform bccl.
   */
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

 private:
  at::Tensor oneShotAllReduce(const at::Tensor& input, c10::supa::SUPAStream& stream);

  at::Tensor twoShotAllReduce(const at::Tensor& input, c10::supa::SUPAStream& stream);

  c10::intrusive_ptr<Store> store_;
  size_t rank_;
  size_t worldSize_;
  size_t bufferSize_;

  /**
   * Members initialized after rendezvous
   */
  bool isInitialized_ = false;
  int deviceIdx_{0};
  Topology topology_ = Topology::UNKNOWN;
#if TORCH_VER >= TORCH_2_5_0
  void* symmetricMemoryPtr_ = nullptr;
  c10::intrusive_ptr<SymmetricMemory> symmetricMemory_ = nullptr;
#else
  std::array<void*, kMaxDevices> p2pStates_{};
  std::array<void*, kMaxDevices> buffers_{};
  void* p2pStatesDev_{};
  void* buffersDev_{};
  void* topoInfo_{};
#endif
};

class IntraNodeCommWork : public c10d::Work {
 public:
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }
};

TORCH_SUPA_API int64_t getIntraNodeCommUsageCounter();

bool isIntraNodeCommSupported();
} // namespace c10d::supa::intra_node_comm
