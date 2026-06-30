/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#elif TORCH_VER >= TORCH_2_6_0
#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>
#endif
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/distributed/Utils.hpp"

#include "torch_supa/csrc/distributed/symm_mem/IntraNodeComm.hpp"

namespace c10d::supa::intra_node_comm {

bool isIntraNodeCommSupported() {
  return true;
}

#if TORCH_VER < TORCH_2_6_0
std::optional<HybridCubeMesh> getHybridCubeMesh(BlMesh blMesh);

void* initP2pState();

void* initTopoInfo(Topology topology, BlMesh blMesh, size_t rank);
#endif

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {"ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without BLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};
static int intraNodeCommIdx = 0;

/**
 * Query the blink connection among devices.
 */
static BlMesh getBlMesh(const std::vector<int>& rankToDeviceIdx) {
#if TORCH_VER >= TORCH_2_6_0
  auto connectivity = detect_dma_connectivity(c10::DeviceType::PrivateUse1, "blink");
  BlMesh blMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() && dstRank < rankToDeviceIdx.size()) {
        blMesh[srcRank][dstRank] = connectivity->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return blMesh;
#else
  using namespace c10::supa;

  BlMesh blMesh = {};
  auto driverApi = DriverAPI::get();
  if (driverApi == nullptr) {
    return blMesh;
  }

  const auto worldSize = rankToBusId.size();
  std::vector<brmlDevice_t> devices(worldSize, nullptr);
  std::unordered_map<std::string, size_t> busIdToRank;
  std::vector<size_t> switchLinkCount(worldSize, 0);

  for (size_t r = 0; r < worldSize; ++r) {
    busIdToRank.emplace(rankToBusId[r], r);
    TORCH_CHECK(driverApi->brmlDeviceGetHandleByPciBusId_v2_(rankToBusId[r].c_str(), &devices[r]) == BRML_SUCCESS);
  }

  // TODO: find a better way to determine this
  constexpr size_t kMaxBLinks = 20;

  // For each device, loop over devices connected to it via NVLink
  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxBLinks; ++link) {
      brmlReturn_t ret;
      brmlIntBLinkDeviceType_t deviceType;
      ret = driverApi->brmlDeviceGetBLinkRemoteDeviceType_(devices[idx], link, &deviceType);
      if (ret != BRML_SUCCESS) {
        // We've exhausted the NVLinks connected to this device.
        // This error is benign. There doesn't seem to be a reliable
        // way to obtain the maximum link value that can be passed to
        // the API, so we simply increment the link value until the
        // API fails or we hit a predefined maximum value.
        break;
      }
      // Remote device is GPU
      if (deviceType == BRML_BLINK_DEVICE_TYPE_GPU) {
        brmlPciInfo_t pciInfo;
        ret = driverApi->brmlDeviceGetBLinkRemotePciInfo_v2_(devices[idx], link, &pciInfo);
        if (ret != BRML_SUCCESS) {
          // Unexpected error. Return an empty NvlMesh
          return {};
        }
        auto it = busIdToRank.find(pciInfo.busId);
        if (it != busIdToRank.end()) {
          if (idx != it->second) {
            blMesh[idx][it->second] += 1;
          }
        }
        // Remote device is NVSwitch
      } else if (deviceType == BRML_BLINK_DEVICE_TYPE_SWITCH) {
        switchLinkCount[idx] += 1;
      }
    }
  }
  // Process NVSwitch connections. For simplicity, we assume
  // all NVSwitches are interconnected.
  for (size_t i = 0; i < worldSize; ++i) {
    for (size_t j = 0; j < worldSize; ++j) {
      if (i == j) {
        continue;
      }
      blMesh[i][j] += std::min(switchLinkCount[i], switchLinkCount[j]);
    }
  }
  return blMesh;
#endif
}

/**
 * Detech topology given a BlMesh.
 */
static Topology detectTopology(const BlMesh blMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (blMesh[i][j] == 0 || blMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
};

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
#if TORCH_VER >= TORCH_2_6_0
  auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  allocator->free(symmetricMemoryPtr_);
#else
  for (size_t r = 0; r < worldSize_; ++r) {
    if (r == rank_) {
      continue;
    }
    C10_SUPA_CHECK(supaIpcCloseMemHandle(p2pStates_[r]));
    C10_SUPA_CHECK(supaIpcCloseMemHandle(buffers_[r]));
  }
  C10_SUPA_CHECK(supaFree(p2pStates_[rank_]));
  C10_SUPA_CHECK(supaFree(buffers_[rank_]));
  if (topoInfo_ != nullptr) {
    C10_SUPA_CHECK(supaFree(topoInfo_));
  }
  C10_SUPA_CHECK(supaFree(p2pStatesDev_));
  C10_SUPA_CHECK(supaFree(buffersDev_));
#endif
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << "-" << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(reinterpret_cast<uint8_t*>(&val), reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  if (!isIntraNodeCommSupported() || worldSize_ < 2 || worldSize_ > kMaxDevices) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  deviceIdx_ = c10::supa::current_device();

  // Exchange hostname and device bus ID
  struct DevInfo {
    // NOLINTNEXTLINE
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

  auto peerDevInfos = storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ")";
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  {
    std::unordered_set uniqueDeviceIdxs(rankToDeviceIdx.begin(), rankToDeviceIdx.end());
    if (uniqueDeviceIdxs.size() != worldSize_) {
      LOG(WARNING) << "Skipping IntraNodeComm::rendezvous() because participants have "
                      "overlapping devices. To resolve this, call torch.supa.set_device() "
                      "before init_process_group().";
      return false;
    }
  }

  // Query blink connection
  auto blMesh = getBlMesh(rankToDeviceIdx);

  // Detect topology
  topology_ = detectTopology(blMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }
#if TORCH_VER >= TORCH_2_6_0
  auto groupName = "IntraNodeComm" + std::to_string(intraNodeCommIdx++);
  set_group_info(groupName, static_cast<int>(rank_), static_cast<int>(worldSize_), store_);
  auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  symmetricMemoryPtr_ = allocator->alloc(bufferSize_, deviceIdx_, groupName);
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_, std::nullopt);
#else
  // Initialize p2p state
  auto p2pState = initP2pState();

  // Allocate buffer
  void* buffer = nullptr;
  C10_SUPA_CHECK(supaMalloc(&buffer, bufferSize_));

  // Second handshake: exchange topology and CUDA IPC handles
  struct IpcInfo {
    BlMesh blMesh;
    Topology topology;
    supaIpcMemHandle_t p2pStateHandle, bufferHandle;
  };

  // Make p2p state and buffer available for IPC
  supaIpcMemHandle_t p2pStateHandle, bufferHandle;
  C10_SUPA_CHECK(supaIpcGetMemHandle(&p2pStateHandle, p2pState));
  C10_SUPA_CHECK(supaIpcGetMemHandle(&bufferHandle, buffer));

  IpcInfo ipcInfo{
      .blMesh = blMesh, .topology = topology, .p2pStateHandle = p2pStateHandle, .bufferHandle = bufferHandle};

  auto peerIpcInfos = storeAllGather(store_, "handshake-1", rank_, worldSize_, ipcInfo);

  for (const auto& info : peerIpcInfos) {
    if (!isSame(info.blMesh, peerIpcInfos.front().blMesh) || info.topology != peerIpcInfos.front().topology) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are observing different topologies ("
                   << int(info.topology) << " and " << int(topology) << ")";
      C10_SUPA_CHECK(supaFree(p2pState));
      C10_SUPA_CHECK(supaFree(buffer));
      return false;
    }
  }

  std::array<void*, kMaxDevices> p2pStates = {}, buffers = {};
  for (size_t r = 0; r < peerIpcInfos.size(); ++r) {
    if (r == rank_) {
      p2pStates[r] = p2pState;
      buffers[r] = buffer;
    } else {
      C10_SUPA_CHECK(
          supaIpcOpenMemHandle(&p2pStates[r], peerIpcInfos[r].p2pStateHandle, supaIpcMemLazyEnablePeerAccess));
      C10_SUPA_CHECK(supaIpcOpenMemHandle(&buffers[r], peerIpcInfos[r].bufferHandle, supaIpcMemLazyEnablePeerAccess));
    }
  }
  void* p2pStatesDev = nullptr;
  C10_SUPA_CHECK(supaMalloc(&p2pStatesDev, sizeof(p2pStates)));
  C10_SUPA_CHECK(supaMemcpy(p2pStatesDev, p2pStates.data(), sizeof(p2pStates), supaMemcpyHostToDevice));

  void* buffersDev = nullptr;
  C10_SUPA_CHECK(supaMalloc(&buffersDev, sizeof(buffers)));
  C10_SUPA_CHECK(supaMemcpy(buffersDev, buffers.data(), sizeof(buffers), supaMemcpyHostToDevice));

  void* topoInfo = initTopoInfo(topology, blMesh, rank_);
#endif

  isInitialized_ = true;
  return true;
}

} // namespace c10d::supa::intra_node_comm
