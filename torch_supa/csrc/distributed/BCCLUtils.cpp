/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <cstring>
#include <mutex>

#include "torch_supa/csrc/distributed/BCCLUtils.hpp"
#include "torch_supa/csrc/utils/Utils.h"

namespace c10d::supa {
bcclComm_t BCCLComm::getBcclComm() {
  LockType lock(mutex_);
  if (aborted_) {
    auto commFailureMsg =
        commFailureReason_ != std::nullopt ? c10::str(" Original reason for failure was: ", *commFailureReason_) : "";
    TORCH_CHECK_WITH(
        DistBackendError, false, c10::str("BCCL communicator was aborted on rank ", rank_, ". ", commFailureMsg));
  }
  // In non-blocking mode, ensure comm is ready.
  if (nonBlocking_) {
    // Wait with long interval if communicator is being initialized.
    bool longInterval = !initialized_;
    waitReady(longInterval);
    // bcclComm_ should be initialized by now
  }
  if (!initialized_) {
    // TODO: see if we can consolidate other `initialized_` flipping here.
    // Maintaining it elsewhere is some work.
    initialized_ = true;
    LOG(INFO) << "Rank " << rank_ << ": BCCL communicator " << repr() << " is initialized.";
  }
  return bcclComm_;
}

at::DeviceIndex BCCLComm::getDeviceIndex() const {
  return deviceIndex_;
}

// Wait for the communicator to be ready. This is a blocking function.
// Arguments:
//   longInterval: if true, wait with sleep of an interval; otherwise, wait
//   with `sched_yield` which is faster (but acquires CPU more frequently).
void BCCLComm::waitReady(bool longInterval) {
  LockType lock(mutex_);
  if (aborted_) {
    return;
  }
  // If timeout is reached, throw an exception.
  if (longInterval) {
    C10D_BCCL_CHECK_TIMEOUT_SLEEP(bcclInProgress, bcclComm_, std::nullopt);
  } else {
    C10D_BCCL_CHECK_TIMEOUT(bcclInProgress, bcclComm_, std::nullopt);
  }
}

// last argument to split() API is not used to support
// multiple implementations
std::shared_ptr<BCCLComm> BCCLComm::split(
    BCCLComm* source,
    int color_id,
    int rank,
    bcclConfig_t& config,
    std::vector<uint64_t>& ranks_ull) {
  TORCH_CHECK(
      color_id >= BCCL_SPLIT_NOCOLOR,
      "Color must be a non-negative value or BCCL_SPLIT_NOCOLOR (-1)"
      ", but got ",
      color_id);
  LOG(INFO) << "Rank " << source->rank_ << ": split from parent comm " << source->repr() << " with color_id "
            << color_id << " and rank " << rank;
  c10::supa::OptionalSUPAGuard gpuGuard(source->deviceIndex_);
  auto comm = std::make_shared<BCCLComm>();
  // This call will block until the source communicator is initialized
  auto* sourceComm = source->getBcclComm();
#ifndef BCCL_HAS_COMM_NONBLOCKING
  C10D_BCCL_CHECK(bcclCommSplit(sourceComm, color_id, rank, &(comm->bcclComm_), &config), std::nullopt);
#else
  // After calling bcclCommSplit in non-blocking mode, we should wait for the
  // source communicator to be out of bcclInProgress state.
  // Reason 1:
  //   it's unsafe to call new operations on the parent comm while it's in
  //   bcclInProgress state.
  // Reason 2:
  //   as of BCCL 2.23, the ptr value of child comm will not be filled until the
  //   state of parent comm is bcclSuccess. This may change in the future. See:
  //   https://github.com/NVIDIA/nccl/issues/1472
  C10D_BCCL_CHECK_TIMEOUT_SLEEP(
      bcclCommSplit(sourceComm, color_id, rank, &(comm->bcclComm_), &config),
      sourceComm, // wait on parent comm
      std::nullopt);
  if (color_id >= 0) {
    // Waiting for parent comm above still does not seem to guarantee the child
    // comm ptr is valid. Therefore we add a manual wait here for safety.
    // TODO: remove this wait after BCCL fix the semantics.
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = bccl_nonblocking_timeout();
    while (!comm->bcclComm_) {
      C10D_CHECK_TIMEOUT(startTime, timeout);
      C10D_SCHED_SLEEP();
    }
  }
  // comm->bcclComm_ should have valid ptr by now, but not necessarily
  // initialized. Rely on getBcclComm() to wait for its initialization.
#endif
  ++source->bcclCommSplitCounter_;
  comm->rank_ = rank;
  // Child comm should be on the same device as parent comm
  comm->deviceIndex_ = source->deviceIndex_;
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << source->rank_ << ": created child comm " << comm->repr() << " with color_id " << color_id;
  return comm;
}

void BCCLComm::finalize() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_ << ": BCCL communicator already Invalidated. Skip finalize.";
    return;
  }
  c10::supa::OptionalSUPAGuard gpuGuard(deviceIndex_);
  auto* comm = getBcclComm();
  C10D_BCCL_CHECK_NONBLOCKING(bcclCommFinalize(comm), std::nullopt);
}

void BCCLComm::destroy() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_ << ": BCCL communicator already Invalidated. Skip destroy.";
    return;
  }
  c10::supa::OptionalSUPAGuard gpuGuard(deviceIndex_);
  auto* comm = getBcclComm();
  C10D_BCCL_CHECK(bcclCommDestroy(comm), std::nullopt);
  // Poison future getBcclComm
  aborted_ = true;
}

bcclResult_t BCCLComm::registerSegment(void* ptr, size_t size, bool errorOnRereg, bool window) {
  LockType lock(mutex_);
#ifdef BCCL_HAS_COMM_REGISTER
  // We register only segments from cache allocator
  // which are guaranteed to be with disjoint addr ranges. Thus, a ptr always
  // maps to a unique handle and should not be registered before the current
  // ptr is deregistered and freed.
  if (registeredSegmentHandles_.count(ptr) > 0) {
    TORCH_CHECK(!errorOnRereg, "Segment with ptr ", ptr, " has already been registered on bcclComm_ ", bcclComm_);
    // Skip below
    return bcclSuccess;
  }

  void* handle = nullptr;
  // Use getNcclComm to make sure comm is ready before calling bccl APIs
  auto* comm = getBcclComm();
#ifdef BCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_BCCL_CHECK(
        bcclCommWindowRegister(comm, ptr, size, (bcclWindow_t*)&handle, BCCL_WIN_COLL_SYMMETRIC),
        c10::str("Failed to window register segment with ptr ", ptr, ", size ", size, " on bcclComm_ ", comm));
  } else {
    C10D_BCCL_CHECK(
        bcclCommRegister(comm, ptr, size, &handle),
        c10::str("Failed to register segment with ptr ", ptr, ", size ", size, " on bcclComm_ ", comm));
  }
#else
  C10D_BCCL_CHECK(
      bcclCommRegister(comm, ptr, size, &handle),
      c10::str("Failed to register segment with ptr ", ptr, ", size ", size, " on bcclComm_ ", comm));
#endif
  registeredSegmentHandles_[ptr] = handle;
  return bcclSuccess;
#else
  return bcclInvalidUsage;
#endif
}

bcclResult_t BCCLComm::deregisterSegment(void* ptr, bool window /*false*/) {
  LockType lock(mutex_);
#ifdef BCCL_HAS_COMM_REGISTER
  TORCH_CHECK(
      registeredSegmentHandles_.count(ptr) == 1,
      "Segment with ptr ",
      ptr,
      " is not registered on bcclComm_ ",
      bcclComm_);

  void* handle = registeredSegmentHandles_[ptr];
  // Use getNcclComm to make sure comm is ready before calling bccl APIs
  auto* comm = getBcclComm();
#ifdef BCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_BCCL_CHECK(
        bcclCommWindowDeregister(comm, (bcclWindow_t)handle),
        c10::str("Failed to window deregister segment handle ", handle, ", with ptr ", ptr, " on bcclComm_ ", comm));
  } else {
    C10D_BCCL_CHECK(
        bcclCommDeregister(comm, handle),
        c10::str("Failed to deregister segment handle ", handle, ", with ptr ", ptr, " on bcclComm_ ", comm));
  }
#else
  C10D_BCCL_CHECK(
      bcclCommDeregister(comm, handle),
      c10::str("Failed to deregister segment handle ", handle, ", with ptr ", ptr, " on bcclComm_ ", comm));
#endif
  registeredSegmentHandles_.erase(ptr);
  return bcclSuccess;
#else
  return bcclInvalidUsage;
#endif
}

std::string getBcclVersion() {
  static std::once_flag bcclGetVersionFlag;
  static std::string versionString;

  std::call_once(bcclGetVersionFlag, []() {
    int version = 0;
    bcclResult_t status = bcclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != bcclSuccess || version < 100) {
      versionString = "Unknown BCCL version";
    } else {
      auto bcclMajor = version / 10000;
      auto bcclMinor = (version % 10000) / 100;
      auto bcclPatch = version % (bcclMajor * 10000 + bcclMinor * 100);
      versionString = std::to_string(bcclMajor) + "." + std::to_string(bcclMinor) + "." + std::to_string(bcclPatch);
    }
  });

  return versionString;
}

size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  size_t hash = 0;
  for (const auto& tensor : tensors) {
    if (tensor.numel() > 0 && tensor.storage()) {
      size_t data_size = tensor.storage().nbytes();
      if (data_size > 0 && tensor.storage().data_ptr()) {
        const auto* src = static_cast<const char*>(tensor.storage().data_ptr().get());
        std::vector<char> dst(data_size);
        // This is needed so that we trigger a device synchronization so we can
        // get the collective finished if launched on GPU and hash its output.
        supaMemcpy(dst.data(), src, data_size, supaMemcpyDeviceToHost);
        for (size_t i = 0; i < data_size; ++i) {
          // Update the hash for each byte in the tensor
          hash = c10::hash_combine(hash, c10::get_hash(dst[i], data_size));
        }
      }
    }
  }
  return hash;
}

// BCCL uses Non-negative int to represent in-group according to API
// requirement. We take a list of ranks and generate a hash value based on the
// list and ensure its range of 32-bit int.
int genBcclSplitColor(const std::vector<int>& ranks) {
  // Combine the hash values using a simple reducer (std::hash + fold)
  std::size_t combined_hash =
      std::accumulate(ranks.begin(), ranks.end(), std::size_t(0), [](std::size_t acc, int rank) {
        return acc ^ (std::hash<int>{}(rank) + 0x9e3779b9 + (acc << 6) + (acc >> 2));
      });

  // max positive value of int32_t
  constexpr int32_t max_c_int = std::numeric_limits<int32_t>::max();
  int color = static_cast<int>(std::abs(static_cast<int64_t>(combined_hash)) % max_c_int);
  return color;
}

// Default value: 30 minutes
int bccl_nonblocking_timeout() {
  static int timeout = -2; // -2 means not initialized
  if (timeout == -2) {
    const auto val = torch_supa::utils::get_env("BCCL_NONBLOCKING_TIMEOUT");
    if (val.has_value() && !val.value().empty()) {
      timeout = stoi(val.value());
    } else {
      // Default value consistent with kBackendDefaultTimeout
      timeout = 30 * 60;
    }
  }
  return timeout;
}

std::string bcclGetErrorWithVersion(bcclResult_t error) {
  return std::string(bcclGetErrorString(error)) + ", BCCL version " + getBcclVersion();
}

// Provides additional detail into BCCL error codes based on when these are
// thrown in the BCCL codebase.
std::string getBcclErrorDetailStr(
    bcclResult_t error,
    std::optional<std::string> processGroupFailureReason /* = std::nullopt */
) {
  // Prioritize failure reason provided by PG BCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != std::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_BCCL_GET_LAST_ERROR
  const auto* ret = bcclGetLastError(nullptr);
  if (ret) {
    err = "\nLast error:\n" + std::string(ret);
  } else {
    err = "\nLast error: Unknown BCCL Error\n";
  }
#endif
  switch (error) {
    case bcclUnhandledSupaError:
      interpret = "bcclUnhandledSupaError: Call to SUPA function failed.";
      break;
    case bcclSystemError:
      interpret =
          "bcclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
      // Before bcclRemoteError was created, unexpected remote disconnect was
      // categorized as bcclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
      break;
    case bcclInternalError:
      interpret = "bcclInternalError: Internal check failed.";
      break;
    case bcclInvalidArgument:
      interpret = "bcclInvalidArgument: Invalid value for an argument.";
      break;
    case bcclInvalidUsage:
      interpret = "bcclInvalidUsage: This usually reflects invalid usage of BCCL library.";
      break;
    case bcclRemoteError:
      interpret =
          "bcclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
    default:
      interpret = "Unknown BCCL error!";
  }
  return interpret + err;
}

} // namespace c10d::supa
