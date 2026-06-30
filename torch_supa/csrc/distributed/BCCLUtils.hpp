/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>

#include <bccl.h>
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/supa/bccl.h"

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 14, 0)
#define BCCL_HAS_COMM_NONBLOCKING
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 18, 0)
#define BCCL_HAS_COMM_SPLIT
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 23, 0)
#define BCCL_HAS_INIT_RANK_SCALABLE
#endif

// bcclGetLastError() is enabled only for BCCL versions 2.13+
// bcclRemoteError only exists in BCCL versions 2.13+
#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 13, 0)
#define ENABLE_BCCL_GET_LAST_ERROR
#define BCCL_REMOTE_ERROR
#endif

static_assert(BCCL_VERSION_CODE >= BCCL_VERSION(2, 7, 0), "BCCL version must be 2.7 or later");
// The following macros represent features supported prior to BCCL 2.7,
// therefore we can define them unconditionally, given the static_assert above.
// TODO: remove these macros from code.
#define ENABLE_BCCL_ERROR_CHECKING
#define ENABLE_BCCL_P2P_SUPPORT
// End of macros for BCCL 2.7 and below.

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 11, 0)
#define ENABLE_BCCL_PREMUL_SUM_SUPPORT
#endif

// Note: the first version that supports bcclConfig_t is 2.14. Here we
// fast-forward the version requirement to 2.17 where bcclConfig_t has CTA and
// CGA fields because they have already been pybinded out.
#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 17, 0)
#define BCCL_HAS_CONFIG
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 19, 0)
#define BCCL_HAS_COMM_REGISTER
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 27, 0)
#define BCCL_HAS_COMM_WINDOW_REGISTER
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 19, 0)
#define BCCL_HAS_MEM_ALLOC
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 26, 0)
#define BCCL_HAS_QOS
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 24, 0)
#define BCCL_SUPPORTS_FP8
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 27, 0)
#define BCCL_HAS_COLLNET
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 27, 0)
#define BCCL_HAS_CTA_POLICY
#endif

#if BCCL_VERSION_CODE >= BCCL_VERSION(2, 27, 0)
#define BCCL_HAS_NVLS_CTAS
#endif

constexpr int64_t kCommInitBusyWaitMillis = 2;

// Macro to throw on a non-successful BCCL return value.
#define C10D_BCCL_CHECK(cmd, failureReason)                                                                 \
  do {                                                                                                      \
    bcclResult_t result = cmd;                                                                              \
    if (result != bcclSuccess) {                                                                            \
      std::string err = "BCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", " + \
          bcclGetErrorWithVersion(result) + "\n" + getBcclErrorDetailStr(result, failureReason);            \
      TORCH_CHECK_WITH(DistBackendError, false, err);                                                       \
    }                                                                                                       \
  } while (0)

// Macro to throw on a non-successful BCCL return value for NONBLOCKING calls.
#define C10D_BCCL_CHECK_NONBLOCKING(cmd, failureReason)                                                     \
  do {                                                                                                      \
    bcclResult_t result = cmd;                                                                              \
    if (result != bcclSuccess && result != bcclInProgress) {                                                \
      std::string err = "BCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", " + \
          bcclGetErrorWithVersion(result) + "\n" + getBcclErrorDetailStr(result, failureReason);            \
      TORCH_CHECK_WITH(DistBackendError, false, err);                                                       \
    }                                                                                                       \
  } while (0)

// Error out if (current time - startTime) is greater than timeout (sec).
#define C10D_CHECK_TIMEOUT(startTime, timeout)                                                            \
  do {                                                                                                    \
    auto currentTime = std::chrono::steady_clock::now();                                                  \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count(); \
    if (timeElapsed > timeout) {                                                                          \
      std::string err = "BCCL timeout in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__);     \
      TORCH_CHECK_WITH(DistBackendError, false, err);                                                     \
    }                                                                                                     \
  } while (0)

// Macro to throw on a non-successful BCCL return value, non-blocking.
#define C10D_BCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)                                    \
  do {                                                                                                      \
    bcclResult_t result = cmd;                                                                              \
    auto startTimepoint = std::chrono::steady_clock::now();                                                 \
    auto timeout = bccl_nonblocking_timeout();                                                              \
    while (result == bcclInProgress) {                                                                      \
      C10D_CHECK_TIMEOUT(startTimepoint, timeout);                                                          \
      yield_fn;                                                                                             \
      bcclCommGetAsyncError(comm, &result);                                                                 \
    }                                                                                                       \
    if (result != bcclSuccess) {                                                                            \
      std::string err = "BCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", " + \
          bcclGetErrorWithVersion(result) + "\n" + getBcclErrorDetailStr(result, failureReason);            \
      TORCH_CHECK_WITH(DistBackendError, false, err);                                                       \
    }                                                                                                       \
  } while (0)

// Sleep for kCommInitBusyWaitMillis milliseconds.
#define C10D_SCHED_SLEEP() std::this_thread::sleep_for(std::chrono::milliseconds(kCommInitBusyWaitMillis))

// Macro to throw exception on a non-successful BCCL return value or timeout.
// This macro uses sched_yield() to yield the CPU.
// Thus suitable for BCCL calls that would quickly turn bcclSuccess, e.g.
// collectives.
#define C10D_BCCL_CHECK_TIMEOUT(cmd, comm, failureReason) \
  C10D_BCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, sched_yield())

// Macro to throw exception on a non-successful BCCL return value or timeout.
// This macro uses sleep to yield the CPU.
// Thus suitable for BCCL calls that would take longer to turn bcclSuccess, e.g.
// bcclCommInitRankConfig, bcclCommFinalize, etc.
#define C10D_BCCL_CHECK_TIMEOUT_SLEEP(cmd, comm, failureReason) \
  C10D_BCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, C10D_SCHED_SLEEP())

#define C10D_BCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)                                          \
  do {                                                                                                      \
    bcclResult_t state = cmd;                                                                               \
    auto startTimepoint = std::chrono::steady_clock::now();                                                 \
    auto timeout = bccl_nonblocking_timeout();                                                              \
    if (state == bcclInProgress) {                                                                          \
      do {                                                                                                  \
        C10D_CHECK_TIMEOUT(startTimepoint, timeout);                                                        \
        sched_yield();                                                                                      \
        bcclCommGetAsyncError(comm->getBcclComm(), &state);                                                 \
      } while (state == bcclInProgress);                                                                    \
    }                                                                                                       \
    if (state != bcclSuccess) {                                                                             \
      std::string err = "BCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", " + \
          bcclGetErrorWithVersion(state) + "\n" + getBcclErrorDetailStr(state, failureReason);              \
      TORCH_CHECK_WITH(DistBackendError, false, err);                                                       \
    }                                                                                                       \
  } while (0)

// Macro to print and abort on a non-successful BCCL return value.
#define C10D_BCCL_ASSERT(cmd)                                                         \
  do {                                                                                \
    bcclResult_t result = cmd;                                                        \
    if (result != bcclSuccess) {                                                      \
      std::string err = bcclGetErrorWithVersion(result);                              \
      fprintf(stderr, "BCCL error in: %s:%d, %s\n", __FILE__, __LINE__, err.c_str()); \
      abort();                                                                        \
    }                                                                                 \
  } while (0)

namespace c10d::supa {

// BCCL type typing
static std::map<at::ScalarType, bcclDataType_t> bcclDataType = {
    {at::kChar, bcclInt8},
    {at::kByte, bcclUint8},
    {at::kFloat, bcclFloat},
    {at::kDouble, bcclDouble},
    {at::kInt, bcclInt32},
    {at::kLong, bcclInt64},
    {at::kHalf, bcclHalf},
    {at::kBool, bcclUint8},
#if TORCH_VER >= TORCH_2_4_0
#ifdef BCCL_SUPPORTS_FP8
    {at::kFloat8_e5m2, bcclFloat8e5m2},
    {at::kFloat8_e4m3fn, bcclFloat8e4m3},
#else
    {at::kFloat8_e5m2, bcclUint8},
    {at::kFloat8_e4m3fn, bcclUint8},
#endif
    {at::kFloat8_e4m3fnuz, bcclUint8},
    {at::kFloat8_e5m2fnuz, bcclUint8},
#endif
    {at::kBFloat16, bcclBfloat16},
};

TORCH_API size_t hashTensors(const std::vector<at::Tensor>& tensors);
TORCH_API int genBcclSplitColor(const std::vector<int>& ranks);
TORCH_API std::string getBcclVersion();
TORCH_API std::string bcclGetErrorWithVersion(bcclResult_t error);
int bccl_nonblocking_timeout();

// Provides additional detail into BCCL error codes based on when these are
// thrown in the BCCL codebase.
TORCH_API std::string getBcclErrorDetailStr(
    bcclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

// RAII wrapper for BCCL communicator
class BCCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit BCCLComm(bcclComm_t bcclComm) : bcclComm_(bcclComm) {}

  BCCLComm() = default;

  ~BCCLComm() noexcept {
    // (kwen2501) Making CUDA/BCCL calls in this destructor can hit CUDA driver
    // shutdown error if CUDA context has exited first. Thus, we are not
    // destroying or aborting BCCL communicators here. We just detect and warn
    // about the risk of memory leak. Normally, a user would have called
    // `destroy_process_group` or `abort_process_group`, and such risk would be
    // avoided.
    LockType lock(mutex_);
    if (bcclComm_ && initialized_ && !aborted_) {
      TORCH_WARN_ONCE(
          "WARNING: BCCL communicator hasn't been destroyed. This may cause "
          "memory leaks. To avoid the risk, you can call `destroy_process_group` "
          "during normal exit or `_abort_process_group` when handling failures.")
    }
  }

  static std::shared_ptr<BCCLComm> create(int numRanks, int rank, bcclUniqueId commId, at::DeviceIndex deviceIndex) {
    c10::supa::OptionalSUPAGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<BCCLComm>();
    C10D_BCCL_CHECK(bcclCommInitRank(&(comm->bcclComm_), numRanks, commId, rank), std::nullopt);
    comm->bcclId_ = commId;
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    comm->initialized_ = true;
    // Old style comm is always blocking.
    comm->nonBlocking_ = false;
    return comm;
  }

#ifdef BCCL_HAS_CONFIG
  static std::shared_ptr<BCCLComm> create(
      int numRanks,
      int rank,
      bcclUniqueId commId,
      at::DeviceIndex deviceIndex,
      bcclConfig_t& config) {
    c10::supa::OptionalSUPAGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<BCCLComm>();
    comm->nonBlocking_ = config.blocking == 0;
    LOG(INFO) << "Rank " << rank
              << ": creating BCCL communicator with mode: " << (comm->nonBlocking_ ? "nonblocking" : "blocking");
    C10D_BCCL_CHECK_NONBLOCKING(
        bcclCommInitRankConfig(&(comm->bcclComm_), numRanks, commId, rank, &config), std::nullopt);
    comm->bcclId_ = commId;
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    // Under blocking mode, comm is initialized immediately after BCCL init
    // returns; Under nonblocking mode, we check whether comm is initialized the
    // *next* time bcclComm_ is accessed.
    comm->initialized_ = !comm->nonBlocking_;
    return comm;
  }
#endif

#ifdef BCCL_HAS_COMM_SPLIT
  static std::shared_ptr<BCCLComm> split(
      BCCLComm* source,
      int color_id,
      int rank,
      bcclConfig_t& config,
      std::vector<uint64_t>& ranks_ull);
#endif

#if defined(IS_BCCLX) && defined(BCCL_COMM_DUMP)
  std::unordered_map<std::string, std::string> bcclCommDump() {
    std::unordered_map<std::string, std::string> dump;
    if (isAborted()) {
      LOG(INFO) << "Communicator was aborted before trying to dump its state.";
      return dump;
    }
    C10D_BCCL_CHECK(::bcclCommDump(bcclComm_, dump), std::nullopt);
    return dump;
  }
#endif

  at::DeviceIndex getDeviceIndex() const;

  bcclUniqueId getBcclId() {
    return bcclId_;
  }

  // Must not be copyable
  BCCLComm(const BCCLComm&) = delete;
  BCCLComm& operator=(const BCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  BCCLComm& operator=(BCCLComm&& other) = delete;

  // Move constructable
  // NOLINTNEXTLINE(*-noexcept-move-*)
  BCCLComm(BCCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    LockType lock(other.mutex_);
    std::swap(bcclComm_, other.bcclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(bcclAsyncErr_, other.bcclAsyncErr_);
    std::swap(initialized_, other.initialized_);
    std::swap(nonBlocking_, other.nonBlocking_);
    std::swap(deviceIndex_, other.deviceIndex_);
  }

  bcclComm_t getBcclComm();

  // Wait for the communicator to be ready. This is a blocking function.
  // Useful in nonblocking mode: BCCL requires the communicator to be ready
  // before issuing a second command.
  // Arguments:
  //   longInterval: if true, wait with sleep of an interval; otherwise, wait
  //   with `sched_yield` which is faster (but acquires CPU more frequently).
  //   Use `longInterval=true` when waiting for initialization or finalize to
  //   complete. Use `longInterval=false` when waiting collective call to return
  //   bcclSuccess.
  void waitReady(bool longInterval);

  std::optional<std::string> getBcclCommFailureReason() const {
    LockType lock(mutex_);
    return commFailureReason_;
  }

  void abort(std::optional<std::string> commFailureReason = std::nullopt) {
    LockType lock(mutex_);
    c10::supa::OptionalSUPAGuard gpuGuard(deviceIndex_);
#ifdef ENABLE_BCCL_ERROR_CHECKING
    if (aborted_ && !initialized_) {
      // Should not abort twice.
      return;
    }

#ifdef BCCL_HAS_COMM_REGISTER
    // Deregister all registered segments before aborting.
    for (auto& it : registeredSegmentHandles_) {
      void* handle = it.second;
      C10D_BCCL_CHECK(
          ::bcclCommDeregister(bcclComm_, handle),
          c10::str("Failed to deregister segment handle ", handle, " on bcclComm_ ", bcclComm_));
    }
    registeredSegmentHandles_.clear();
#endif

    // Set true failure reason if provided by ProcessGroupBCCL (e.g. work
    // timeout)
    commFailureReason_ = commFailureReason;
    LOG(INFO) << "Aborting bcclComm_ " << bcclComm_
              << " with reason: " << (commFailureReason ? *commFailureReason : "No abort reason provided.");
#ifndef BCCL_HAS_COMM_NONBLOCKING
    C10D_BCCL_CHECK(::bcclCommAbort(bcclComm_), commFailureReason_);
#else
    C10D_BCCL_CHECK_TIMEOUT(::bcclCommAbort(bcclComm_), bcclComm_, commFailureReason_);
#endif
    aborted_ = true;
    bcclComm_ = nullptr;

    // Set an appropriate error so that we avoid using the communicator.
    if (bcclAsyncErr_ == bcclSuccess) {
      bcclAsyncErr_ = bcclSystemError;
    }
#else
    // This is a NOOP, if error checks are disabled.
    return;
#endif
  }

  // Finalize a communicator -- asking it to flush its operations. When the
  // communicator is marked as nonblocking, this is a nonblocking function;
  // otherwise, it will block till all operations complete.
  void finalize();

  // Destroy a communicator. This is a blocking function.
  void destroy();

  bool isInitialized() const {
    LockType lock(mutex_);
    return initialized_;
  }

  bool isAborted() const {
    LockType lock(mutex_);
    return aborted_;
  }

  uint64_t getCommSplitCounter() const {
    return bcclCommSplitCounter_;
  }

  bcclResult_t checkForBcclError() {
    LockType lock(mutex_);
#ifdef ENABLE_BCCL_ERROR_CHECKING
    if (bcclAsyncErr_ != bcclSuccess) {
      return bcclAsyncErr_;
    }
    C10D_BCCL_CHECK(bcclCommGetAsyncError(bcclComm_, &bcclAsyncErr_), commFailureReason_);
    return bcclAsyncErr_;
#else
    // Always return success, if error checks are disabled.
    return bcclSuccess;
#endif
  }

  bcclResult_t registerSegment(void* ptr, size_t size, bool errorOnRereg = true, bool window = false);

  bcclResult_t deregisterSegment(void* ptr, bool window = false);

  std::string repr() const {
    return c10::str((void*)bcclComm_);
  }

  friend class ProcessGroupBCCL;

 protected:
  // Unique bccl_id for this communicator.
  bcclUniqueId bcclId_{};
  bool aborted_{false};
  uint64_t bcclCommSplitCounter_{0};
  bcclResult_t bcclAsyncErr_{bcclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_{};
  // Optional reason for communicator failure, provided by ProcessGroupBCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_{};
  bool initialized_{false};
  // Whether this communicator is using nonblocking mode. Recorded during comm
  // creation or split. For safety, we give a default value of true (more
  // protection).
  bool nonBlocking_{true};
  // Device index for which the BCCL comm is created
  at::DeviceIndex deviceIndex_{-1};
#ifdef BCCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by BCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif

 private:
  bcclComm_t bcclComm_{nullptr};
};

// Helper that automatically cleans up premul sums.
struct bcclRedOpRAII {
  bcclRedOpRAII() = default;
  bcclRedOpRAII(bcclRedOp_t op) : op_(op) {}
  bcclRedOpRAII(bcclRedOp_t op, bcclComm_t comm) : op_(op), comm_(comm), premul_sum_(true) {}
  bcclRedOpRAII(const bcclRedOpRAII&) = delete;
  bcclRedOpRAII& operator=(const bcclRedOpRAII&) = delete;
  bcclRedOpRAII(bcclRedOpRAII&& tmp) noexcept : bcclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
  bcclRedOpRAII& operator=(bcclRedOpRAII&&) = delete;
#if defined(ENABLE_BCCL_PREMUL_SUM_SUPPORT)
  ~bcclRedOpRAII() {
    if (premul_sum_) {
      bcclRedOpDestroy(op_, comm_);
    }
  }
#endif
  operator bcclRedOp_t() const {
    return op_;
  }
  bcclRedOp_t op_{};
  bcclComm_t comm_{};
  bool premul_sum_ = false;
};

} // namespace c10d::supa
