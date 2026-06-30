/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/core/functional.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>

#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/supa/bccl.h"
#include "torch_supa/csrc/supa/device_set.h"

#include <bccl.h>

#include <sched.h>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#if (BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 13))
#define BCCL_HAS_REMOTE_ERROR 1
#endif

#if (BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 14))
#define BCCL_HAS_COMM_NONBLOCKING 1
#endif

bcclComm_t* to_bccl_comm(torch_supa::supa::bccl::bcclComm_t* var) {
  return reinterpret_cast<bcclComm_t*>(var);
}

bcclComm_t to_bccl_comm(torch_supa::supa::bccl::bcclComm_t var) {
  return reinterpret_cast<bcclComm_t>(var);
}

bcclUniqueId* to_bccl_unique_id(torch_supa::supa::bccl::bcclUniqueId* var) {
  return reinterpret_cast<bcclUniqueId*>(var);
}

bcclResult_t to_bccl_result(torch_supa::supa::bccl::bcclResult var) {
  switch (var) {
    case torch_supa::supa::bccl::bcclResult::Success:
      return bcclResult_t::bcclSuccess;
    case torch_supa::supa::bccl::bcclResult::UnhandledSupaError:
      return bcclResult_t::bcclUnhandledSupaError;
    case torch_supa::supa::bccl::bcclResult::SystemError:
      return bcclResult_t::bcclSystemError;
    case torch_supa::supa::bccl::bcclResult::InternalError:
      return bcclResult_t::bcclInternalError;
    case torch_supa::supa::bccl::bcclResult::InvalidArgument:
      return bcclResult_t::bcclInvalidArgument;
    case torch_supa::supa::bccl::bcclResult::InvalidUsage:
      return bcclResult_t::bcclInvalidUsage;
#ifdef BCCL_HAS_REMOTE_ERROR
    case torch_supa::supa::bccl::bcclResult::RemoteError:
      return bcclResult_t::bcclRemoteError;
#endif
#ifdef BCCL_HAS_COMM_NONBLOCKING
    case torch_supa::supa::bccl::bcclResult::InProgress:
      return bcclResult_t::bcclInProgress;
#endif
    case torch_supa::supa::bccl::bcclResult::NumResults:
      return bcclResult_t::bcclNumResults;
    default:
      throw std::runtime_error("Unconvertible BCCL type");
  }
}

torch_supa::supa::bccl::bcclResult from_bccl_result(bcclResult_t var) {
  switch (var) {
    case bcclSuccess:
      return torch_supa::supa::bccl::bcclResult::Success;
    case bcclUnhandledSupaError:
      return torch_supa::supa::bccl::bcclResult::UnhandledSupaError;
    case bcclSystemError:
      return torch_supa::supa::bccl::bcclResult::SystemError;
    case bcclInternalError:
      return torch_supa::supa::bccl::bcclResult::InternalError;
    case bcclInvalidArgument:
      return torch_supa::supa::bccl::bcclResult::InvalidArgument;
    case bcclInvalidUsage:
      return torch_supa::supa::bccl::bcclResult::InvalidUsage;
#ifdef BCCL_HAS_REMOTE_ERROR
    case bcclRemoteError:
      return torch_supa::supa::bccl::bcclResult::RemoteError;
#endif
#ifdef BCCL_HAS_COMM_NONBLOCKING
    case bcclInProgress:
      return torch_supa::supa::bccl::bcclResult::InProgress;
#endif
    case bcclNumResults:
      return torch_supa::supa::bccl::bcclResult::NumResults;
    default:
      throw std::runtime_error("Unconvertible BCCL type");
  }
}

bcclDataType_t to_bccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return bcclDataType_t::bcclFloat;
    case at::kHalf:
      return bcclDataType_t::bcclHalf;
    case at::kDouble:
      return bcclDataType_t::bcclDouble;
    case at::kLong:
      return bcclDataType_t::bcclInt64;
    case at::kInt:
      return bcclDataType_t::bcclInt;
    case at::kChar:
      return bcclDataType_t::bcclChar;
    // NOLINTNEXTLINE(*-narrowing-conversions, bugprone-branch-clone)
    case at::kByte:
      return bcclDataType_t::bcclUint8;
    case at::kBool:
      return bcclDataType_t::bcclUint8;
    case at::kFloat8_e4m3fn:
      return bcclDataType_t::bcclUint8;
    case at::kFloat8_e5m2:
      return bcclDataType_t::bcclUint8;
    case at::kFloat8_e4m3fnuz:
      return bcclDataType_t::bcclUint8;
    case at::kFloat8_e5m2fnuz:
      return bcclDataType_t::bcclUint8;

#if HAS_BCCL_BF16_DATATYPE
    case at::kBFloat16:
      return bcclDataType_t::bcclBfloat16;
#endif
    default:
      TORCH_CHECK(false, "Unconvertible BCCL type ", type);
  }
}

bcclDataType_t to_bccl_data_type(const at::Tensor& t) {
  if (!t.device().is_privateuseone()) {
    TORCH_CHECK(false, "BCCL only supports SUPA tensors, but got a tensor on ", t.device());
  }
  return to_bccl_data_type(t.scalar_type());
}

bcclRedOp_t to_bccl_red_op(int var) {
  return (bcclRedOp_t)(var);
}

namespace torch_supa::supa::bccl {

using namespace at;

namespace detail {

static void BCCL_CHECK(bcclResult_t result) {
  BCCL_CHECK(from_bccl_result(result));
}

bool bccl_use_nonblocking() {
  static bool bccl_use_nonblocking_ = []() {
    if (c10::utils::check_env("TORCH_BCCL_USE_COMM_NONBLOCKING")) {
      TORCH_WARN("Using experimental non-blocking BCCL communicator (TORCH_BCCL_USE_COMM_NONBLOCKING).");
      return true;
    }
    if (c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING")) {
      TORCH_WARN("Using experimental non-blocking BCCL communicator (TORCH_NCCL_USE_COMM_NONBLOCKING fallback).");
      return true;
    }
    return false;
  }();
  return bccl_use_nonblocking_;
}

// Default value: 30 minutes
static int bccl_nonblocking_timeout() {
  static int timeout = -2; // -2 means not initialized
  if (timeout == -2) {
    // E01643: use nccl env var first
    const auto val = c10::utils::get_env("TORCH_NCCL_NONBLOCKING_TIMEOUT");
    const auto val_b = c10::utils::get_env("TORCH_BCCL_NONBLOCKING_TIMEOUT");
    if (val && !val.value().empty()) {
      // NOLINTNEXTLINE(*-narrowing-conversions)
      timeout = strtol(val->c_str(), nullptr, 0);
    } else if (val_b && !val_b.value().empty()) {
      // NOLINTNEXTLINE(*-narrowing-conversions)
      timeout = strtol(val_b->c_str(), nullptr, 0);
    } else {
      // Default value consistent with kBackendDefaultTimeout
      timeout = 30 * 60;
    }
  }
  return timeout;
}

static void BCCL_CHECK_TIMEOUT(bcclResult status, bcclComm_t comm) {
#ifdef BCCL_HAS_COMM_NONBLOCKING
  bcclResult_t result = to_bccl_result(status);
  auto startTimepoint = std::chrono::steady_clock::now();
  while (result == bcclInProgress) {
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTimepoint - startTimepoint).count();
    if (timeElapsed > bccl_nonblocking_timeout()) {
      throw std::runtime_error("BCCL timeout when waiting for nonblocking call to become successful.");
    }
    sched_yield(); // yield to other threads
    bcclCommGetAsyncError(to_bccl_comm(comm), &result);
  }
  if (result != bcclSuccess) {
    throw_bccl_error(from_bccl_result(result));
  }
#else
  TORCH_INTERNAL_ASSERT(false, "BCCL COMM NONBLOCKING USED WITH UNSUPPORTED BCCL VERSION.");
#endif
}

static void BCCL_CHECK_TIMEOUT(bcclResult_t result, bcclComm_t comm) {
  BCCL_CHECK_TIMEOUT(from_bccl_result(result), comm);
}

static void BCCL_CHECK_TIMEOUT(bcclResult status, std::vector<bcclComm_t>& comms) {
#ifdef BCCL_HAS_COMM_NONBLOCKING
  bcclResult_t result = to_bccl_result(status);
  auto startTimepoint = std::chrono::steady_clock::now();
  if (result == bcclInProgress) {
    for (const auto i : c10::irange(comms.size())) {
      do {
        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTimepoint - startTimepoint).count();
        if (timeElapsed > bccl_nonblocking_timeout()) {
          throw std::runtime_error("BCCL timeout when waiting for nonblocking call to become successful.");
        }
        sched_yield(); // yield to other threads
        bcclCommGetAsyncError(to_bccl_comm(comms[i]), &result);
      } while (result == bcclInProgress);
      if (result != bcclSuccess) {
        break; /* fall through to failed case */
      }
    }
  }
  if (result != bcclSuccess) {
    throw_bccl_error(from_bccl_result(result));
  }
#else
  TORCH_INTERNAL_ASSERT(false, "BCCL COMM NONBLOCKING USED WITH UNSUPPORTED BCCL VERSION.");
#endif
}

static void BCCL_CHECK_TIMEOUT(bcclResult_t result, std::vector<bcclComm_t>& comms) {
  BCCL_CHECK_TIMEOUT(from_bccl_result(result), comms);
}

void throw_bccl_error(torch_supa::supa::bccl::bcclResult status) {
  std::ostringstream err;
  err << "BCCL Error " << static_cast<int>(status) << ": " << bcclGetErrorString(to_bccl_result(status));
  throw std::runtime_error(err.str());
}

struct BcclCommList {
  // NOLINTNEXTLINE(*array*)
  std::unique_ptr<bcclComm_t[]> comms;
  size_t ndevices;
  BcclCommList(const std::vector<int>& devices) : comms(new bcclComm_t[devices.size()]), ndevices(devices.size()) {
    BCCL_CHECK(bcclCommInitAll(to_bccl_comm(comms.get()), static_cast<int>(devices.size()), devices.data()));
  }
  BcclCommList(BcclCommList&& foo) = default;
  BcclCommList(const BcclCommList&) = delete;
  BcclCommList& operator=(const BcclCommList&) = delete;
  BcclCommList& operator=(BcclCommList&&) = delete;
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~BcclCommList() {
    if (comms) {
      for (const auto i : c10::irange(ndevices)) {
        int dummy_var = 0;
        if (C10_SUPA_ERROR_HANDLED(supaGetDevice(&dummy_var)) != supaSuccess) {
          /* there are cases when this destructor is called after the
           SUPA driver is already unloaded from the process.
           In these cases, skip bcclCommDestroy */
          return;
        }
        comm_destroy(comms[i]);
      }
    }
  }
  ArrayRef<bcclComm_t> ref() const {
    return ArrayRef<bcclComm_t>(comms.get(), ndevices);
  }
};

using device_list = std::vector<int>;
// accesses to this object have to be guarded by THC's SupaFreeMutex
static std::unordered_map<device_list, BcclCommList, c10::hash<device_list>> _communicators;

ArrayRef<bcclComm_t> get_communicators(TensorList inputs) {
  static auto get_device = [](const at::Tensor& t) -> int { return t.get_device(); };
  device_list devices = fmap(inputs, get_device);
  auto it = _communicators.find(devices);
  if (it == _communicators.end()) {
    it = _communicators.emplace(devices, devices).first;
  }
  return it->second.ref();
}

static void check_tensor(
    const at::Tensor& input,
    const std::optional<at::Tensor>& output,
    size_t input_multiplier,
    size_t output_multiplier,
    int64_t ref_numel,
    ScalarType ref_dtype) {
  auto check_one = [&](const at::Tensor& tensor) {
    if (!tensor.device().is_privateuseone() || tensor.is_sparse()) {
      throw std::runtime_error("input and output elements have to be supa dense Tensors");
    }

    if (ref_dtype != tensor.scalar_type()) {
      throw std::runtime_error("all inputs and outputs must be of the same Tensor dtype");
    }

    if (!tensor.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }
  };

  check_one(input);

  // all inputs must be same size
  if (input.numel() != ref_numel) {
    throw std::runtime_error("all inputs must have the same number of elements");
  }

  if (output) {
    check_one(*output);

    // inputs and outputs must be on same device respectively
    if (input.get_device() != output->get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    if (output->numel() * output_multiplier != ref_numel * input_multiplier) {
      throw std::runtime_error("output must be of size input_size * size_multiplier");
    }
  }
}

void check_inputs(TensorList inputs, TensorList outputs, size_t input_multiplier, size_t output_multiplier) {
  // len(inputs) == len(outputs)
  size_t len = inputs.size();

  if (len == 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but got input of length " << len
        << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (const auto i : c10::irange(len)) {
    const auto& input = inputs[i];
    auto output = outputs[i];

    check_tensor(input, output, input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

void check_inputs(TensorList inputs, const at::Tensor& output, int root, int input_multiplier, int output_multiplier) {
  auto len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (const auto i : c10::irange(len)) {
    const auto& input = inputs[i];

    check_tensor(
        input,
        i == static_cast<std::remove_cv_t<decltype(i)>>(root) ? std::optional<at::Tensor>{output} : std::nullopt,
        input_multiplier,
        output_multiplier,
        numel,
        dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

} // namespace detail

AutoBcclGroup::AutoBcclGroup() : comm_(nullptr), comm_nonblocking_(false) {
#if defined(BCCL_MAJOR) && (BCCL_MAJOR >= 2)
  detail::BCCL_CHECK(bcclGroupStart());
#endif
}

AutoBcclGroup::AutoBcclGroup(bcclComm_t comm, bool comm_nonblocking)
    : comm_(comm), comm_nonblocking_(comm_nonblocking) {
#if defined(BCCL_MAJOR) && (BCCL_MAJOR >= 2)
  detail::BCCL_CHECK(bcclGroupStart());
#endif
}

// NOLINTNEXTLINE(bugprone-exception-escape)
AutoBcclGroup::~AutoBcclGroup() noexcept(false) {
#if defined(BCCL_MAJOR) && (BCCL_MAJOR >= 2)
  if (comm_nonblocking_ && comm_ != nullptr) {
    detail::BCCL_CHECK_TIMEOUT(bcclGroupEnd(), comm_);
  } else {
    detail::BCCL_CHECK(bcclGroupEnd());
  }
#endif
}

bool is_available(TensorList tensors) {
#ifdef USE_BCCL
  device_set devices;
  for (const auto& tensor : tensors) {
    if (!tensor.device().is_privateuseone() || tensor.is_sparse()) {
      return false;
    }
    if (!tensor.is_contiguous()) {
      return false;
    }
    auto device = tensor.get_device();
    if (devices[device]) {
      return false;
    }
    devices[device] = true;
  }
  return true;
#else
  return false;
#endif
}

std::uint64_t version() {
#if defined(BCCL_MAJOR)
  constexpr std::uint64_t ver =
      (((uint64_t)BCCL_MAJOR) << 32) | (((uint64_t)BCCL_MINOR) << 16) | ((uint64_t)BCCL_PATCH);
  return ver;
#elif defined(USE_BCCL)
  // return major version "1"
  return ((uint64_t)1) << 32;
#else
  return 0;
#endif
}

const char* version_suffix() {
#if defined(BCCL_SUFFIX)
  return BCCL_SUFFIX;
#else
  return "";
#endif
}

void get_unique_id(bcclUniqueId& id) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  BCCL_CHECK(bcclGetUniqueId(to_bccl_unique_id(&id)));
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

bcclComm_t comm_init_rank(int nranks, const bcclUniqueId& comm_id, int rank) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  bcclComm_t comm = nullptr;
  bcclUniqueId id = comm_id;
  BCCL_CHECK(bcclCommInitRank(to_bccl_comm(&comm), nranks, *(to_bccl_unique_id(&id)), rank));
  return comm;
#else
  return nullptr;
#endif
}

void comm_destroy(bcclComm_t comm) {
  /*
   * Temporarily disable calling bcclCommDestroy
   * Calling bcclCommDestroy while program exiting is undefined,
   * and lead to segfault in BCCL 2
   * (whether it is called before or after the SUPA runtime destructor).
   * Temporarily disable it in destructor to avoid segfault.
   */
  return;

#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  BCCL_CHECK(bcclCommDestroy(to_bccl_comm(comm)));
#endif
}

namespace {
// BCCL changed the numerical type used for count between BCCL1 and BCCL2.
// So we use the following struct, which gets the type of the second argument
// of T, if T is a function type, with bcclBcast, to get that type statically
// and programmatically.

template <typename T>
struct GetSecondArgType;

template <typename R, typename Arg0, typename Arg1, typename... Args>
struct GetSecondArgType<R(Arg0, Arg1, Args...)> {
  typedef std::decay_t<Arg1> type;
};

constexpr auto count_max = std::numeric_limits<GetSecondArgType<decltype(bcclBcast)>::type>::max();

// Since BCCL 2.12.10, BCCL supports send/recv 0 byte:
// The issue of skipping send/recv
// is that it can cause deadlock when a rank send and recv 0 bytes so it's
// completely skipping the collective, causing mismatch across ranks
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR > 13)))
template <typename T>
constexpr bool _bccl_should_send_recv([[maybe_unused]] T _unused_) {
  return true;
}
#else
// old BCCL uses 0 byte message for synchronization
// Avoid send/recv when message size is zero
template <typename T>
inline bool _bccl_should_send_recv(T value) {
  return value != 0;
}
#endif
} // namespace

size_t get_max_count() {
  return count_max;
}

void broadcast(TensorList tensors, const stream_list& streams, const comm_list& user_comms) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  check_inputs(tensors, tensors, 1, 1);
  auto data_type = to_bccl_data_type(tensors[0]);
  int64_t numel = tensors[0].numel();

  const auto comms = user_comms.empty() ? get_communicators(tensors) : ArrayRef<bcclComm_t>(user_comms);

  AutoBcclGroup bccl_group_guard;
  c10::supa::OptionalSUPAGuard device_guard;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; i++) {
    auto device = tensors[i].get_device();
    device_guard.set_index(device);
    // Default to the current stream
    auto* const stream =
        (streams.empty() || !streams[i]) ? c10::supa::getCurrentSUPAStream(device).stream() : streams[i]->stream();
    TORCH_CHECK(
        static_cast<uint64_t>(numel) <= static_cast<uint64_t>(count_max),
        "Broadcast tensor has ",
        numel,
        " elements, which exceeds the "
        "maximum BCCL supports (",
        count_max,
        ")");
    bcclComm_t comm = comms[i];
    BCCL_CHECK(bcclBcast(tensors[i].data_ptr(), numel, data_type, 0, to_bccl_comm(comm), stream));
  }
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  TORCH_CHECK(root >= 0 && static_cast<size_t>(root) < inputs.size(), "invalid root");

  check_inputs(inputs, output, root, 1, 1);
  const auto len = inputs.size();

  auto data_type = to_bccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs) : ArrayRef<bcclComm_t>(user_comms);

  AutoBcclGroup bccl_group_guard;
  c10::supa::OptionalSUPAGuard device_guard;
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    auto* const stream =
        (streams.empty() || !streams[i]) ? c10::supa::getCurrentSUPAStream(device).stream() : streams[i]->stream();

    bcclComm_t comm = comms_ref[i];
    BCCL_CHECK(bcclReduce(
        inputs[i].data_ptr(),
        static_cast<std::remove_cv_t<decltype(i)>>(root) == i ? output.data_ptr() : nullptr,
        count,
        data_type,
        to_bccl_red_op(op),
        root,
        to_bccl_comm(comm),
        stream));
  }
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
  reduce(inputs, /*output=*/inputs[root], root, op, streams, user_comms);
}

void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  check_inputs(inputs, outputs, 1, 1);
  const auto len = inputs.size();

  auto data_type = to_bccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs) : ArrayRef<bcclComm_t>(user_comms);

  AutoBcclGroup bccl_group_guard;
  c10::supa::OptionalSUPAGuard device_guard;
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    auto* const stream =
        (streams.empty() || !streams[i]) ? c10::supa::getCurrentSUPAStream(device).stream() : streams[i]->stream();

    bcclComm_t comm = comms_ref[i];
    BCCL_CHECK(bcclAllReduce(
        inputs[i].data_ptr(), outputs[i].data_ptr(), count, data_type, to_bccl_red_op(op), to_bccl_comm(comm), stream));
  }
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, 1, len);

  auto data_type = to_bccl_data_type(inputs[0]);

  const auto count = inputs[0].numel() / len;
  auto comms_ref = user_comms.empty() ? get_communicators(inputs) : ArrayRef<bcclComm_t>(user_comms);

  AutoBcclGroup bccl_group_guard;
  c10::supa::OptionalSUPAGuard device_guard;
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    auto* const stream =
        (streams.empty() || !streams[i]) ? c10::supa::getCurrentSUPAStream(device).stream() : streams[i]->stream();

    bcclComm_t comm = comms_ref[i];
    BCCL_CHECK(bcclReduceScatter(
        inputs[i].data_ptr(), outputs[i].data_ptr(), count, data_type, to_bccl_red_op(op), to_bccl_comm(comm), stream));
  }
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_BCCL
  using namespace torch_supa::supa::bccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, len, 1);

  auto data_type = to_bccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs) : ArrayRef<bcclComm_t>(user_comms);

  AutoBcclGroup bccl_group_guard;
  c10::supa::OptionalSUPAGuard device_guard;
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    auto* const stream =
        (streams.empty() || !streams[i]) ? c10::supa::getCurrentSUPAStream(device).stream() : streams[i]->stream();

    bcclComm_t comm = comms_ref[i];
#if defined(BCCL_MAJOR) && (BCCL_MAJOR >= 2)
    BCCL_CHECK(
        bcclAllGather(inputs[i].data_ptr(), outputs[i].data_ptr(), count, data_type, to_bccl_comm(comm), stream));
#else
    BCCL_CHECK(
        bcclAllGather(inputs[i].data_ptr(), count, data_type, outputs[i].data_ptr(), to_bccl_comm(comm), stream));
#endif
  }
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;

  auto type = to_bccl_data_type(input);
  size_t count = input.numel() / size;
  [[maybe_unused]] size_t rankdiff = input.nbytes() / size;
  const auto* sendbuff = reinterpret_cast<const char*>(input.const_data_ptr());
  auto* recvbuff = reinterpret_cast<char*>(output.data_ptr());
  auto* comm = to_bccl_comm(_comm);
#if defined(USE_ROCM) || defined(BCCL_ALLTOALL_SUPPORTED)
  // BCCL_ALLTOALL_SUPPORTED is used so BCCL can differentiate send/recv
  // operations issued as a part of the collective (e.g. alltoall) vs those
  // inside traditional p2p operations.
  BCCL_CHECK(bcclAllToAll(sendbuff, recvbuff, count, type, comm, stream));
#else
  int numranks = 0;
  BCCL_CHECK(bcclCommCount(comm, &numranks));
  BCCL_CHECK(bcclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    if (_bccl_should_send_recv(count)) {
      BCCL_CHECK(bcclSend(sendbuff + r * rankdiff, count, type, r, comm, stream));
      BCCL_CHECK(bcclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream));
    }
  }
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclGroupEnd());
#else
  BCCL_CHECK_TIMEOUT(bcclGroupEnd(), _comm);
#endif
#endif
#else
  TORCH_CHECK(false, "all2all is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;

  auto type = to_bccl_data_type(_type);
  auto* comm = to_bccl_comm(_comm);
#if defined(USE_ROCM) || defined(BCCL_ALLTOALLV_SUPPORTED)
  // BCCL_ALLTOALLV_SUPPORTED is used so BCCL can differentiate send/recv
  // operations issued as a part of the collective (e.g. alltoallv) vs those
  // inside traditional p2p operations.
  BCCL_CHECK(
      bcclAllToAllv(sendbuff, sendcounts, senddispls, recvbuff, recvcounts, recvdispls, type, comm, stream.stream()));
#else
  int numranks = 0;
  BCCL_CHECK(bcclCommCount(comm, &numranks));
  BCCL_CHECK(bcclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    if (_bccl_should_send_recv(sendcounts[r])) {
      BCCL_CHECK(bcclSend(((char*)sendbuff) + senddispls[r] * size, sendcounts[r], type, r, comm, stream));
    }
    if (_bccl_should_send_recv(recvcounts[r])) {
      BCCL_CHECK(bcclRecv(((char*)recvbuff) + recvdispls[r] * size, recvcounts[r], type, r, comm, stream));
    }
  }
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclGroupEnd());
#else
  BCCL_CHECK_TIMEOUT(bcclGroupEnd(), _comm);
#endif
#endif
#else
  TORCH_CHECK(false, "all2all is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;
  auto* comm = to_bccl_comm(_comm);

  BCCL_CHECK(bcclGroupStart());
  for (const int r : c10::irange(static_cast<int>(outputTensors.size()))) {
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];

    if (_bccl_should_send_recv(input.numel())) {
      BCCL_CHECK(bcclSend(input.data_ptr(), input.numel(), to_bccl_data_type(input), r, comm, stream.stream()));
    }
    if (_bccl_should_send_recv(output.numel())) {
      BCCL_CHECK(bcclRecv(output.data_ptr(), output.numel(), to_bccl_data_type(output), r, comm, stream.stream()));
    }
  }
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclGroupEnd());
#else
  BCCL_CHECK_TIMEOUT(bcclGroupEnd(), _comm);
#endif
#else
  TORCH_CHECK(false, "all2all is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void send(const at::Tensor& input, bcclComm_t comm, c10::supa::SUPAStream stream, int dst) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(
      bcclSend(input.data_ptr(), input.numel(), to_bccl_data_type(input), dst, to_bccl_comm(comm), stream.stream()));
#else
  BCCL_CHECK_TIMEOUT(
      bcclSend(input.data_ptr(), input.numel(), to_bccl_data_type(input), dst, to_bccl_comm(comm), stream.stream()),
      comm);
#endif
#else
  TORCH_CHECK(false, "Send is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void recv(at::Tensor& output, bcclComm_t comm, c10::supa::SUPAStream stream, int src) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(
      bcclRecv(output.data_ptr(), output.numel(), to_bccl_data_type(output), src, to_bccl_comm(comm), stream.stream()));
#else
  BCCL_CHECK_TIMEOUT(
      bcclRecv(output.data_ptr(), output.numel(), to_bccl_data_type(output), src, to_bccl_comm(comm), stream.stream()),
      comm);
#endif
#else
  TORCH_CHECK(false, "Recv is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream,
    int32_t root) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;

  auto* comm = to_bccl_comm(_comm);
  int numranks = 0;
  int cur_rank = 0;
  BCCL_CHECK(bcclCommCount(comm, &numranks));
  BCCL_CHECK(bcclCommUserRank(comm, &cur_rank));

  size_t count = inputs.numel();
  auto type = to_bccl_data_type(inputs);
  const auto* sendbuff = reinterpret_cast<const char*>(inputs.const_data_ptr());

  BCCL_CHECK(bcclGroupStart());

  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());
        BCCL_CHECK(bcclRecv(recvbuff, count, type, r, comm, stream));
      } else {
        // on its own rank, simply copy from the input
        outputs[r].copy_(inputs);
      }
    }
  } else {
    BCCL_CHECK(bcclSend(sendbuff, count, type, root, comm, stream));
  }
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclGroupEnd());
#else
  BCCL_CHECK_TIMEOUT(bcclGroupEnd(), _comm);
#endif

#else
  TORCH_CHECK(false, "gather is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream,
    int32_t root) {
#ifdef USE_BCCL
#if defined(BCCL_MAJOR) && ((BCCL_MAJOR > 2) || ((BCCL_MAJOR == 2) && (BCCL_MINOR >= 7)))
  using namespace torch_supa::supa::bccl::detail;

  auto* comm = to_bccl_comm(_comm);
  int numranks = 0;
  int cur_rank = 0;
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclCommCount(comm, &numranks));
  BCCL_CHECK(bcclCommUserRank(comm, &cur_rank));
#else
  BCCL_CHECK_TIMEOUT(bcclCommCount(comm, &numranks), _comm);
  BCCL_CHECK_TIMEOUT(bcclCommUserRank(comm, &cur_rank), _comm);
#endif
  BCCL_CHECK(bcclGroupStart());
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        size_t send_count = inputs[r].numel();
        auto send_type = to_bccl_data_type(inputs[r]);
        const auto* sendbuff = reinterpret_cast<const char*>(inputs[r].const_data_ptr());
        BCCL_CHECK(bcclSend(sendbuff, send_count, send_type, r, comm, stream));
      } else {
        // on its own rank, simply copy it to the output
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    size_t recv_count = outputs.numel();
    auto recv_type = to_bccl_data_type(outputs);
    auto* recvbuff = reinterpret_cast<char*>(outputs.data_ptr());
    BCCL_CHECK(bcclRecv(recvbuff, recv_count, recv_type, root, comm, stream));
  }
#ifndef BCCL_HAS_COMM_NONBLOCKING
  BCCL_CHECK(bcclGroupEnd());
#else
  BCCL_CHECK_TIMEOUT(bcclGroupEnd(), _comm);
#endif
#else
  TORCH_CHECK(false, "scatter is only supported for BCCL lib version >= 2.7.0");
#endif
#else
  TORCH_CHECK(false, "PyTorch built without BCCL support");
#endif
}

} // namespace torch_supa::supa::bccl
