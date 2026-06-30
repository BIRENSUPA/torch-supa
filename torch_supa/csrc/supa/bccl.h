/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/ATen.h>
#include "torch_supa/csrc/core/supa/SUPAContext.h"

#include <cstddef>
#include <optional>
#include <vector>

#define HAS_BCCL_BF16_DATATYPE 1

namespace torch_supa::supa::bccl {

/* The following are copied from <bccl.h> and redefined in torch_supa::supa::bccl
 * namespace */
/* pytorch should only use the following definition within pytorch scope */

/* Opaque handle to communicator to bcclComm*, this will reinterpret as bcclComm
 * in bccl.cpp */
typedef void* bcclComm_t;

/** redefine bccl unique ID in torch scope. this should be identical to native
 * bccl impp. */
#define BCCL_UNIQUE_ID_BYTES 128
typedef struct {
  // NOLINTNEXTLINE(*array*)
  char internal[BCCL_UNIQUE_ID_BYTES];
} bcclUniqueId;

/* Error type */
enum class bcclResult {
  Success = 0,
  UnhandledSupaError = 1,
  SystemError = 2,
  InternalError = 3,
  InvalidArgument = 4,
  InvalidUsage = 5,
  RemoteError = 6,
  InProgress = 7,
  NumResults = 8
};

/* Reduction operation selector */
enum class bcclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, NumOps = 4 };

/* Data types */
enum class bcclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

// RAII helper class to manage BCCL group API and SUPA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct TORCH_SUPA_API AutoBcclGroup {
  AutoBcclGroup();
  AutoBcclGroup(bcclComm_t comm, bool comm_nonblocking);
  AutoBcclGroup(const AutoBcclGroup&) = delete;
  AutoBcclGroup& operator=(const AutoBcclGroup&) = delete;
  AutoBcclGroup(AutoBcclGroup&&) = delete;
  AutoBcclGroup& operator=(AutoBcclGroup&&) = delete;
  ~AutoBcclGroup() noexcept(false);
  bcclComm_t comm_;
  bool comm_nonblocking_;
};

// NOTE: this is exposed only so that python_bccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

TORCH_SUPA_API void throw_bccl_error(bcclResult status);

inline void BCCL_CHECK(bcclResult status) {
  if (status != bcclResult::Success) {
    throw_bccl_error(status);
  }
}

TORCH_SUPA_API at::ArrayRef<bcclComm_t> get_communicators(at::TensorList inputs);
TORCH_SUPA_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    size_t input_multiplier,
    size_t output_multiplier);
TORCH_SUPA_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    size_t input_multiplier,
    size_t output_multiplier);

} // namespace detail

using comm_list = std::vector<bcclComm_t>;
using stream_list = std::vector<std::optional<c10::supa::SUPAStream>>;

TORCH_SUPA_API std::uint64_t version();
TORCH_SUPA_API const char* version_suffix();

bool is_available(at::TensorList tensors);

TORCH_SUPA_API void get_unique_id(bcclUniqueId& id);
TORCH_SUPA_API bcclComm_t comm_init_rank(int nranks, const bcclUniqueId& comm_id, int rank);
TORCH_SUPA_API void comm_destroy(bcclComm_t comm);

TORCH_SUPA_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

size_t get_max_count();

TORCH_SUPA_API void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(bcclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_SUPA_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(bcclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_SUPA_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(bcclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_SUPA_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(bcclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_SUPA_API void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    bcclComm_t comm,
    c10::supa::SUPAStream& stream,
    int32_t root = 0);

TORCH_SUPA_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_SUPA_API void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    bcclComm_t comm,
    c10::supa::SUPAStream& stream,
    int32_t root = 0);

TORCH_SUPA_API void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    bcclComm_t comm,
    c10::supa::SUPAStream& stream);

TORCH_SUPA_API void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType type,
    bcclComm_t comm,
    c10::supa::SUPAStream& stream);

TORCH_SUPA_API void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    bcclComm_t _comm,
    c10::supa::SUPAStream& stream);

TORCH_SUPA_API void send(const at::Tensor& input, bcclComm_t comm, c10::supa::SUPAStream stream, int dst);

TORCH_SUPA_API void recv(at::Tensor& output, bcclComm_t comm, c10::supa::SUPAStream stream, int src);
} // namespace torch_supa::supa::bccl
