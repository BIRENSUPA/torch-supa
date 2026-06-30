/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

// copy form pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
#include "torch_supa/csrc/aten/ops/kernels/transformers/SdpUtils.h"
#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/util/Array.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <torch_supa/csrc/core/supa/TorchVersion.h>
// cudaDeviceProp->major = 9 is defined by suda/csrc/host/src/cuda_runtime/cuda_runtime_br2xx_util.h
// Avoid potential compiler -Wall -Werror complains undefined macro

/**
 * Note [SDPA Runtime Dispatch]
 * SDPA relies on a runtime dispatch mechanism to select the appropriate
 * kernel. This file contains exposes this through the `select_sdp_backend`
 * The basic structure of this function is to call `priority_order` to get a
 * list of backends to try, and then iterate through them until one succeeds.
 * Each backend defines a use_<backend> function that returns true if the
 * backend can be run with the given SDP parameters. The use_<backend> function
 * will iterate over a list of "filters" that check for specific properties of
 * the SDP parameters. If all filters pass, the backend can be used and use_<backend>
 * returns true. If any filter fails, then use_<backend> returns false.
 *
 * In order to aid in debugging, each filter takes sdp_params and a debug flag.
 * If the debug flag is set, the filter will print a warning message if it fails.
 * The behavior of select_sdp_backend is to return the first backend that
 * succeeds. If no backend is viable then it will run each use_<backend> function
 * with debug=true and return SDPBackend::error.
 */

namespace sdp {
namespace {

// flash_attention V2 is universally faster than efficient_attention and Math
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  return at::globalContext().sDPPriorityOrder();
}

int64_t minimum_gemm_alignment(sdp_params const& params) {
  bool is_half = (params.query.dtype() == at::kHalf) || (params.query.dtype() == at::kBFloat16);
  bool use_tc = true;
  int64_t matmul_alignment_mn = 4;
  int64_t bits_per_scalar = is_half ? 16 : 32;
  if (use_tc) {
    matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);
  }
  return matmul_alignment_mn;
}

// On ROCM, ME and FA share the backend, and hence they share the checking
// function for fundamental limitations by the GPU kernel
// caller_is_meff is added to make the TORCH_WARN message showing the correct result
template <bool caller_is_meff = false>
bool check_head_dim_size_flash(sdp_params const& params, bool debug) {
  // All head_dim sizes must be equal and less than or equal to 512
  const auto max_size = c10::SymInt(512);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size = query_size_last == key_size_last && query_size_last == value_size_last;
  // for third-party/flash-attention, head_dim must be a multiple of 8
  if (!(same_head_dim_size && (query_size_last % 8 == 0) && (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          caller_is_meff ? "Efficient attention on ROCM" : "Flash attention",
          " requires q,k,v to have the same last dimension and to be a multiple of 8 and to be less than or equal to 512.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

// See check_head_dim_size_flash above for the purpose of caller_is_meff
template <bool caller_is_meff = false>
bool check_head_dim_size_flash_nested(sdp_params const& params, bool debug) {
  const auto max_size = c10::SymInt(512);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size = query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) && (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs,",
          caller_is_meff ? " Efficient attention on ROCM " : " Flash attention",
          " requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 512.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_mem_efficient(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
  if (!(query_size_last == params.key.sym_size(-1) && query_size_last % alignment == 0 && query_size_last > 0 &&
        value_size_last % alignment == 0 && value_size_last > 0)) {
    if (debug) {
      TORCH_WARN(
          "Mem efficient attention requires last dimension of inputs to be divisible by ",
          alignment,
          ". ",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_flash_attention_hardware_support(sdp_params const& params, bool debug) {
  // Flash attention supports hardware in the range [sm_80, sm_121]
  return true;
}

bool check_mem_efficient_hardware_support(sdp_params const& params, bool debug) {
  // Mem Efficient attention supports hardware in the range [sm_50, sm_90]
  // E01643: currently mem_efficient just use flash_attn, which may mismatch
  return false;
}

bool check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120(sdp_params const& params, bool debug) {
  // Flash Attention will raise an error in the backward pass if the head_dim
  // size is greater than 192 And the device is between in the range [sm86, sm89]
  return true;
}

bool check_flash_causal_non_square_seqlens(sdp_params const& params, bool debug) {
  // FlashAttention 2 updated the default mask meaning for causal in this PR:
  // 9e5e8bc91e it is now aligned to lower_right which would be a BC break
  // for non-square masks. We will not support non-square masks for causal w/ FAV2
  if (params.is_causal && !params.query.is_nested() && !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ",
          params.query.sym_size(-2),
          " seqlen_k: ",
          params.key.sym_size(-2),
          ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // Check that all tensors are on the GPU device
  // This should be handled by the stub dispatch, but whe call can_use_*_attention
  // directly from python we need to ensure that the tensors are on cuda
  if (params.query.device().type() != c10::DeviceType::PrivateUse1) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on cuda device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

bool check_cudnn_tensor_shapes(sdp_params const& params, bool debug) {
  constexpr int cudnn_version = 91002; // defined by suda
  const auto s_q = params.query.sym_size(2);
  const auto s_k = params.key.sym_size(2);
  const auto d_qk = params.query.sym_size(3);
  const auto d_v = params.value.sym_size(3);
  // sudnn head_dim_limit is 256
  auto head_dim_limit = 256;
  if (d_qk > head_dim_limit || d_v > head_dim_limit) {
    if (debug) {
      TORCH_WARN("head_dim should be no more than ", head_dim_limit);
    }
    return false;
  }
  if (d_qk % 8 != 0 || d_v % 8 != 0) {
    if (debug) {
      TORCH_WARN("head_dim should be a multiple of 8");
    }
    return false;
  }
  if (cudnn_version < 8906 && s_k % 64 != 0) {
    if (debug) {
      TORCH_WARN("not-multiple-of-64 seq_kv is not supported below 8.9.6");
    }
    return false;
  }
  if (cudnn_version < 90000) {
    if (s_q < 64) {
      if (debug) {
        TORCH_WARN("s_q less than 64 is not supported before cudnn 9.0.0");
      }
      return false;
    }
    if (params.dropout != 0.0 && (s_q % 64 != 0 || s_k % 64 != 0)) {
      if (debug) {
        TORCH_WARN("s_q not a multiple of 64 with padding/dropout is not supported with cudnn version 9.0.0");
      }
      return false;
    }
  }
  return true;
}

bool check_cudnn_layout(sdp_params const& params, bool debug) {
  const int64_t h = params.query.size(1);
  const int64_t s_q = params.query.size(2);
  const int64_t d = params.query.size(3);
  const int64_t s_k = params.key.size(2);
  const int64_t s_v = params.value.size(2);
  // corresponds to cuDNN's "packed QKV" layout
  const bool packed_query_layout_ok = (params.query.stride(0) == s_q * 3 * h * d) && (params.query.stride(1) == d) &&
      (params.query.stride(2) == 3 * h * d) && (params.query.stride(3) == 1);
  const bool packed_key_layout_ok = (params.key.stride(0) == s_k * 3 * h * d) && (params.key.stride(1) == d) &&
      (params.key.stride(2) == 3 * h * d) && (params.key.stride(3) == 1);
  const bool packed_value_layout_ok = (params.value.stride(0) == s_v * 3 * h * d) && (params.value.stride(1) == d) &&
      (params.value.stride(2) == 3 * h * d) && (params.value.stride(3) == 1);

  const bool packed_layout_ok = packed_query_layout_ok && packed_key_layout_ok && packed_value_layout_ok;

  const bool query_layout_ok = (params.query.stride(0) == s_q * h * d) && (params.query.stride(1) == d) &&
      (params.query.stride(2) == h * d) && (params.query.stride(3) == 1);
  const bool key_layout_ok = (params.key.stride(0) == s_k * h * d) && (params.key.stride(1) == d) &&
      (params.key.stride(2) == h * d) && (params.key.stride(3) == 1);
  const bool value_layout_ok = (params.value.stride(0) == s_v * h * d) && (params.value.stride(1) == d) &&
      (params.value.stride(2) == h * d) && (params.value.stride(3) == 1);

  const bool layout_ok = query_layout_ok && key_layout_ok && value_layout_ok;

  if (!packed_value_layout_ok && !layout_ok) {
    if (debug) {
      if (!packed_layout_ok) {
        if (!packed_query_layout_ok) {
          TORCH_WARN("Query tensor was not in cuDNN-supported packed QKV layout", params.query.strides());
        }
        if (!packed_key_layout_ok) {
          TORCH_WARN("Key tensor was not in cuDNN-supported packed QKV layout", params.key.strides());
        }
        if (!packed_value_layout_ok) {
          TORCH_WARN("Value tensor was not in cuDNN-supported packed QKV layout", params.value.strides());
        }
      }
      if (!layout_ok) {
        if (!query_layout_ok) {
          TORCH_WARN("Query tensor was not in cuDNN-supported unpacked QKV layout", params.query.strides());
        }
        if (!key_layout_ok) {
          TORCH_WARN("Key tensor was not in cuDNN-supported unpacked QKV layout", params.key.strides());
        }
        if (!value_layout_ok) {
          TORCH_WARN("Value tensor was not in cuDNN-supported unpacked QKV layout", params.value.strides());
        }
      }
    }
    return false;
  }
  return true;
}

bool check_cudnn_hardware_support(sdp_params const& params, bool debug) {
  return true;
}

bool check_for_nested_inputs(sdp_params const& params, bool debug) {
  static const bool enable_cudnn_nested = c10::utils::check_env("TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED") == true;
  if (has_for_nested_inputs(params) && !enable_cudnn_nested) {
    if (debug) {
      TORCH_WARN("Experimental cuDNN SDPA nested tensor support is not enabled.");
    }
    return false;
  }
  if (has_for_nested_inputs(params) &&
      (params.query.requires_grad() || params.key.requires_grad() || params.value.requires_grad())) {
    if (debug) {
      TORCH_WARN(
          "Experimental cuDNN SDPA nested tensor support does not support "
          "backward.");
      return false;
    }
  }
  return true;
}

bool check_dtypes_low_precision(sdp_params const& params, bool debug) {
  constexpr auto sm80_dtypes = c10::array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
  return check_tensor_dtype(params, sm80_dtypes, debug);
}

bool check_runtime_disabled_cudnn(sdp_params const& params, bool debug) {
  // We check the global context to see if user has explicitly turned of cudnn
  // sdp kernels
  if (!at::globalContext().userEnabledCuDNNSDP()) {
    if (debug) {
      TORCH_WARN("CuDNN attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

bool check_cudnn_deterministic(const sdp_params& params, bool debug) {
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (!ctx.deterministicAlgorithmsWarnOnly()) {
      if (debug) {
        TORCH_WARN("cuDNN SDPA is not deterministic.");
      }
      return false;
    }
  }
  return true;
}

} // namespace

bool can_use_cudnn_attention(const sdp_params& params, bool debug) {
  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_cudnn,
      check_for_nested_inputs,
      check_nonzero_sequence_lengths_dense,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_cudnn_tensor_shapes,
      check_cudnn_deterministic,
      check_dtypes_low_precision,
      check_attn_mask_shape,
      check_cudnn_hardware_support);
  for (const auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
#if TORCH_VER >= TORCH_2_8_0
  constexpr auto dense_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>,
      check_batch_size_and_num_heads_dense<true /*enable_gqa*/, false /*requires_same_num_heads*/>);

  if (has_only_dense_inputs(params)) {
    for (const auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
#endif
  return true;
}

bool is_flash_attention_available() {
#ifdef USE_FLASH_ATTENTION
  return true;
#else
  return false;
#endif
}

bool can_use_flash_attention(sdp_params const& params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  if (debug) {
    TORCH_WARN("Torch was not compiled with flash attention.");
  }
  return false;
#else // defined(USE_FLASH_ATTENTION)
  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_for_attn_mask,
      check_head_dim_size_flash<false /*caller_is_meff*/>,
      check_flash_attention_hardware_support,
      check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120,
      check_flash_causal_non_square_seqlens,
      check_dtypes_low_precision);
  for (const auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_nested,
        check_head_dim_size_flash_nested<false /*caller_is_meff*/>,
        check_for_seq_len_0_nested_tensor);
    for (const auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  constexpr bool backend_supports_grouped_query_attention = true;
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense<backend_supports_grouped_query_attention>,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>);
    for (const auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
#endif
}

bool can_use_mem_efficient_attention(sdp_params const& params, bool debug) {
  // Constraints specific to mem efficient attention
  constexpr auto less_than_sm80_mem_efficient_dtypes = c10::array_of<at::ScalarType>(at::kHalf, at::kFloat);
  constexpr auto greater_than_or_equal_sm80_mem_efficient_dtypes =
      c10::array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);

  //  Define gate functions that determine if a mem efficient kernel can be ran
  constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_mem_efficient,
      check_all_tensors_on_device,
      check_mem_efficient_hardware_support,
      check_tensor_shapes,
      check_head_dim_size_mem_efficient);
  for (const auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_requires_grad_and_nested, check_batch_size_nested, check_for_seq_len_0_nested_tensor);
    for (const auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim=*/>,
        check_batch_size_and_num_heads_dense<false /*supports_grouped_query_attention=*/>);
    for (const auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return check_tensor_dtype(params, greater_than_or_equal_sm80_mem_efficient_dtypes, debug);
}

SDPBackend select_sdp_backend(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() && !ctx.userEnabledMemEfficientSDP() &&
      !ctx.userEnabledCuDNNSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (const auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
          return SDPBackend::cudnn_attention;
        }
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Memory efficient kernel not used because:");
  sdp::can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  sdp::can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("SuDNN attention kernel not used because:");
  sdp::can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return SDPBackend::error;
}

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  auto* const nt_q_tensor_impl = at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN("Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }
  return true;
}

} // namespace sdp
