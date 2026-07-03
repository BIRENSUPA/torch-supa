/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <array>
#include <tuple>
#include <type_traits>

#include <torch_supa/csrc/core/supa/SUPAContext.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>
#include <torch_supa/csrc/utils/EnvConfig.h>
#include <torch_supa/csrc/utils/logger/Logger.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeCast.h>
#include "MemoryAccess.cuh"
#include "OffsetCalculator.cuh"

#include "torch_supa/csrc/aten/ops/Utils.h"

#define ASSERT_HOST_DEVICE_LAMBDA(type)

namespace at::native {

template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 0) {
      return 0;
    } else {
      return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
    }
}

template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else {
    return 8;
  }
}


//thread work size of 8 regresses the perf of elementwise kernel on cuda
//this doesn't change ROCm behavior as thread_work_size is already 4 on ROCm
constexpr int elementwise_thread_work_size() {return 4;}
constexpr int elementwise_block_work_size() {
  return elementwise_thread_work_size() * num_threads();
}

template <int io_sizes>
constexpr auto io_block_work_size() {
  return num_threads() * elems_per_thread<io_sizes>();
}

template <typename func_t>
constexpr auto calc_io_size(){
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  constexpr auto input_size = at::native::sum_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return input_size + output_size;
}


// To save on binary size of libtorch_cuda.so, we split the vectorized_elementwise_kernel
// into two: one for vec_size=8 and one for vec_size=[2, 4], since vec8 is going to be
// used on sm_90 and sm_100 exclusively.
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  if constexpr (vec_size == 8) {
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
    using traits = function_traits<func_t>;
    constexpr auto io_size = calc_io_size<func_t>();
    int remaining = N - io_block_work_size<io_size>() * blockIdx.x;

    if (remaining < io_block_work_size<io_size>()) { // if this block handles the reminder,
                  // just do a naive unrolled loop
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      auto policy = memory::policies::unroll<
      array_t,
      decltype(input_calc),
      decltype(output_calc),
      memory::LoadWithoutCast,
      memory::StoreWithoutCast,
      elems_per_thread<io_size>()>(
      data, remaining, input_calc, output_calc, loader, storer);
      elementwise_kernel_helper(f, policy);
    } else { // if this block has a full `block_work_size` data to handle, use
        // vectorized memory access
      elementwise_kernel_helper(
      f, memory::policies::vectorized<vec_size, array_t, elems_per_thread<io_size>()>(data));
    }
#endif // __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
  } else {
    using traits = function_traits<func_t>;
    constexpr auto io_size = calc_io_size<func_t>();
    int remaining = N - io_block_work_size<io_size>() * blockIdx.x;

    if (remaining < io_block_work_size<io_size>()) { // if this block handles the reminder,
                   // just do a naive unrolled loop
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      auto policy = memory::policies::unroll<
      array_t,
      decltype(input_calc),
      decltype(output_calc),
      memory::LoadWithoutCast,
      memory::StoreWithoutCast,
      elems_per_thread<io_size>()>(
      data, remaining, input_calc, output_calc, loader, storer);
      elementwise_kernel_helper(f, policy);
    } else { // if this block has a full `block_work_size` data to handle, use
         // vectorized memory access
      elementwise_kernel_helper(
      f, memory::policies::vectorized<vec_size, array_t, elems_per_thread<io_size>()>(data));
    }
  }
}



template <int vec_size, int tuned_elems_per_thread, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel_tuned(int N, func_t f, array_t data) {
  if constexpr (vec_size == 8) {
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
    using traits = function_traits<func_t>;
    constexpr int block_work_size = tuned_elems_per_thread * num_threads();
    int remaining = N - block_work_size * blockIdx.x;

    if (remaining < block_work_size) {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      auto policy = memory::policies::unroll<
          array_t,
          decltype(input_calc),
          decltype(output_calc),
          memory::LoadWithoutCast,
          memory::StoreWithoutCast,
          tuned_elems_per_thread>(
          data, remaining, input_calc, output_calc, loader, storer);
      elementwise_kernel_helper(f, policy);
    } else {
      elementwise_kernel_helper(
          f, memory::policies::vectorized<vec_size, array_t, tuned_elems_per_thread>(data));
    }
#endif
  } else {
    using traits = function_traits<func_t>;
    constexpr int block_work_size = tuned_elems_per_thread * num_threads();
    int remaining = N - block_work_size * blockIdx.x;

    if (remaining < block_work_size) {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      auto policy = memory::policies::unroll<
          array_t,
          decltype(input_calc),
          decltype(output_calc),
          memory::LoadWithoutCast,
          memory::StoreWithoutCast,
          tuned_elems_per_thread>(
          data, remaining, input_calc, output_calc, loader, storer);
      elementwise_kernel_helper(f, policy);
    } else {
      elementwise_kernel_helper(
          f, memory::policies::vectorized<vec_size, array_t, tuned_elems_per_thread>(data));
    }
  }
}

template <typename func_t, typename array_t>
static inline bool launch_vectorized_kernel_tuned(
    int64_t N,
    const func_t& f,
    array_t data,
    int tuned_elems_per_thread) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  TORCH_CHECK(
      tuned_elems_per_thread == 8 || tuned_elems_per_thread == 16 || tuned_elems_per_thread == 24,
      "tuned elements_per_thread must be one of {8, 16, 24}, but got ",
      tuned_elems_per_thread);

  auto stream = c10::supa::getCurrentSUPAStream();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  supaDeviceProp* p = at::supa::getDeviceProperties(stream.device_index());
  const int computeCapability = p->major * 10 + p->minor;
  if (computeCapability != 90 && computeCapability != 100) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }

  auto launch = [&](auto vec_tag, auto tws_tag) {
    constexpr int kVecSize = decltype(vec_tag)::value;
    constexpr int kTws = decltype(tws_tag)::value;
    constexpr int kBlockWorkSize = kTws * num_threads();
    int64_t grid = (N + kBlockWorkSize - 1) / kBlockWorkSize;
    vectorized_elementwise_kernel_tuned<kVecSize, kTws, func_t, array_t>
        <<<grid, num_threads(), 0, stream>>>(N, f, data);
    C10_SUPA_KERNEL_LAUNCH_CHECK();
  };

  switch (tuned_elems_per_thread) {
    case 8:
      switch (vec_size) {
        case 8:
          launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{});
          return true;
        case 4:
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 8>{});
          return true;
        case 2:
          launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{});
          return true;
        default:
          return false;
      }
    case 16:
      switch (vec_size) {
        case 8:
          launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 16>{});
          return true;
        case 4:
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 16>{});
          return true;
        case 2:
          launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 16>{});
          return true;
        default:
          return false;
      }
    case 24:
      switch (vec_size) {
        case 4:
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 24>{});
          return true;
        case 2:
          launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 24>{});
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}


template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;
  auto policy = memory::policies::
      unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t, elems_per_thread>(
          data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// ============================================================================
// Hybrid Vectorized Kernel for Stride-Matched Binary Operations
// ============================================================================

// Hybrid vectorized kernel: processes 32 elements per thread using
// vectorized load (vec_size=4) for stride-matched input and scalar load for other input
template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    int MatchedInputIdx,
    typename InputType0,  // Actual dtype of input 0
    typename InputType1,  // Actual dtype of input 1
    int elems_per_thread>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void hybrid_vectorized_elementwise_kernel(
    int N, func_t f, array_t data, inp_calc_t input_calc) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;

  if (remaining < elems_per_thread * num_threads()) {
    // Tail: fallback to unroll with scalar loads with static type conversion
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::detail::LoadWithStaticCast<InputType0, InputType1>();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t, inp_calc_t, decltype(output_calc),
        memory::detail::LoadWithStaticCast<InputType0, InputType1>, memory::StoreWithoutCast, elems_per_thread>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {
    // Main: hybrid vectorized with hybrid offset calculator
    auto policy = memory::policies::hybrid_vectorized<
        vec_size, array_t, elems_per_thread, inp_calc_t, MatchedInputIdx, InputType0, InputType1>(
        data, remaining, input_calc);
    elementwise_kernel_helper(f, policy);
  }
}

// Launch function for hybrid vectorized kernel
template <typename func_t, typename array_t, typename inp_calc_t, int MatchedInputIdx, typename InputType0, typename InputType1>
static inline void launch_hybrid_vectorized_kernel(
    int64_t N, const func_t& f, array_t data, inp_calc_t input_calc, int64_t io_size) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  constexpr int vec_size = 4;
  auto stream = c10::supa::getCurrentSUPAStream();
  if (io_size <= 96) {
    constexpr int elems_per_thread = 16;
    int bws = elems_per_thread * num_threads();
    int64_t grid = (N + bws - 1) / bws;
    hybrid_vectorized_elementwise_kernel<vec_size, func_t, array_t, inp_calc_t, MatchedInputIdx, InputType0, InputType1, elems_per_thread>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc);
  } else {
    constexpr int elems_per_thread = 8;
    int bws = elems_per_thread * num_threads();
    int64_t grid = (N + bws - 1) / bws;
    hybrid_vectorized_elementwise_kernel<vec_size, func_t, array_t, inp_calc_t, MatchedInputIdx, InputType0, InputType1, elems_per_thread>
        <<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc);
  }
  C10_SUPA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    int tuned_elems_per_thread) {
  if (tuned_elems_per_thread > 0 &&
      launch_vectorized_kernel_tuned(N, f, data, tuned_elems_per_thread)) {
    return;
  }

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = c10::supa::getCurrentSUPAStream();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  supaDeviceProp* p = at::supa::getDeviceProperties(stream.device_index());
  const int computeCapability = p->major * 10 + p->minor;
  if (computeCapability != 90 && computeCapability != 100) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }

  constexpr int tws = elems_per_thread<io_size>();
  int bws = tws * num_threads();
  int64_t grid = (N + bws - 1) / bws;
  switch (vec_size) {
    case 8:
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_SUPA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_SUPA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_SUPA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      int64_t grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
          <<<grid_unrolled, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_SUPA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  launch_vectorized_kernel(N, f, data, 0);
}

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int64_t grid = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
  auto stream = c10::supa::getCurrentSUPAStream();
  unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_SUPA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = c10::supa::getCurrentSUPAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_SUPA_KERNEL_LAUNCH_CHECK();
}


template <typename traits, size_t INDEX, typename index_t>
C10_HOST_DEVICE inline typename traits::template arg<INDEX>::type load_nocast_arg_ldcg(
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using arg_t = typename traits::template arg<INDEX>::type;
  auto* ptr = reinterpret_cast<const arg_t*>(data[INDEX] + i * strides[INDEX]);
  return memory::load_global_cached<arg_t>(ptr);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl_ldcg(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  return f(load_nocast_arg_ldcg<traits, INDEX>(data, strides, i)...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <
    bool UseLdcg,
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  if constexpr (UseLdcg) {
    return invoke_impl_ldcg<traits>(f, data, strides, i, Indices{});
  }
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <typename traits, typename func_t, typename index_t, typename... InputTypes, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl_static(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::convert<typename traits::template arg<I>::type>(
      c10::load<std::tuple_element_t<I, std::tuple<InputTypes...>>>(
          data[I] + i * strides[I]))...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

// Check if output is contiguous and an input tensor has the same element
// strides as the output tensor. This enables optimized hybrid vectorization.
template <int input_idx>
bool input_matches_contiguous_output(const TensorIteratorBase& iter) {
  if (iter.ndim() == 0) {
    return true;
  }
  auto out_strides = iter.strides(0);
  auto in_strides = iter.strides(input_idx + 1);  // +1 because output is at index 0
  auto out_elem_size = iter.element_size(0);
  auto in_elem_size = iter.element_size(input_idx + 1);
  int64_t expected_stride = 1;

  for (int i = 0; i < iter.ndim(); i++) {
    int64_t out_elem_stride = out_strides[i] / out_elem_size;
    if (out_elem_stride != expected_stride) {
      return false;
    }

    int64_t in_elem_stride = in_strides[i] / in_elem_size;
    if (out_elem_stride != in_elem_stride) {
      return false;
    }
    expected_stride *= iter.shape()[i];
  }
  return true;
}

template <typename traits, size_t... I>
constexpr bool legacy_path_all_inputs_support_ldcg_impl(
    std::index_sequence<I...>) {
  return ((memory::can_use_ldcg<typename traits::template arg<I>::type>::value &&
           !std::is_same_v<typename traits::template arg<I>::type, bool>) && ...);
}

template <typename traits>
constexpr bool legacy_path_all_inputs_support_ldcg() {
  return legacy_path_all_inputs_support_ldcg_impl<traits>(
      std::make_index_sequence<traits::arity>{});
}

template <int input_idx>
bool input_inner_dim_byte_stride_gt_256(const TensorIteratorBase& iter) {
  if (iter.ndim() == 0) {
    return false;
  }
  auto in_strides = iter.strides(input_idx + 1);
  // TensorIterator strides are consumed in iterator order; dimension 0 is the
  // innermost loop dimension paired with iter.shape().data() by OffsetCalculator.
  int64_t inner_dim_byte_stride = in_strides[0];
  return inner_dim_byte_stride > 256;
}

template <typename traits, size_t... I>
bool legacy_path_has_large_stride_input_impl(
    const TensorIteratorBase& iter,
    std::index_sequence<I...>) {
  return (input_inner_dim_byte_stride_gt_256<I>(iter) || ...);
}

template <typename traits>
bool legacy_path_use_ldcg(const TensorIteratorBase& iter) {
  if constexpr (!legacy_path_all_inputs_support_ldcg<traits>()) {
    return false;
  }
  return legacy_path_has_large_stride_input_impl<traits>(
      iter,
      std::make_index_sequence<traits::arity>{});
}

template <typename func_t, typename InputType1, typename InputType2, typename OutputType>
bool try_launch_static_contiguous_unrolled_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const std::array<char*, 3>& data,
    int64_t numel,
    const std::array<ScalarType, 3>& dtypes) {
  using traits = function_traits<func_t>;

  if constexpr (
      traits::arity == 2 &&
      std::is_same_v<OutputType, typename traits::result_type>) {
    if (dtypes[1] == CppTypeToScalarType<InputType1>::value &&
        dtypes[2] == CppTypeToScalarType<InputType2>::value &&
        dtypes[0] == CppTypeToScalarType<OutputType>::value) {
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      auto loader = memory::detail::LoadWithStaticCast<InputType1, InputType2>();
      auto storer = memory::StoreWithoutCast();
      launch_unrolled_kernel(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
      return true;
    }
  }
  return false;
}

template <typename func_t, typename InputType1, typename InputType2, typename OutputType>
bool try_launch_static_binary_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const std::array<char*, 3>& data,
    int64_t numel,
    const std::array<ScalarType, 3>& dtypes) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;

  if constexpr (traits::arity == 2) {
    if (dtypes[1] == CppTypeToScalarType<InputType1>::value &&
        dtypes[2] == CppTypeToScalarType<InputType2>::value &&
        dtypes[0] == CppTypeToScalarType<OutputType>::value) {

      using Indices = std::make_index_sequence<traits::arity>;
      int64_t ndim = iter.ndim();

      // Check which input has same strides as output for hybrid vectorization
      if (input_matches_contiguous_output<0>(iter)) {
        // Input0 matches output strides - use hybrid kernel with MatchedInputIdx=0
        // Create hybrid offset calculator: input0 uses trivial offset, input1 uses actual offset

        auto in0_strides = iter.strides(1);      // input0 strides
        auto in1_strides = iter.strides(2);      // input1 strides
        // in0_stride_1 is the shape at last dim.
        int64_t in0_stride_1 = in0_strides[ndim - 1] / iter.element_size(1);
        // in1_stride_0 is the stride of the non-contiguous input at last dim.
        int64_t in1_stride_0 = in1_strides[0] / iter.element_size(2);

        // If in1_stride_0 == 0, the last dim is broadcast and one warp roughly
        // loads max(1L, 32 / in0_stride_1) elements from input1; otherwise it
        // roughly loads min(in0_stride_1, 32L) elements.
        int64_t bct_load = in1_stride_0 == 0 ? std::max(1L, 32 / in0_stride_1) : std::min(in0_stride_1, 32L);
        int64_t io_size = 32 * iter.element_size(1) + bct_load * iter.element_size(2);

        auto hybrid_input_calc = make_input_hybrid_offset_calculator<traits::arity, 0>(iter);
        launch_hybrid_vectorized_kernel<func_t, std::array<char*, 3>, decltype(hybrid_input_calc), 0, InputType1, InputType2>(
            numel, f, data, hybrid_input_calc, io_size);
      } else if (input_matches_contiguous_output<1>(iter)) {
        // Input1 matches output strides - use hybrid kernel with MatchedInputIdx=1
        // Create hybrid offset calculator: input1 uses trivial offset, input0 uses actual offset

        auto in0_strides = iter.strides(1);      // input0 strides
        auto in1_strides = iter.strides(2);      // input1 strides
        int64_t in0_stride_0 = in0_strides[0] / iter.element_size(1);
        int64_t in1_stride_1 = in1_strides[ndim - 1] / iter.element_size(2);

        int64_t bct_load = in0_stride_0 == 0 ? std::max(1L, 32 / in1_stride_1) : std::min(in1_stride_1, 32L);
        int64_t io_size = 32 * iter.element_size(2) + bct_load * iter.element_size(1);

        auto hybrid_input_calc = make_input_hybrid_offset_calculator<traits::arity, 1>(iter);
        launch_hybrid_vectorized_kernel<func_t, std::array<char*, 3>, decltype(hybrid_input_calc), 1, InputType1, InputType2>(
            numel, f, data, hybrid_input_calc, io_size);
      } else {
        // No match - fallback to legacy kernel
        auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
          auto offsets = offset_calc.get(idx);
          arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
          *out = invoke_impl_static<traits, func_t, unsigned int, InputType1, InputType2>(
              f, &data[1], &offsets[1], 1, Indices{});
        });
      }
      return true;
    }
  }
  return false;
}



template <typename func_t>
void gpu_kernel_impl_nocast(
    TensorIteratorBase& iter,
    const func_t& f,
    int tuned_elems_per_thread) {

  TORCH_SUPA_DEBUG(
    "[gpu_kernel_nocast] tensor info={}",
    at::supa::format_tensor_iterator(iter));
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  std::array<char*, ntensors> data;
  std::array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
    dtypes[i] = iter.dtype(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data, tuned_elems_per_thread);
  }

  TORCH_SUPA_DEBUG(
      "[gpu_kernel_impl_nocast] entered using ldcg ? type_support={} large_stride={}",
      legacy_path_all_inputs_support_ldcg<traits>(),
      legacy_path_has_large_stride_input_impl<traits>(
          iter, std::make_index_sequence<traits::arity>{}));

  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  if constexpr (legacy_path_all_inputs_support_ldcg<traits>()) {
    bool use_ldcg = legacy_path_has_large_stride_input_impl<traits>(
        iter,
        std::make_index_sequence<traits::arity>{});
    if (use_ldcg) {
      launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke<true>(f, &data[1], &offsets[1], 1);
      });
      return;
    }
  }
  // For broadcast case, we could use hybrid vectorized launch to improve data loading.
  if constexpr (traits::arity == 2 && (std::is_same<arg0_t, c10::BFloat16>::value || std::is_same<arg0_t, float>::value)) {
    if (try_launch_static_binary_kernel<func_t, c10::BFloat16, bool, c10::BFloat16>(
            iter, f, data, numel, dtypes) ||
        try_launch_static_binary_kernel<func_t, float, float, float>(
            iter, f, data, numel, dtypes)) {
      return;
    } 
  }
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke<false>(f, &data[1], &offsets[1], 1);
  });
}

template <typename func_t> 
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f, int tuned_elems_per_thread) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f, tuned_elems_per_thread);
  }
  TORCH_SUPA_DEBUG(
    "[gpu_kernel] tensor info={}",
    at::supa::format_tensor_iterator(iter));
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  std::array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    dtypes[i] = iter.dtype(i);
  }

  bool contiguous = iter.is_contiguous();
  if (contiguous) {
    if constexpr (traits::arity == 2 && (std::is_same<arg0_t, c10::BFloat16>::value || std::is_same<arg0_t, float>::value)) {
      if (try_launch_static_contiguous_unrolled_kernel<func_t, c10::BFloat16, float, float>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_contiguous_unrolled_kernel<func_t, c10::BFloat16, float, c10::BFloat16>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_contiguous_unrolled_kernel<func_t, float, c10::BFloat16, float>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_contiguous_unrolled_kernel<func_t, c10::BFloat16, double, c10::BFloat16>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_contiguous_unrolled_kernel<func_t, float, double, float>(
              iter, f, data, numel, dtypes)) {
        return;
      }
    }
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
  } else {

    if constexpr (traits::arity == 2 && (std::is_same<arg0_t, c10::BFloat16>::value || std::is_same<arg0_t, float>::value)) {
      if (try_launch_static_binary_kernel<func_t, c10::BFloat16, float, float>(
              iter, f, data, numel, dtypes) || 
          try_launch_static_binary_kernel<func_t, c10::BFloat16, float, c10::BFloat16>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_binary_kernel<func_t, float, c10::BFloat16, float>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_binary_kernel<func_t, c10::BFloat16, double, c10::BFloat16>(
              iter, f, data, numel, dtypes) ||
          try_launch_static_binary_kernel<func_t, float, double, float>(
              iter, f, data, numel, dtypes)) {
        return;
      } 
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
  }
}

} // namespace at::native
