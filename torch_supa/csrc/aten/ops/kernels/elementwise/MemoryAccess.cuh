/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <c10/core/DynamicCast.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/detail/FunctionTraits.h>
#include "OffsetCalculator.cuh"
#include "thread_constants.h"

#include <thrust/tuple.h>
#include <supa_device.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at::native::memory {

namespace detail {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... /*args*/) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx, int block_work_size) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + block_work_size * idx;
    auto args_accessor = [&args] __device__ (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<typename data_t, typename offsets_t, typename ...Args>
  C10_HOST_DEVICE static void apply(
      const data_t& data,
      const offsets_t& offsets,
      thrust::tuple<Args...> ret) {
    using T = typename thrust::tuple_element<current, thrust::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = thrust::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return c10::load(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <typename T>
struct can_use_ldcg : std::false_type {};

template <>
struct can_use_ldcg<c10::BFloat16> : std::true_type {};

template <>
struct can_use_ldcg<float> : std::true_type {};

template <typename T>
__device__ inline T load_global_cached(const T* ptr) {
  return c10::load(ptr);
}

template <>
__device__ inline float load_global_cached<float>(const float* ptr) {
  union {
    __nv_bfloat162 raw;
    float value;
  } value = {__ldcg(reinterpret_cast<const __nv_bfloat162*>(ptr))};
  return value.value;
}

template <>
__device__ inline c10::BFloat16 load_global_cached<c10::BFloat16>(
    const c10::BFloat16* ptr) {
  auto raw = __ldcg(reinterpret_cast<const __nv_bfloat16*>(ptr));
  return c10::BFloat16(raw);
}

template <int N>
struct LoadWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  LoadWithCast(const TensorIteratorBase& iter) {
    CUDA_KERNEL_ASSERT(iter.ninputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i + iter.noutputs());
      element_sizes[i] = c10::elementSize(iter.dtype(i + iter.noutputs()));
    }
  }

  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

template <int N = 1>
struct StoreWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  StoreWithCast(const TensorIteratorBase& iter) {
    CUDA_KERNEL_ASSERT(iter.noutputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i);
      element_sizes[i] = c10::elementSize(iter.dtype(i));
    }
  }

  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    c10::cast_and_store<scalar_t>(dtypes[arg], ptr, value);
  }
};

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
__device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
#if defined(USE_ROCM) && defined(__gfx942__)
  using longx2 = __attribute__((__vector_size__(4*sizeof(int)))) int;
  if constexpr (sizeof(vec_t) == sizeof(int)) {
   union {
     vec_t v;
     int   i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const int *>(&(from[offset]))) };
   return tmpt.v;
  }
  else if constexpr (sizeof(vec_t) == sizeof(long)) {
   union {
     vec_t v;
     long   i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const long *>(&(from[offset]))) };
   return tmpt.v;
  }
  else if constexpr (sizeof(vec_t) == sizeof(longx2)) {
   union {
     vec_t v;
     longx2  i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const longx2 *>(&(from[offset]))) };
   return tmpt.v;
  }
#endif
  return from[offset];
}

template <int vec_size>
__device__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
  // See NOTE [Loading boolean values]
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

namespace policies {

template <
    int num_threads,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int elems_per_thread,
    int num_outputs = 1>
struct unroll_base {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  static constexpr int tws = elems_per_thread;
  static constexpr int block_work_size = elems_per_thread * num_threads;

  __device__ unroll_base(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((int)(threadIdx.x + thread_work_elem * num_threads) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx < remaining) {
        int linear_idx = thread_idx + block_work_size * idx;
        auto offset = input_offset_calculator.get(linear_idx);
        detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
            *this, args, offset, loader, i, num_outputs);
        thread_idx += num_threads;
      }
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx < remaining) {
        int linear_idx = thread_idx + block_work_size * idx;
        int offset = output_offset_calculator.get(linear_idx)[0];
        storer.store(from[i], data[0], offset);
        thread_idx += num_threads;
      }
    }
  }
};

// Utility type for all users of unroll that extract the num_threads value from
// the caller scope.
template <
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int elems_per_thread,
    int num_outputs = 1>
using unroll = unroll_base<
    num_threads(),
    data_t,
    inp_calc_t,
    out_calc_t,
    loader_t,
    storer_t,
    elems_per_thread,
    num_outputs>;

template <int vec_size, typename data_t, int elems_per_thread>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(elems_per_thread % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = elems_per_thread / vec_size;
  static constexpr int tws = elems_per_thread;

  data_t data;

  __device__ vectorized(data_t data) : data(data) {}

  __device__ inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      auto v = load_vector<vec_size>(from, index);
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, idx, elems_per_thread * num_threads());
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + elems_per_thread * num_threads() * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  //multi_outputs_unroll struct members and check_inbounds and load methods are copypasted from unroll struct
  //we don't use inheritance because of compiler bug in cuda 10.2+
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;
  static constexpr int tws = thread_work_size();

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
  data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((int)(threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads();
    }
  }


  template <typename return_t>
  __device__ inline void store(return_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= this->remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      memory::detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      thread_idx += num_threads();
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
// in order to compile
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(const char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<scalar_t, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<scalar_t, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<scalar_t, 8>>;
  if (address % vec8_alignment == 0) {
   return 8;
  } else
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  return can_vectorize_up_to<scalar_t>(static_cast<const char*>(pointer));
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits /*_*/) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}



template <typename T>
__inline__ size_t get_alignment(T ptr_or_size) {
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0) {
    return 16;
  } else if (val % 8 == 0) {
    return 8;
  } else if (val % 4 == 0) {
    return 4;
  } else if (val % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

template <>
__inline__ size_t get_alignment<size_t>(size_t size) {
  return get_alignment(reinterpret_cast<void*>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

template <int Size>
union Vec;

template <>
union Vec<4> {
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8> {
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16> {
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> ld_vec(const T* addr) {
  Vec<Alignment> vec;
  if constexpr (Alignment == 16) {
#if defined(USE_ROCM) || !defined(__MIRA__)
    vec.u128 = *reinterpret_cast<const uint4*>(addr);
  } else if constexpr (Alignment == 8) {
    vec.u64 = *reinterpret_cast<const uint64_t*>(addr);
  } else if constexpr (Alignment == 4) {
    vec.u32 = *reinterpret_cast<const uint32_t*>(addr);
#else
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1]), "=r"(vec.u32[2]), "=r"(vec.u32[3])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("ld.global.v2.u32 {%0,%1}, [%2];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("ld.global.u32 %0, [%1];" : "=r"(vec.u32) : "l"(addr) : "memory");
#endif
  } else {
    static_assert(dependent_false<T>);
  }
  return vec;
}

template <int Alignment, typename T>
__device__ __inline__ void st_vec(T* addr, const Vec<Alignment>& vec) {
  if constexpr (Alignment == 16) {
#if defined(USE_ROCM) || !defined(__MIRA__)
    reinterpret_cast<uint64_t*>(addr)[0] = vec.u64[0];
    reinterpret_cast<uint64_t*>(addr)[1] = vec.u64[1];
  } else if constexpr (Alignment == 8) {
    *reinterpret_cast<uint64_t*>(addr) = vec.u64;
  } else if constexpr (Alignment == 4) {
    *reinterpret_cast<uint32_t*>(addr) = vec.u32;
#else
    asm("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(addr),
          "r"(vec.u32[0]),
          "r"(vec.u32[1]),
          "r"(vec.u32[2]),
          "r"(vec.u32[3])
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("st.global.v2.u32 [%0], {%1,%2};"
        :
        : "l"(addr), "r"(vec.u32[0]), "r"(vec.u32[1])
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("st.global.u32 [%0], %1;" : : "l"(addr), "r"(vec.u32) : "memory");
#endif
  } else {
    static_assert(dependent_false<T>);
  }
}



} // namespace at::native::memory

namespace at::native {

// ============================================================================
// Hybrid Vectorized Policy for Stride-Matched Binary Operations
// ============================================================================

namespace memory::detail {

// Loader for static type conversion - used in tail case of hybrid kernel
// Compatible interface with LoadWithoutCast but supports different input types
template <typename InputType0, typename InputType1>
struct LoadWithStaticCast {
  template <typename scalar_t>
  __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg) {
    if (arg == 0) {
      auto val = c10::load(reinterpret_cast<const InputType0*>(base_ptr) + offset);
      if constexpr (std::is_same_v<InputType0, scalar_t>) {
        return val;
      } else {
        return static_cast<scalar_t>(val);
      }
    } else {
      auto val = c10::load(reinterpret_cast<const InputType1*>(base_ptr) + offset);
      if constexpr (std::is_same_v<InputType1, scalar_t>) {
        return val;
      } else {
        return static_cast<scalar_t>(val);
      }
    }
  }
};

// Helper for vectorized load of stride-matched input argument with dtype conversion
// input_t: actual data type in memory
// arg_t: target type (function argument type, may differ from input_t)
template<int arg_index, int vec_size, int loop_size, typename input_t>
struct hybrid_vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    auto ptr = reinterpret_cast<const input_t *>(self.data[arg_index + 1]) +
               policy_t::block_work_size * idx;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      auto v = load_vector<vec_size>(ptr, index);
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        // Convert from input_t to arg_t if needed
        if constexpr (std::is_same_v<input_t, arg_t>) {
          std::get<arg_index>(args[vec_size * i + j]) = v.val[j];
        } else {
          std::get<arg_index>(args[vec_size * i + j]) = static_cast<arg_t>(v.val[j]);
        }
      }
    }
  }
};

// Helper for scalar load of non-matched input argument with offset calculation and dtype conversion
// input_t: actual data type in memory
// arg_t: target type (function argument type, may differ from input_t)
// IMPORTANT: The index pattern must match hybrid_vectorized_load_helper for correct alignment
template<int arg_index, int vec_size, int loop_size, typename input_t>
struct hybrid_scalar_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int linear_index = (thread_idx + i * num_threads()) * vec_size + policy_t::block_work_size * idx;
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        auto offset = self.input_offset_calculator.get(linear_index + j);
        auto val = c10::load(reinterpret_cast<const input_t*>(self.data[arg_index + 1]) + offset[arg_index]);
        // Convert from input_t to arg_t if needed
        if constexpr (std::is_same_v<input_t, arg_t>) {
          std::get<arg_index>(args[vec_size * i + j]) = val;
        } else {
          std::get<arg_index>(args[vec_size * i + j]) = static_cast<arg_t>(val);
        }
      }
    }
  }
};

} // namespace memory::detail

namespace memory::policies {

// Hybrid vectorized policy for binary ops where one input matches output strides
// This allows processing 32 elements per thread (vec_size=4, unroll=8) instead of 4
template <
    int vec_size,
    typename data_t,
    int elems_per_thread,
    typename inp_calc_t,
    int MatchedInputIdx,
    typename InputType0,  // Actual dtype of input 0 (may differ from function arg type)
    typename InputType1>  // Actual dtype of input 1 (may differ from function arg type)
struct hybrid_vectorized {
  static_assert(elems_per_thread % vec_size == 0, "elems_per_thread must be a multiple of vec_size");
  static constexpr int loop_size = elems_per_thread / vec_size;  // 8 when elems_per_thread=32, vec_size=4
  static constexpr int tws = elems_per_thread;  // 32
  static constexpr int block_work_size = elems_per_thread * num_threads();

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;

  __device__ hybrid_vectorized(data_t data, int remaining, inp_calc_t calc)
      : data(data), remaining(remaining), input_offset_calculator(calc) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return true;
  }

  // Helper to get the actual input type for a given arg_index
  template<int arg_index>
  using actual_input_type_t = std::conditional_t<arg_index == 0, InputType0, InputType1>;

  // Load helper that dispatches based on whether this arg matches output strides
  template<int arg_index>
  struct load_helper {
    template <typename args_t, typename policy_t>
    static __device__ void apply(policy_t &self, args_t *args, int idx) {
      using input_t = actual_input_type_t<arg_index>;
      if constexpr (arg_index == MatchedInputIdx) {
        // Vectorized load for matched input with dtype conversion
        detail::hybrid_vectorized_load_helper<arg_index, vec_size, loop_size, input_t>::apply(self, args, idx);
      } else {
        // Scalar load for non-matched input with dtype conversion
        // Use same vec_size and loop_size to match the index pattern
        detail::hybrid_scalar_load_helper<arg_index, vec_size, loop_size, input_t>::apply(self, args, idx);
      }
    }
  };

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    // Use compile-time dispatch for each argument
    detail::static_unroll<load_helper, arity>::with_args(*this, args, idx);
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    // Vectorized store since output always matches stride
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      vec_t v;
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

} // namespace memory::policies
}