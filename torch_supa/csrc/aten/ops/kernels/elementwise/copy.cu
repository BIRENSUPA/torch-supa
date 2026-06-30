/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <torch_supa/csrc/core/supa/CachingHostAllocator.h>
#include <torch_supa/csrc/core/supa/SUPAContext.h>
#include <torch_supa/csrc/core/supa/SUPAEvent.h>
#include <torch_supa/csrc/core/supa/SUPAGuard.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <torch_supa/csrc/core/supa/TorchVersion.h>
#include <torch_supa/csrc/core/supa/PeerToPeerAccess.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <torch_supa/csrc/core/supa/SUPACachingAllocator.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>

// TODO(NS): Investigate why FP8 conversion intrinsics end up being slower
#ifdef AT_USE_NV_CVT_INTRINSICS
#include <cuda_fp8.h>
#endif

namespace at::native {

void neg_kernel_cuda(TensorIteratorBase &iter);
void conj_kernel_cuda(TensorIteratorBase &iter);
void neg_conj_kernel_cuda(TensorIteratorBase &iter);

void float16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
        return static_cast<at::Half>(value);
    });
}

void bfloat16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
        return static_cast<at::BFloat16>(value);
    });
}

void bfloat16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(at::BFloat16 value) {
        return static_cast<float>(value);
    });
}

void float16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(at::Half value) {
        return static_cast<float>(value);
    });
}

#if TORCH_VER >= TORCH_2_8_0
void float8_e8m0fnu_to_float_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float8_e8m0fnu value) {
        return static_cast<float>(value);
    });
}

void float8_e8m0fnu_to_float16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float8_e8m0fnu value) {
        return static_cast<Half>(static_cast<float>(value));
    });
}

void float8_e8m0fnu_to_bfloat16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float8_e8m0fnu value) {
        return static_cast<BFloat16>(static_cast<float>(value));
    });
}
#endif

void float8_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fn(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2 x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fnuz x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2fnuz x) { return x; });
         break;
    }
#if TORCH_VER >= TORCH_2_8_0
  } else if (dtype == kFloat8_e8m0fnu) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e8m0fnu(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e8m0fnu x) { return x; });
         break;
    }
#endif
  } else {
    TORCH_CHECK(false, "This supposed ot be called only for Float8 types");
  }
}

// This API is for detecting whether the permute parameter of a three-dimensional tensor
// in the Copy operation from src to dst is from [0, 1, 2] to [0, 2, 1].
bool is_permute_021(TensorIteratorBase &iter) {
  const auto& input = iter.tensor(1);
  const auto& output = iter.tensor(0);

  bool is_permute = false;
  if (input.dim() == 3) {
    is_permute = true;
    is_permute &= input.dim() == output.dim();
    is_permute &= input.stride(0) == input.size(1) * input.size(2);
    is_permute &= input.stride(1) == 1;
    is_permute &= input.stride(2) == input.size(1);
    is_permute &= output.is_contiguous();
  }
  return is_permute;
}

template<class _T, int _WG>
__global__ void transpose_tile_big_kernel(const void* __restrict a, void* __restrict c, const int N, const int K)
{
  constexpr uint32_t BIG_TILE_SIZE = 64;
  constexpr uint32_t max_swizzle = 0;
  // pad LDS row by dword
  constexpr uint32_t LDS_PAD = (4 / sizeof(_T));
  constexpr uint32_t element_size = sizeof(_T);  // in bytes
  constexpr uint32_t elements_in_16B = 16 / element_size;

  union BLOCK_16B
  {
      _T e[elements_in_16B];
      __uint128_t ow;
  };
  // Round up processing to next full tile
  const uint32_t n_tiles = (N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
  const uint32_t k_tiles = (K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
  const uint32_t nk_tiles = n_tiles * k_tiles;
  const uint32_t m = blockIdx.x / nk_tiles;
  const uint64_t stride_n = N * sizeof(_T);
  const uint64_t stride_k = K * sizeof(_T);
  const uint64_t stride_nk = N * K * sizeof(_T);

  // Walk destination tiles continuously for cache coherency
  constexpr uint32_t XCD = 8;
  constexpr uint32_t SEQ = 8;
  constexpr uint32_t sblk = XCD * SEQ;
  uint32_t tIdx = blockIdx.x % nk_tiles;
  tIdx = tIdx > max_swizzle ? tIdx :
      (tIdx / sblk) * sblk + (tIdx % sblk) / SEQ + (tIdx % SEQ) * XCD;
  uint32_t ti = tIdx / k_tiles;
  uint32_t tj = tIdx % k_tiles;

   __shared__ _T sa[BIG_TILE_SIZE][BIG_TILE_SIZE + LDS_PAD];

  // Detect partial tiles
  uint32_t max_part_n = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE) != 0) ? (N % BIG_TILE_SIZE) : BIG_TILE_SIZE;
  uint32_t max_part_k = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE) != 0) ? (K % BIG_TILE_SIZE) : BIG_TILE_SIZE;

  if (max_part_n == BIG_TILE_SIZE && max_part_k == BIG_TILE_SIZE) {
    // Copy full tile with large loads
    constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
    constexpr uint32_t vmem_per_row = row_bytes / sizeof(__uint128_t);
    constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
    constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
    // Make sure WG isn't too large
    static_assert(vmem_per_thread >= 1);

    const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
    #pragma unroll
    for (uint32_t t = 0; t < vmem_per_thread; t++)
    {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        uint64_t offset = row * stride_n + col * sizeof(__uint128_t);
        const __uint128_t* pfa = (const __uint128_t*)(pat + offset);
        BLOCK_16B d;
        d.ow = *pfa;
        #pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
            sa[row][col * elements_in_16B + i] = d.e[i];
        }
    }
    __syncthreads();

    const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
    #pragma unroll
    for (uint32_t t = 0; t < vmem_per_thread; t++)
    {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
        BLOCK_16B d;
        // Transpose tile on read from LDS
        #pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
            d.e[i] = sa[col * elements_in_16B + i][row];
        }
        __uint128_t* pfc = (__uint128_t*)(pc + offset);
        *pfc = d.ow;
    }
  } else {
      // Copy partial tiles with element accesses
      constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
      constexpr uint32_t vmem_per_row = BIG_TILE_SIZE;
      constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
      #pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
          uint32_t col = threadIdx.x % vmem_per_row;
          uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
          uint64_t offset = (col < max_part_n && row < max_part_k) ? row * stride_n + col * 2 : 0;
          const uint16_t* pfa = (const uint16_t*)(pat + offset);
          sa[row][col] = *pfa;
      }
      __syncthreads();

      const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
      #pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
          uint32_t col = threadIdx.x % vmem_per_row;
          uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
          if (col < max_part_k && row < max_part_n)
          {
              uint64_t offset = row * stride_k + col * 2;
              uint16_t* pfc = (uint16_t*)(pc + offset);
              *pfc = sa[col][row];
          }
      }
  }
}

void transpose_last2dim(TensorIteratorBase &iter) {
  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  const auto& input = iter.tensor(1);

  int M = input.size(0);
  int N = input.size(1);
  int K = input.size(2);

  auto stream = c10::supa::getCurrentSUPAStream();
  constexpr uint32_t BIG_TILE_SIZE = 64;
  int big_tile_wg = M * ((N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE) * ((K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE);
  const dim3 grid_dim(big_tile_wg, 1, 1);
  const dim3 block_dim(256, 1, 1);
  transpose_tile_big_kernel<uint16_t, 256><<<grid_dim, block_dim, 0, stream>>>(src, dst, N, K);
  C10_SUPA_KERNEL_LAUNCH_CHECK();
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
TORCH_SUPA_API void direct_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (isFloat8Type(dtype)) {
     float8_copy_kernel_cuda(iter);
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       bfloat16_copy_kernel_cuda(iter);
     } else {
       float16_copy_kernel_cuda(iter);
     }
  }
  else if ((iter.dtype(1) == kBFloat16 || iter.dtype(1) == kHalf) && dtype == kFloat) {
    if (iter.dtype(1) == kBFloat16) {
      bfloat16tofloat32_copy_kernel_cuda(iter);
    } else {
      float16tofloat32_copy_kernel_cuda(iter);
    }
  }
#if TORCH_VER >= TORCH_2_8_0
  else if (iter.dtype(1) == kFloat8_e8m0fnu && (dtype == kFloat || dtype == kBFloat16 || dtype == kHalf)) {
    if (dtype == kFloat) {
      float8_e8m0fnu_to_float_copy_kernel_cuda(iter);
    } else if (dtype == kBFloat16) {
      float8_e8m0fnu_to_bfloat16_copy_kernel_cuda(iter);
    } else {
      float8_e8m0fnu_to_float16_copy_kernel_cuda(iter);
    }
  }
#endif
  else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (is_permute_021(iter) && (dtype == kBFloat16 || dtype == kHalf)) {
      transpose_last2dim(iter);
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cuda", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

} // namespace at::native
