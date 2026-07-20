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

#include <limits>

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

void float32toint16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
        if (value < -32768.0f) {
            return static_cast<int16_t>(0);
        }
        return static_cast<int16_t>(value);
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
    TORCH_CHECK(false, "This supposed to be called only for Float8 types");
  }
}

namespace {

// Fast-path copy for a narrow set of 3D permute stride layouts (aten::copy_ /
// aten::contiguous). This is a targeted perf optimization, not a general permute.
//
// Entry gate (direct_copy_kernel_cuda):
//   - dtype is Half or BFloat16 (same on src/dst)
//   - src tensor is 3D
//   - analyze_permute_3d_transpose(src, dst) returns true; otherwise generic
//     elementwise_kernel handles the copy.
//
// Shared preconditions (detect_3d_permute_strides):
//   - src and dst are 3D with identical sizes
//   - dst is contiguous (row-major)
//   - src is non_overlapping_and_dense
//   - src strides match exactly one 3D permute layout perm[p]:
//       stride[i] = product of sizes on axes after p[i] in the original tensor
//   - perm is not identity (0,1,2)
//
// Kernel routing (analyze_permute_3d_transpose):
//   ContiguousTail -> permute_102_contig_tail_kernel when perm=(1,0,2) and
//     src.stride(2)==dst.stride(2)==1. Each (i,j) plane copies dim-2 as a
//     contiguous 1D slice with same logical coords: dst[i,j,k]=src[i,j,k].
//   Tile -> transpose_tile_big_kernel for all other non-identity perms, and
//     for perm=(1,0,2) when dim-2 is not stride-1. Decomposes one batch axis
//     and transposes the inner 2D slice with 64x64 tiles.
//
// Launch limits (can_launch_permute_3d_fastpath):
//   This fast path does not handle grid dimensions that overflow int or exceed
//   backend launch limits. When checks fail, direct_copy_kernel_cuda falls back
//   to generic elementwise_kernel (no error).
//   ContiguousTail: 2D grid (grid_x, grid_y, 1) with grid_x=min(plane_rows,
//     plane_cols) and grid_y=max(plane_rows, plane_cols) after axis swap.
//   Tile: 1D grid batch_count * ceil(tile_n/64) * ceil(tile_k/64).
enum class Permute3DTransposeKind {
  Tile,
  ContiguousTail,
};

struct Permute3DTransposeConfig {
  Permute3DTransposeKind kind{Permute3DTransposeKind::Tile};
  int batch_dim{-1};
  int dim_n{-1};  // inner dim mapped to kernel N (contiguous in src when possible)
  int dim_k{-1};  // inner dim mapped to kernel K (strided in src when possible)
  int tile_n{0};
  int tile_k{0};
  int batch_count{0};
  int plane_rows{0};
  int plane_cols{0};
  int tail_len{0};
  bool src_n_contiguous{false};
  uint64_t src_stride_k_bytes{0};
  uint64_t src_stride_n_bytes{0};
  uint64_t src_batch_stride_bytes{0};
  uint64_t src_stride_row_bytes{0};
  uint64_t src_stride_col_bytes{0};
  uint64_t dst_stride_row_bytes{0};
  uint64_t dst_stride_col_bytes{0};
  uint64_t dst_batch_stride_bytes{0};
};

// Detect whether src strides come from a single 3D permute of a dense tensor.
// On success, perm[out_dim] is the original tensor axis index at logical out_dim.
bool detect_3d_permute_strides(
    const Tensor& input,
    const Tensor& output,
    int perm[3]) {
  if (input.dim() != 3 || output.dim() != 3) {
    return false;
  }
  if (!input.sizes().equals(output.sizes())) {
    return false;
  }
  if (!output.is_contiguous()) {
    return false;
  }
  if (!input.is_non_overlapping_and_dense()) {
    return false;
  }

  const int64_t sizes[3] = {input.size(0), input.size(1), input.size(2)};

  for (int p0 = 0; p0 < 3; ++p0) {
    for (int p1 = 0; p1 < 3; ++p1) {
      if (p1 == p0) {
        continue;
      }
      for (int p2 = 0; p2 < 3; ++p2) {
        if (p2 == p0 || p2 == p1) {
          continue;
        }
        const int candidate_perm[3] = {p0, p1, p2};
        int64_t orig_sizes[3];
        orig_sizes[p0] = sizes[0];
        orig_sizes[p1] = sizes[1];
        orig_sizes[p2] = sizes[2];
        bool match = true;
        for (int i = 0; i < 3; ++i) {
          const int64_t expected_stride = candidate_perm[i] == 0
              ? orig_sizes[1] * orig_sizes[2]
              : (candidate_perm[i] == 1 ? orig_sizes[2] : 1);
          if (input.stride(i) != expected_stride) {
            match = false;
            break;
          }
        }
        if (!match) {
          continue;
        }
        for (int i = 0; i < 3; ++i) {
          perm[i] = candidate_perm[i];
        }
        return true;
      }
    }
  }
  return false;
}

// Classify a detected 3D permute layout and fill launch parameters.
// Returns false when detect fails or perm is identity.
bool analyze_permute_3d_transpose(
    const Tensor& input,
    const Tensor& output,
    Permute3DTransposeConfig& cfg) {
  int perm[3];
  if (!detect_3d_permute_strides(input, output, perm)) {
    return false;
  }
  if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
    return false;
  }

  // Route A: permute_102_contig_tail_kernel
  //   perm=(1,0,2), stride(2)==1 on both src and dst.
  // Example shapes from profiler cases:
  //   copy_:      src[2,4096,8192] stride [8192,16384,1] -> dst contiguous
  //   contiguous: src[4096,2,8192] stride [8192,33554432,1] -> dst contiguous
  if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2 &&
      input.stride(2) == 1 && output.stride(2) == 1) {
    const uint64_t elem_size = static_cast<uint64_t>(input.element_size());
    cfg.kind = Permute3DTransposeKind::ContiguousTail;
    cfg.plane_rows = static_cast<int>(output.size(0));
    cfg.plane_cols = static_cast<int>(output.size(1));
    cfg.tail_len = static_cast<int>(output.size(2));
    // Plane strides are byte strides along output/input tensor dims 0 and 1.
    cfg.src_stride_row_bytes = static_cast<uint64_t>(input.stride(0)) * elem_size;
    cfg.src_stride_col_bytes = static_cast<uint64_t>(input.stride(1)) * elem_size;
    cfg.dst_stride_row_bytes = static_cast<uint64_t>(output.stride(0)) * elem_size;
    cfg.dst_stride_col_bytes = static_cast<uint64_t>(output.stride(1)) * elem_size;
    return true;
  }

  // Route B: transpose_tile_big_kernel
  //   Any non-identity perm not handled above, including perm=(1,0,2) when
  //   stride(2)!=1. Pick a batch axis and transpose the other two axes as an
  //   N x K matrix per batch slice. Requires perm[v0] > perm[v1] for the two
  //   inner axes v0,v1; prefer a decomposition where src has a stride-1 axis.
  int best_batch = -1;
  int best_dim_n = -1;
  int best_dim_k = -1;
  bool best_src_n_contiguous = false;
  bool found = false;

  for (int batch_dim = 0; batch_dim < 3; ++batch_dim) {
    int inner_dims[2];
    int inner_count = 0;
    for (int d = 0; d < 3; ++d) {
      if (d != batch_dim) {
        inner_dims[inner_count++] = d;
      }
    }
    const int v0 = inner_dims[0];
    const int v1 = inner_dims[1];
    if (perm[v0] <= perm[v1]) {
      continue;
    }

    int dim_n = -1;
    int dim_k = -1;
    bool src_n_contiguous = false;
    if (input.stride(v0) == 1) {
      dim_n = v0;
      dim_k = v1;
      src_n_contiguous = true;
    } else if (input.stride(v1) == 1) {
      dim_n = v1;
      dim_k = v0;
      src_n_contiguous = true;
    } else {
      dim_n = v0;
      dim_k = v1;
      src_n_contiguous = false;
    }

    if (!found) {
      found = true;
    } else if (!best_src_n_contiguous && src_n_contiguous) {
      // Prefer a batch decomposition with a contiguous src axis for vectorized loads.
    } else {
      continue;
    }

    best_batch = batch_dim;
    best_dim_n = dim_n;
    best_dim_k = dim_k;
    best_src_n_contiguous = src_n_contiguous;
  }

  if (!found) {
    return false;
  }

  const uint64_t elem_size = static_cast<uint64_t>(input.element_size());
  cfg.kind = Permute3DTransposeKind::Tile;
  cfg.batch_dim = best_batch;
  cfg.dim_n = best_dim_n;
  cfg.dim_k = best_dim_k;
  cfg.tile_n = static_cast<int>(input.size(best_dim_n));
  cfg.tile_k = static_cast<int>(input.size(best_dim_k));
  cfg.batch_count = static_cast<int>(input.size(best_batch));
  cfg.src_n_contiguous = best_src_n_contiguous;
  cfg.src_stride_n_bytes = static_cast<uint64_t>(input.stride(best_dim_n)) * elem_size;
  cfg.src_stride_k_bytes = static_cast<uint64_t>(input.stride(best_dim_k)) * elem_size;
  cfg.src_batch_stride_bytes = static_cast<uint64_t>(input.stride(best_batch)) * elem_size;
  cfg.dst_stride_row_bytes = static_cast<uint64_t>(output.stride(best_dim_n)) * elem_size;
  cfg.dst_stride_col_bytes = static_cast<uint64_t>(output.stride(best_dim_k)) * elem_size;
  cfg.dst_batch_stride_bytes = static_cast<uint64_t>(output.stride(best_batch)) * elem_size;
  return true;
}

constexpr int64_t kPermute3DFastpathMaxGridDim =
    static_cast<int64_t>(std::numeric_limits<int>::max());

// Return false to fall back to elementwise_kernel when launch grid would overflow
// int32 grid dimensions. Profiler hot shapes are well inside these limits.
bool can_launch_permute_3d_fastpath(const Permute3DTransposeConfig& cfg) {
  if (cfg.kind == Permute3DTransposeKind::ContiguousTail) {
    const bool plane_j_on_x = cfg.plane_cols <= cfg.plane_rows;
    const int64_t grid_x = plane_j_on_x ? cfg.plane_cols : cfg.plane_rows;
    const int64_t grid_y = plane_j_on_x ? cfg.plane_rows : cfg.plane_cols;
    return grid_x > 0 && grid_y > 0 &&
        grid_x <= kPermute3DFastpathMaxGridDim &&
        grid_y <= kPermute3DFastpathMaxGridDim;
  }

  constexpr int64_t kBigTileSize = 64;
  const int64_t n_tiles =
      (static_cast<int64_t>(cfg.tile_n) + kBigTileSize - 1) / kBigTileSize;
  const int64_t k_tiles =
      (static_cast<int64_t>(cfg.tile_k) + kBigTileSize - 1) / kBigTileSize;
  const int64_t grid =
      static_cast<int64_t>(cfg.batch_count) * n_tiles * k_tiles;
  return grid > 0 && grid <= kPermute3DFastpathMaxGridDim;
}

} // namespace

// ContiguousTail fast path. One block per (plane_i, plane_j) pair; threads copy
// tail_len elements along dim 2 with burst4x2 vector loads when 16B-aligned.
// Misaligned plane bases skip burst and use scalar prologue/epilogue only.
// Entry: analyze_permute_3d_transpose selected Permute3DTransposeKind::ContiguousTail.
template<class _T>
__global__ void permute_102_contig_tail_kernel(
    const void* __restrict src,
    void* __restrict dst,
    const int plane_rows,
    const int plane_cols,
    const int tail_len,
    const uint64_t src_stride_row_bytes,
    const uint64_t src_stride_col_bytes,
    const uint64_t dst_stride_row_bytes,
    const uint64_t dst_stride_col_bytes,
    const bool plane_j_maps_to_block_x) {
  static_assert(sizeof(_T) == 2, "permute_102_contig_tail_kernel requires 2-byte element type");
  constexpr uint32_t element_size = sizeof(_T);
  constexpr uint32_t elements_in_16B = 16 / element_size;

  const int plane_i = plane_j_maps_to_block_x
      ? static_cast<int>(blockIdx.y)
      : static_cast<int>(blockIdx.x);
  const int plane_j = plane_j_maps_to_block_x
      ? static_cast<int>(blockIdx.x)
      : static_cast<int>(blockIdx.y);
  if (plane_i >= plane_rows || plane_j >= plane_cols) {
    return;
  }

  // plane_i/plane_j index logical tensor dims 0/1. Tail dim 2 is contiguous (stride 1).
  const uint8_t* src_base = static_cast<const uint8_t*>(src) +
      static_cast<uint64_t>(plane_i) * src_stride_row_bytes +
      static_cast<uint64_t>(plane_j) * src_stride_col_bytes;
  uint8_t* dst_base = static_cast<uint8_t*>(dst) +
      static_cast<uint64_t>(plane_i) * dst_stride_row_bytes +
      static_cast<uint64_t>(plane_j) * dst_stride_col_bytes;

  constexpr uint32_t burst_bytes = 16;
  constexpr uint32_t bursts_per_iter = 2;
  constexpr uint32_t iter_bytes = burst_bytes * bursts_per_iter;
  constexpr int elems_per_burst = static_cast<int>(elements_in_16B);
  constexpr int iter_elems = static_cast<int>(iter_bytes / element_size);

  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src_base);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_base);
  const uintptr_t addr_mod16 = src_addr % burst_bytes;
  const bool can_burst = addr_mod16 == (dst_addr % burst_bytes);
  const int align_elems =
      can_burst && addr_mod16 != 0
          ? static_cast<int>((burst_bytes - addr_mod16) / element_size)
          : 0;
  const int burst_start = align_elems;
  const int burst_end =
      can_burst
          ? burst_start +
                ((tail_len - burst_start) / iter_elems) * iter_elems
          : 0;

  const _T* src_elem = reinterpret_cast<const _T*>(src_base);
  _T* dst_elem = reinterpret_cast<_T*>(dst_base);

  for (int k = static_cast<int>(threadIdx.x) * 2; k < align_elems;
       k += static_cast<int>(blockDim.x) * 2) {
    #pragma unroll
    for (int p = 0; p < 2; p++) {
      const int elem_idx = k + p;
      if (elem_idx < align_elems) {
        dst_elem[elem_idx] = c10::load(src_elem + elem_idx);
      }
    }
  }

  for (int k = burst_start + static_cast<int>(threadIdx.x) * iter_elems;
       k < burst_end;
       k += static_cast<int>(blockDim.x) * iter_elems) {
    const auto burst0 = memory::ld_vec<burst_bytes>(src_elem + k);
    const auto burst1 = memory::ld_vec<burst_bytes>(src_elem + k + elems_per_burst);
    memory::st_vec<burst_bytes>(dst_elem + k, burst0);
    memory::st_vec<burst_bytes>(dst_elem + k + elems_per_burst, burst1);
  }

  for (int k = burst_end + static_cast<int>(threadIdx.x) * 2; k < tail_len;
       k += static_cast<int>(blockDim.x) * 2) {
    #pragma unroll
    for (int p = 0; p < 2; p++) {
      const int elem_idx = k + p;
      if (elem_idx < tail_len) {
        dst_elem[elem_idx] = c10::load(src_elem + elem_idx);
      }
    }
  }
}

// Tile fast path. One block per (batch, 64x64 tile) of an inner N x K transpose.
// Full 64x64 tiles use 16B vector loads when src N is contiguous and dst K is
// contiguous; partial tiles or non-contiguous src N use scalar element access.
// Entry: analyze_permute_3d_transpose selected Permute3DTransposeKind::Tile.
template<class _T, int _WG>
__global__ void transpose_tile_big_kernel(
    const void* __restrict a,
    void* __restrict c,
    const int N,
    const int K,
    const uint64_t src_stride_k_bytes,
    const uint64_t src_stride_n_bytes,
    const uint64_t src_batch_stride_bytes,
    const uint64_t dst_stride_row_bytes,
    const uint64_t dst_stride_col_bytes,
    const uint64_t dst_batch_stride_bytes,
    const bool src_n_contiguous)
{
  static_assert(sizeof(_T) == 2, "transpose_tile_big_kernel requires 2-byte element type");
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
  const uint64_t stride_n = src_stride_k_bytes;
  const uint64_t stride_k = dst_stride_row_bytes;
  const uint64_t stride_nk = src_batch_stride_bytes;
  const uint64_t stride_nk_dst = dst_batch_stride_bytes;
  const bool dst_k_contiguous = dst_stride_col_bytes == element_size;

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

  if (src_n_contiguous && dst_k_contiguous &&
      max_part_n == BIG_TILE_SIZE && max_part_k == BIG_TILE_SIZE) {
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

    uint8_t* pc = static_cast<uint8_t*>(c) + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk_dst;
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
      // Copy partial tiles with element accesses (also used when src N is not contiguous).
      constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
      constexpr uint32_t vmem_per_row = BIG_TILE_SIZE;
      constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint64_t src_tile_n = static_cast<uint64_t>(ti) * BIG_TILE_SIZE * src_stride_n_bytes;
      const uint64_t src_tile_k = static_cast<uint64_t>(tj) * BIG_TILE_SIZE * src_stride_k_bytes;
      const uint8_t* pat = (const uint8_t*)a + m * stride_nk + src_tile_n + src_tile_k;
      #pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
          uint32_t col = threadIdx.x % vmem_per_row;
          uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
          if (col < max_part_n && row < max_part_k)
          {
              uint64_t offset = row * src_stride_k_bytes + col * src_stride_n_bytes;
              const _T* pfa = reinterpret_cast<const _T*>(pat + offset);
              sa[row][col] = *pfa;
          }
      }
      __syncthreads();

      const uint64_t dst_tile_n = static_cast<uint64_t>(ti) * BIG_TILE_SIZE * dst_stride_row_bytes;
      const uint64_t dst_tile_k = static_cast<uint64_t>(tj) * BIG_TILE_SIZE * dst_stride_col_bytes;
      uint8_t* pc = static_cast<uint8_t*>(c) + m * stride_nk_dst + dst_tile_n + dst_tile_k;
      #pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
          uint32_t col = threadIdx.x % vmem_per_row;
          uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
          if (col < max_part_k && row < max_part_n)
          {
              uint64_t offset = row * dst_stride_row_bytes + col * dst_stride_col_bytes;
              _T* pfc = reinterpret_cast<_T*>(pc + offset);
              *pfc = sa[col][row];
          }
      }
  }
}

template<typename scalar_t>
void permute_3d_transpose_copy_impl(TensorIteratorBase &iter, const Permute3DTransposeConfig& cfg) {
  static_assert(sizeof(scalar_t) == 2, "permute_3d_transpose_copy only supports 2-byte dtypes");

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  auto stream = c10::supa::getCurrentSUPAStream();

  if (cfg.kind == Permute3DTransposeKind::ContiguousTail) {
    // 2D launch: grid is (grid_x, grid_y, 1), not plane_rows * plane_cols.
    // Keep the smaller plane axis on gridDim.x (large gridDim.x is unreliable on
    // some backends). Swap block-axis mapping when cols > rows.
    const bool plane_j_on_x = cfg.plane_cols <= cfg.plane_rows;
    const int grid_x = plane_j_on_x ? cfg.plane_cols : cfg.plane_rows;
    const int grid_y = plane_j_on_x ? cfg.plane_rows : cfg.plane_cols;
    const dim3 grid_dim(grid_x, grid_y, 1);
    const dim3 block_dim(256, 1, 1);
    permute_102_contig_tail_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        src,
        dst,
        cfg.plane_rows,
        cfg.plane_cols,
        cfg.tail_len,
        cfg.src_stride_row_bytes,
        cfg.src_stride_col_bytes,
        cfg.dst_stride_row_bytes,
        cfg.dst_stride_col_bytes,
        plane_j_on_x);
    C10_SUPA_KERNEL_LAUNCH_CHECK();
    return;
  }

  constexpr uint32_t BIG_TILE_SIZE = 64;
  const int n_tiles = (cfg.tile_n + static_cast<int>(BIG_TILE_SIZE) - 1) / static_cast<int>(BIG_TILE_SIZE);
  const int k_tiles = (cfg.tile_k + static_cast<int>(BIG_TILE_SIZE) - 1) / static_cast<int>(BIG_TILE_SIZE);
  const int big_tile_wg = cfg.batch_count * n_tiles * k_tiles;
  const dim3 grid_dim(big_tile_wg, 1, 1);
  const dim3 block_dim(256, 1, 1);
  transpose_tile_big_kernel<scalar_t, 256><<<grid_dim, block_dim, 0, stream>>>(
      src,
      dst,
      cfg.tile_n,
      cfg.tile_k,
      cfg.src_stride_k_bytes,
      cfg.src_stride_n_bytes,
      cfg.src_batch_stride_bytes,
      cfg.dst_stride_row_bytes,
      cfg.dst_stride_col_bytes,
      cfg.dst_batch_stride_bytes,
      cfg.src_n_contiguous);
  C10_SUPA_KERNEL_LAUNCH_CHECK();
}

void permute_3d_transpose_copy(TensorIteratorBase &iter, const Permute3DTransposeConfig& cfg) {
  const ScalarType dtype = iter.dtype(0);
  TORCH_CHECK(dtype == iter.dtype(1),
      "permute_3d_transpose_copy requires matching input and output dtypes");
  if (dtype == kBFloat16) {
    permute_3d_transpose_copy_impl<BFloat16>(iter, cfg);
  } else if (dtype == kHalf) {
    permute_3d_transpose_copy_impl<Half>(iter, cfg);
  } else {
    TORCH_CHECK(false, "permute_3d_transpose_copy expects Half or BFloat16, got ", dtype);
  }
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
  } else if (iter.dtype(1) == kFloat && dtype == kShort) {
    float32toint16_copy_kernel_cuda(iter);
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
  } else if ((dtype == kBFloat16 || dtype == kHalf) && iter.tensor(1).dim() == 3) {
    // 3D permute-to-contiguous fast path; see namespace comment above for entry
    // conditions and kernel routing (permute_102_contig_tail vs transpose_tile).
    Permute3DTransposeConfig cfg;
    if (analyze_permute_3d_transpose(iter.tensor(1), iter.tensor(0), cfg) &&
        can_launch_permute_3d_fastpath(cfg)) {
      permute_3d_transpose_copy(iter, cfg);
    } else {
      AT_DISPATCH_V2(
          dtype, "copy_", AT_WRAP([&] {
            gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
      }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
    }
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
