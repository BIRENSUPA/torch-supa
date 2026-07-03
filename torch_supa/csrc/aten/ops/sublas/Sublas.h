/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
/*
  Provides a subset of SUPA BLAS functions as templates:

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

    dot<Dtype>(n, x, incx, y, incy, result)

  where Dtype is float, at::Half or at::BFloat16.
  The functions are available in at::sublas namespace.
 */

#include <ATen/OpMathType.h>
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/core/supa/SublasContext.h"

namespace at::sublas {

namespace detail {
template <typename>
inline constexpr bool always_false_v = false;
} // namespace detail

// RAII guard that sets the SuBLAS pointer mode and restores it to
// its previous value when the guard is destroyed
class PointerModeGuard {
 public:
  PointerModeGuard(sublasHandle_t handle, sublasPointerMode_t mode) : handle(handle) {
    AT_SUBLAS_CHECK(sublasGetPointerMode(handle, &previous_mode));
    AT_SUBLAS_CHECK(sublasSetPointerMode(handle, mode));
  }
  PointerModeGuard(const PointerModeGuard&) = delete;
  PointerModeGuard& operator=(const PointerModeGuard&) = delete;
  PointerModeGuard(PointerModeGuard&&) = delete;
  PointerModeGuard& operator=(PointerModeGuard&&) = delete;

  ~PointerModeGuard() {
    sublasSetPointerMode(handle, previous_mode);
  }

 private:
  sublasHandle_t handle;
  sublasPointerMode_t previous_mode{};
};

/***
 *  Gemm
 */

#define SUBLAS_GEMM_ARGTYPES(Dtype) SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, Dtype)

#define SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)                                                   \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha, const Dtype *a, \
      int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta, C_Dtype *c, int64_t ldc

#define SUBLAS_GEMM_ARGS(Dtype) transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::sublas::gemm: not implemented");
}

// Currently Sublas only support float, half, bfloat16
template <>
void gemm<float>(SUBLAS_GEMM_ARGTYPES(float));
template <>
void gemm<at::Half>(SUBLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm<at::BFloat16>(SUBLAS_GEMM_ARGTYPES(at::BFloat16));
template <>
void gemm<at::Half, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template <>
void gemm<at::BFloat16, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::sublas::gemm_internal: not implemented");
}

template <>
void gemm_internal<float>(SUBLAS_GEMM_ARGTYPES(float));
template <>
void gemm_internal<at::Half>(SUBLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm_internal<at::BFloat16>(SUBLAS_GEMM_ARGTYPES(at::BFloat16));
template <>
void gemm_internal<at::Half, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template <>
void gemm_internal<at::BFloat16, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

/***
 *  Batch Gemm
 */

#define SUBLAS_BGEMM_ARGTYPES(Dtype) SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, Dtype)

#define SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)                                                      \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha, const Dtype *a,     \
      int64_t lda, int64_t stridea, const Dtype *b, int64_t ldb, int64_t strideb, at::opmath_type<Dtype> beta, \
      C_Dtype *c, int64_t ldc, int64_t stridec, int64_t num_batches

#define SUBLAS_BGEMM_ARGS(Dtype) \
  transa, transb, m, n, k, alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec, num_batches

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::sublas::bgemm: not implemented");
}

template <>
void bgemm<float>(SUBLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm<at::Half>(SUBLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm<at::BFloat16>(SUBLAS_BGEMM_ARGTYPES(at::BFloat16));
template <>
void bgemm<at::Half, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template <>
void bgemm<at::BFloat16, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::sublas::bgemm_internal: not implemented");
}

template <>
void bgemm_internal<float>(SUBLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm_internal<at::Half>(SUBLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm_internal<at::BFloat16>(SUBLAS_BGEMM_ARGTYPES(at::BFloat16));
template <>
void bgemm_internal<at::Half, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template <>
void bgemm_internal<at::BFloat16, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

enum GEMMAndBiasActivationEpilogue {
  None,
  RELU,
  GELU,
};

template <typename Dtype>
void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<Dtype> alpha_val,
    const Dtype* mat1_ptr,
    int64_t mat1_ld,
    const Dtype* mat2_ptr,
    int64_t mat2_ld,
    const Dtype* bias,
    Dtype* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation = GEMMAndBiasActivationEpilogue::None);

/* LEVEL 2 BLAS FUNCTIONS */

#define SUBLAS_GEMV_ARGTYPES(Dtype)                                                                         \
  char trans, int64_t m, int64_t n, Dtype alpha, const Dtype *a, int64_t lda, const Dtype *x, int64_t incx, \
      Dtype beta, Dtype *y, int64_t incy

template <typename Dtype>
inline void gemv(SUBLAS_GEMV_ARGTYPES(Dtype)) {
  static_assert(detail::always_false_v<Dtype>, "at::cuda::blas::gemv: not implemented");
}

template <>
void gemv<double>(SUBLAS_GEMV_ARGTYPES(double));
template <>
void gemv<float>(SUBLAS_GEMV_ARGTYPES(float));
template <>
void gemv<c10::complex<double>>(SUBLAS_GEMV_ARGTYPES(c10::complex<double>));
template <>
void gemv<c10::complex<float>>(SUBLAS_GEMV_ARGTYPES(c10::complex<float>));
template <>
void gemv<at::Half>(SUBLAS_GEMV_ARGTYPES(at::Half));
template <>
void gemv<at::BFloat16>(SUBLAS_GEMV_ARGTYPES(at::BFloat16));

/* LEVEL 1 BLAS FUNCTIONS */

#define SUBLAS_DOT_ARGTYPES(Dtype) \
  sublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, int incy, Dtype *result

template <typename Dtype>
inline void dot(SUBLAS_DOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::dot: not implemented for ", typeid(Dtype).name());
}

template <>
void dot<double>(SUBLAS_DOT_ARGTYPES(double));
template <>
void dot<float>(SUBLAS_DOT_ARGTYPES(float));
template <>
void dot<at::Half>(SUBLAS_DOT_ARGTYPES(at::Half));
template <>
void dot<at::BFloat16>(SUBLAS_DOT_ARGTYPES(at::BFloat16));
template <>
void dot<c10::complex<double>>(SUBLAS_DOT_ARGTYPES(c10::complex<double>));
template <>
void dot<c10::complex<float>>(SUBLAS_DOT_ARGTYPES(c10::complex<float>));

template <typename Dtype>
inline void vdot(SUBLAS_DOT_ARGTYPES(Dtype)) {
  static_assert(detail::always_false_v<Dtype>, "at::cuda::blas::vdot: not implemented");
}

template <>
void vdot<c10::complex<float>>(SUBLAS_DOT_ARGTYPES(c10::complex<float>));
template <>
void vdot<c10::complex<double>>(SUBLAS_DOT_ARGTYPES(c10::complex<double>));

} // namespace at::sublas