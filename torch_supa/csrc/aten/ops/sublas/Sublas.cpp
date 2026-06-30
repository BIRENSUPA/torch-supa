/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
  Provides the implementations of Sublas function templates.
 */

#include "torch_supa/csrc/aten/ops/sublas/Sublas.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/utils/EnvConfig.h"
#include "torch_supa/csrc/utils/Utils.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <cstring>

#define SUBLAS_POSINT_CHECK(FD, X)                                           \
  TORCH_CHECK(                                                               \
      (X > 0 && X <= INT_MAX),                                               \
      "at::sublas::" #FD " argument " #X " must be positive and less than ", \
      INT_MAX,                                                               \
      " but got ",                                                           \
      X)

#define SUBLAS_NONNEGINT_CHECK(FD, X)                                            \
  TORCH_CHECK(                                                                   \
      (X >= 0 && X <= INT_MAX),                                                  \
      "at::sublas::" #FD " argument " #X " must be non-negative and less than ", \
      INT_MAX,                                                                   \
      " but got ",                                                               \
      X)

namespace {

sublasOperation_t _sublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return SUBLAS_OP_N;
    case 't':
    case 'T':
      return SUBLAS_OP_T;
      // E01643: sublas2 does not support this type?
      // case 'c':
      // case 'C':
      //   return SUBLAS_OP_C;
  }
  TORCH_CHECK(false, "_sublasOpFromChar input should be 't', 'n' but got `", op, "`");
}

void _sublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1) {
    *lda = std::max<int64_t>(m, 1);
  }
}

void _sublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1) {
    *ldc = std::max<int64_t>(m, 1);
  }

  if (transa_) {
    if (m <= 1) {
      *lda = std::max<int64_t>(k, 1);
    }
  } else {
    if (k <= 1) {
      *lda = std::max<int64_t>(m, 1);
    }
  }

  if (transb_) {
    if (k <= 1) {
      *ldb = std::max<int64_t>(n, 1);
    }
  } else {
    if (n <= 1) {
      *ldb = std::max<int64_t>(k, 1);
    }
  }
}

uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}

size_t _parseChosenWorkspaceSize() {
  const auto val = torch_supa::utils::get_env("SUBLASLT_WORKSPACE_SIZE");
  size_t workspace_size = 1024; /* default size in KiB according to #73328 */

  if (val.has_value()) {
    try {
      workspace_size = std::stoi(val.value());
    } catch (std::invalid_argument const& e) {
      TORCH_WARN("invalid SUBLASLT_WORKSPACE_SIZE,", " using default workspace size of ", workspace_size, " KiB.");
    } catch (std::out_of_range const& e) {
      TORCH_WARN("SUBLASLT_WORKSPACE_SIZE out of range,", " using default workspace size of ", workspace_size, " KiB.");
    }
  }
  return workspace_size * 1024;
}

size_t _getWorkspaceSize() {
  static size_t workspace_size = _parseChosenWorkspaceSize();
  return workspace_size;
}

} // anonymous namespace

namespace at::sublas {

/* LEVEL 3 BLAS FUNCTIONS */

#define GEMM_CHECK_ARGVALUES(Dtype)         \
  do {                                      \
    SUBLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    SUBLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    SUBLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    SUBLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    SUBLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    SUBLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

#define BGEMM_CHECK_ARGVALUES(Dtype)                   \
  do {                                                 \
    SUBLAS_NONNEGINT_CHECK(bgemm<Dtype>, m);           \
    SUBLAS_NONNEGINT_CHECK(bgemm<Dtype>, n);           \
    SUBLAS_NONNEGINT_CHECK(bgemm<Dtype>, k);           \
    SUBLAS_POSINT_CHECK(bgemm<Dtype>, lda);            \
    SUBLAS_POSINT_CHECK(bgemm<Dtype>, ldb);            \
    SUBLAS_POSINT_CHECK(bgemm<Dtype>, ldc);            \
    SUBLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches); \
  } while (0)

namespace {
// Following the pattern of CuSparseDescriptor
// Defined here for now because this is the only place sublas_lt interface is
// used but can be moved to a header once sublas_lt interface is used in
// multiple places.
template <typename T, sublasStatus_t (*destructor)(T*)>
struct SuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      AT_SUBLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, sublasStatus_t (*destructor)(T*)>
class SuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, SuBlasLtDeleter<T, destructor>> descriptor_;
};

class SuBlasLtMatmulDescriptor : public SuBlasLtDescriptor<sublasLtMatmulDescOpaque_t, &sublasLtMatmulDescDestroy> {
 public:
  SuBlasLtMatmulDescriptor(sublasComputeType_t compute_type, supaDataType_t scale_type) {
    sublasLtMatmulDesc_t raw_descriptor = nullptr;
    AT_SUBLAS_CHECK(sublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(sublasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    AT_SUBLAS_CHECK(::sublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class SuBlasLtMatrixLayout : public SuBlasLtDescriptor<sublasLtMatrixLayoutOpaque_t, &sublasLtMatrixLayoutDestroy> {
 public:
  SuBlasLtMatrixLayout(supaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld, bool t = false) {
    sublasLtMatrixLayout_t raw_descriptor = nullptr;
    AT_SUBLAS_CHECK(sublasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(sublasLtMatrixLayoutAttribute_t attr, const T value) {
    AT_SUBLAS_CHECK(::sublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class SuBlasLtMatmulPreference
    : public SuBlasLtDescriptor<sublasLtMatmulPreferenceOpaque_t, &sublasLtMatmulPreferenceDestroy> {
 public:
  SuBlasLtMatmulPreference() {
    sublasLtMatmulPreference_t raw_descriptor = nullptr;
    AT_SUBLAS_CHECK(sublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(sublasLtMatmulPreferenceAttributes_t attr, const T value) {
    AT_SUBLAS_CHECK(::sublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};
} // namespace

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal_sublaslt(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  supaDataType_t abType = SUPA_R_32F;
  supaDataType_t cType = SUPA_R_32F;
  sublasComputeType_t computeType = SUBLAS_COMPUTE_32F;
  supaDataType_t scaleType = SUPA_R_32F;
  if constexpr (std::is_same_v<Dtype, float>) {
    if (!at::NoTF32Guard::should_disable_tf32() &&
#if TORCH_VER >= TORCH_2_10_0
        at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
            at::Float32Precision::TF32
#elif TORCH_VER >= TORCH_2_9_0
        at::globalContext().float32Precision("cuda", "matmul") == "tf32"
#else
        at::globalContext().allowTF32CuBLAS()
#endif
    ) {
      computeType = SUBLAS_COMPUTE_32F_FAST_TF32;
    }
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abType = SUPA_R_16F;
    cType = (std::is_same_v<C_Dtype, float>) ? SUPA_R_32F : SUPA_R_16F;
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abType = SUPA_R_16BF;
    cType = (std::is_same_v<C_Dtype, float>) ? SUPA_R_32F : SUPA_R_16BF;
  } else {
    static_assert(detail::always_false_v<Dtype>, "at::sublas::bgemm_internal_sublaslt: not implemented");
  }

  sublasLtHandle_t ltHandle = at::supa::getCurrentSuBlasLtHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  SuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_TRANSB, opb);
  SuBlasLtMatrixLayout Adesc(abType, m, k, lda, opa == SUBLAS_OP_T);
  SuBlasLtMatrixLayout Bdesc(abType, k, n, ldb, opb == SUBLAS_OP_T);
  SuBlasLtMatrixLayout Cdesc(cType, m, n, ldc);

  if (num_batches > 1) {
    int num_batches_as_int = static_cast<int>(num_batches);
    Adesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Bdesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Cdesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Adesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridea);
    Bdesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
    Cdesc.setAttribute(SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
  }

  SuBlasLtMatmulPreference preference;
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(c));
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);

  auto workspace =
      at::empty(static_cast<int64_t>(workspaceSize), at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));

  sublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  AT_SUBLAS_CHECK(sublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    AT_SUBLAS_CHECK(SUBLAS_STATUS_NOT_SUPPORTED);
  }

  sublasStatus_t sublasStatus = sublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha,
      a,
      Adesc.descriptor(),
      b,
      Bdesc.descriptor(),
      &beta,
      c,
      Cdesc.descriptor(),
      c,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_data_ptr(),
      workspaceSize,
      c10::supa::getCurrentSUPAStream());
  TORCH_CHECK(
      sublasStatus == SUBLAS_STATUS_SUCCESS,
      "SUPA error: ",
      at::sublas::_sublasGetErrorEnum(sublasStatus),
      " when calling sublasLtMatmul with transpose_mat1 ",
      (opa == SUBLAS_OP_T),
      " transpose_mat2 ",
      (opb == SUBLAS_OP_T),
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " lda ",
      lda,
      " ldb ",
      ldb,
      " ldc ",
      ldc,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_sublaslt(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  // forward to bgemm implementation but set strides and batches to 0
  bgemm_internal_sublaslt(transa, transb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 0);
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal_sublas(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(detail::always_false_v<Dtype>, "at::sublas::bgemm_internal_sublas: not implemented");
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_sublas(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(detail::always_false_v<Dtype>, "at::sublas::gemm_internal_sublas: not implemented");
}

template <>
void bgemm_internal_sublas<float>(SUBLAS_BGEMM_ARGTYPES(float)) {
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  AT_SUBLAS_CHECK(sublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <typename C_Dtype>
inline void bgemm_internal_sublas_half_helper(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;

  AT_SUBLAS_CHECK(sublasGemmStridedBatchedEx(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      (void*)(&falpha),
      a,
      SUPA_R_16F,
      lda,
      stridea,
      b,
      SUPA_R_16F,
      ldb,
      strideb,
      (void*)(&fbeta),
      c,
      std::is_same_v<C_Dtype, float> ? SUPA_R_32F : SUPA_R_16F,
      ldc,
      stridec,
      num_batches,
      SUBLAS_COMPUTE_32F,
      SUBLAS_GEMM_DEFAULT));
}

template <>
void bgemm_internal_sublas<at::Half>(SUBLAS_BGEMM_ARGTYPES(at::Half)) {
  bgemm_internal_sublas_half_helper<at::Half>(SUBLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm_internal_sublas<at::Half, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  bgemm_internal_sublas_half_helper<float>(SUBLAS_BGEMM_ARGS(at::Half));
}

template <typename C_Dtype>
inline void bgemm_internal_sublas_bfloat16_helper(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  auto compute_type = SUBLAS_COMPUTE_32F;
  AT_SUBLAS_CHECK(sublasGemmStridedBatchedEx(
      handle,
      opa,
      opb,
      (int)m,
      (int)n,
      (int)k,
      (void*)&falpha,
      a,
      SUPA_R_16BF,
      (int)lda,
      stridea,
      b,
      SUPA_R_16BF,
      (int)ldb,
      strideb,
      (void*)&fbeta,
      c,
      std::is_same_v<C_Dtype, float> ? SUPA_R_32F : SUPA_R_16BF,
      (int)ldc,
      stridec,
      (int)num_batches,
      compute_type,
      SUBLAS_GEMM_DEFAULT));
}

template <>
void bgemm_internal_sublas<at::BFloat16>(SUBLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  bgemm_internal_sublas_bfloat16_helper<at::BFloat16>(SUBLAS_BGEMM_ARGS(at::BFloat16));
}

template <>
void bgemm_internal_sublas<at::BFloat16, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  bgemm_internal_sublas_bfloat16_helper<float>(SUBLAS_BGEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal_sublas<float>(SUBLAS_GEMM_ARGTYPES(float)) {
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  AT_SUBLAS_CHECK(sublasSgemm(handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <typename C_Dtype>
inline void gemm_internal_sublas_half_helper(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);

  supaDeviceProp* prop = at::supa::getCurrentDeviceProperties();
  // E01643(TODO): should change based on br200 device properties
  if (prop->major >= 5) {
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    sublasMath_t sublas_flags = SUBLAS_DEFAULT_MATH;
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();

#if TORCH_VER >= TORCH_2_10_0
    TORCH_CHECK(
        fp16_reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
        "..., allow_splitk=False) requires the suBLASLt backend");
#endif

    if (
#if TORCH_VER >= TORCH_2_10_0
        fp16_reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK
#else
        !fp16_reduction
#endif
    ) {
      sublas_flags = static_cast<sublasMath_t>(sublas_flags | SUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
    auto compute_type = SUBLAS_COMPUTE_32F;
    AT_SUBLAS_CHECK(sublasSetMathMode(handle, sublas_flags));
    AT_SUBLAS_CHECK(sublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        SUPA_R_16F,
        lda,
        b,
        SUPA_R_16F,
        ldb,
        &fbeta,
        c,
        std::is_same_v<C_Dtype, float> ? SUPA_R_32F : SUPA_R_16F,
        ldc,
        compute_type,
        SUBLAS_GEMM_DEFAULT));
    AT_SUBLAS_CHECK(sublasSetMathMode(handle, SUBLAS_DEFAULT_MATH));
  } else {
    AT_SUBLAS_CHECK(sublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        SUPA_R_16F,
        lda,
        b,
        SUPA_R_16F,
        ldb,
        &fbeta,
        c,
        std::is_same_v<C_Dtype, float> ? SUPA_R_32F : SUPA_R_16F,
        ldc));
  }
}

template <>
void gemm_internal_sublas<at::Half>(SUBLAS_GEMM_ARGTYPES(at::Half)) {
  gemm_internal_sublas_half_helper<at::Half>(SUBLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm_internal_sublas<at::Half, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  gemm_internal_sublas_half_helper<float>(SUBLAS_GEMM_ARGS(at::Half));
}

template <typename C_Dtype>
inline void gemm_internal_sublas_bfloat16_helper(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t opa = _sublasOpFromChar(transa);
  sublasOperation_t opb = _sublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _sublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::BFloat16);
  sublasMath_t sublas_flags = SUBLAS_DEFAULT_MATH;
  auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();

#if TORCH_VER >= TORCH_2_10_0
  TORCH_CHECK(
      bf16_reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
      "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
      "..., allow_splitk=False) requires the cuBLASLt backend");
#endif

  if (
#if TORCH_VER >= TORCH_2_10_0
      bf16_reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK
#else
      !bf16_reduction
#endif
  ) {
    sublas_flags = static_cast<sublasMath_t>(sublas_flags | SUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
  auto compute_type = SUBLAS_COMPUTE_32F;
  AT_SUBLAS_CHECK(sublasSetMathMode(handle, sublas_flags));
  AT_SUBLAS_CHECK(sublasGemmEx(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      SUPA_R_16BF,
      lda,
      b,
      SUPA_R_16BF,
      ldb,
      &fbeta,
      c,
      std::is_same_v<C_Dtype, float> ? SUPA_R_32F : SUPA_R_16BF,
      ldc,
      compute_type,
      SUBLAS_GEMM_DEFAULT));
  AT_SUBLAS_CHECK(sublasSetMathMode(handle, SUBLAS_DEFAULT_MATH));
}

template <>
void gemm_internal_sublas<at::BFloat16>(SUBLAS_GEMM_ARGTYPES(at::BFloat16)) {
  gemm_internal_sublas_bfloat16_helper<at::BFloat16>(SUBLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal_sublas<at::BFloat16, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  gemm_internal_sublas_bfloat16_helper<float>(SUBLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm<float>(SUBLAS_GEMM_ARGTYPES(float)) {
  gemm_internal<float>(SUBLAS_GEMM_ARGS(float));
}

template <>
void gemm<at::Half>(SUBLAS_GEMM_ARGTYPES(at::Half)) {
  gemm_internal<at::Half>(SUBLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm<at::BFloat16>(SUBLAS_GEMM_ARGTYPES(at::BFloat16)) {
  gemm_internal<at::BFloat16>(SUBLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm<at::Half, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  gemm_internal<at::Half, float>(SUBLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm<at::BFloat16, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  gemm_internal<at::BFloat16, float>(SUBLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal<float>(SUBLAS_GEMM_ARGTYPES(float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    gemm_internal_sublaslt<float>(SUBLAS_GEMM_ARGS(float));
  } else {
    gemm_internal_sublas<float>(SUBLAS_GEMM_ARGS(float));
  }
}

template <>
void gemm_internal<at::Half>(SUBLAS_GEMM_ARGTYPES(at::Half)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    gemm_internal_sublaslt<at::Half>(SUBLAS_GEMM_ARGS(at::Half));
  } else {
    gemm_internal_sublas<at::Half>(SUBLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::BFloat16>(SUBLAS_GEMM_ARGTYPES(at::BFloat16)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    gemm_internal_sublaslt<at::BFloat16>(SUBLAS_GEMM_ARGS(at::BFloat16));
  } else {
    gemm_internal_sublas<at::BFloat16>(SUBLAS_GEMM_ARGS(at::BFloat16));
  }
}

template <>
void gemm_internal<at::Half, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    gemm_internal_sublaslt<at::Half, float>(SUBLAS_GEMM_ARGS(at::Half));
  } else {
    gemm_internal_sublas<at::Half, float>(SUBLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::BFloat16, float>(SUBLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    gemm_internal_sublaslt<at::BFloat16, float>(SUBLAS_GEMM_ARGS(at::BFloat16));
  } else {
    gemm_internal_sublas<at::BFloat16, float>(SUBLAS_GEMM_ARGS(at::BFloat16));
  }
}

template <>
void bgemm<float>(SUBLAS_BGEMM_ARGTYPES(float)) {
  bgemm_internal<float>(SUBLAS_BGEMM_ARGS(float));
}

template <>
void bgemm<at::Half>(SUBLAS_BGEMM_ARGTYPES(at::Half)) {
  bgemm_internal<at::Half>(SUBLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm<at::BFloat16>(SUBLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  bgemm_internal<at::BFloat16>(SUBLAS_BGEMM_ARGS(at::BFloat16));
}

template <>
void bgemm<at::Half, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  bgemm_internal<at::Half, float>(SUBLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm<at::BFloat16, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  bgemm_internal<at::BFloat16, float>(SUBLAS_BGEMM_ARGS(at::BFloat16));
}

template <>
void bgemm_internal<float>(SUBLAS_BGEMM_ARGTYPES(float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    bgemm_internal_sublaslt<float>(SUBLAS_BGEMM_ARGS(float));
  } else {
    bgemm_internal_sublas<float>(SUBLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm_internal<at::Half>(SUBLAS_BGEMM_ARGTYPES(at::Half)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    bgemm_internal_sublaslt<at::Half>(SUBLAS_BGEMM_ARGS(at::Half));
  } else {
    bgemm_internal_sublas<at::Half>(SUBLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::BFloat16>(SUBLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    bgemm_internal_sublaslt<at::BFloat16>(SUBLAS_BGEMM_ARGS(at::BFloat16));
  } else {
    bgemm_internal_sublas<at::BFloat16>(SUBLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <>
void bgemm_internal<at::Half, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    bgemm_internal_sublaslt<at::Half, float>(SUBLAS_BGEMM_ARGS(at::Half));
  } else {
    bgemm_internal_sublas<at::Half, float>(SUBLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::BFloat16, float>(SUBLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  if (strcmp(torch_supa::utils::EnvConfig::GetSublasPreferredBackend(), "Sublaslt") == 0) {
    bgemm_internal_sublaslt<at::BFloat16, float>(SUBLAS_BGEMM_ARGS(at::BFloat16));
  } else {
    bgemm_internal_sublas<at::BFloat16, float>(SUBLAS_BGEMM_ARGS(at::BFloat16));
  }
}

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
    GEMMAndBiasActivationEpilogue activation) {
  using opmath_t = at::opmath_type<Dtype>;
  opmath_t beta_val = 0; // bias is added in epilogue

  supaDataType_t abType = SUPA_R_32F;
  sublasComputeType_t computeType = SUBLAS_COMPUTE_32F;
  supaDataType_t scaleType = SUPA_R_32F;
  if constexpr (std::is_same_v<Dtype, double>) {
    TORCH_CHECK(false, "at::sublas::gemm_and_bias does not support double dtype");
  } else if constexpr (std::is_same_v<Dtype, float>) {
    if (!at::NoTF32Guard::should_disable_tf32() &&
#if TORCH_VER >= TORCH_2_10_0
        at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
            at::Float32Precision::TF32
#elif TORCH_VER >= TORCH_2_9_0
        at::globalContext().float32Precision("cuda", "matmul") == "tf32"
#else
        at::globalContext().allowTF32CuBLAS()
#endif
    ) {
      computeType = SUBLAS_COMPUTE_32F_FAST_TF32;
    }
    abType = SUPA_R_32F;
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abType = SUPA_R_16F;
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abType = SUPA_R_16BF;
  }

  SuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  sublasOperation_t transa = transpose_mat1 ? SUBLAS_OP_T : SUBLAS_OP_N;
  computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_TRANSA, transa);
  sublasOperation_t transb = transpose_mat2 ? SUBLAS_OP_T : SUBLAS_OP_N;
  computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_TRANSB, transb);
  sublasLtEpilogue_t epilogue = SUBLASLT_EPILOGUE_BIAS;
  if (activation == GEMMAndBiasActivationEpilogue::RELU) {
    epilogue = SUBLASLT_EPILOGUE_RELU_BIAS;
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
    epilogue = SUBLASLT_EPILOGUE_GELU_BIAS;
  }

  if (bias != nullptr) {
    computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    computeDesc.setAttribute(SUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
  }

  SuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  SuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  SuBlasLtMatrixLayout Cdesc(abType, m, n, result_ld);

  SuBlasLtMatmulPreference preference;
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
  uint32_t d_alignment = _getAlignment(reinterpret_cast<uintptr_t>(bias));
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
  preference.setAttribute(SUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d_alignment);

  auto workspace =
      at::empty(static_cast<int64_t>(workspaceSize), at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));

  sublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  sublasLtHandle_t ltHandle = at::supa::getCurrentSuBlasLtHandle();
  AT_SUBLAS_CHECK(sublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    AT_SUBLAS_CHECK(SUBLAS_STATUS_NOT_SUPPORTED);
  }

  sublasStatus_t sublasStatus = sublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_data_ptr(),
      workspaceSize,
      c10::supa::getCurrentSUPAStream());
  TORCH_CHECK(
      sublasStatus == SUBLAS_STATUS_SUCCESS,
      "SUPA error: ",
      at::sublas::_sublasGetErrorEnum(sublasStatus),
      " when calling sublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<double> alpha_val,
    const double* mat1_ptr,
    int64_t mat1_ld,
    const double* mat2_ptr,
    int64_t mat2_ld,
    const double* bias,
    double* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<float> alpha_val,
    const float* mat1_ptr,
    int64_t mat1_ld,
    const float* mat2_ptr,
    int64_t mat2_ld,
    const float* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::Half> alpha_val,
    const at::Half* mat1_ptr,
    int64_t mat1_ld,
    const at::Half* mat2_ptr,
    int64_t mat2_ld,
    const at::Half* bias,
    at::Half* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::BFloat16> alpha_val,
    const at::BFloat16* mat1_ptr,
    int64_t mat1_ld,
    const at::BFloat16* mat2_ptr,
    int64_t mat2_ld,
    const at::BFloat16* bias,
    at::BFloat16* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

/* LEVEL 2 BLAS FUNCTIONS */

#define GEMV_CHECK_ARGVALUES(Dtype)         \
  do {                                      \
    SUBLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
    SUBLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
    SUBLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
    SUBLAS_POSINT_CHECK(gemv<Dtype>, incx); \
    SUBLAS_POSINT_CHECK(gemv<Dtype>, incy); \
  } while (0)

template <>
void gemv<c10::complex<double>>(SUBLAS_GEMV_ARGTYPES(c10::complex<double>)) {
  TORCH_CHECK(false, "at::sublas::gemv does not support complex double dtype");
}

template <>
void gemv<c10::complex<float>>(SUBLAS_GEMV_ARGTYPES(c10::complex<float>)) {
  TORCH_CHECK(false, "at::sublas::gemv does not support complex float dtype");
}

template <>
void gemv<double>(SUBLAS_GEMV_ARGTYPES(double)) {
  TORCH_CHECK(false, "at::sublas::gemv does not support double dtype");
}

template <>
void gemv<float>(SUBLAS_GEMV_ARGTYPES(float)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  NoTF32Guard disable_tf32;
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  sublasOperation_t op = _sublasOpFromChar(trans);
  _sublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(float);
  AT_SUBLAS_CHECK(sublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(SUBLAS_GEMV_ARGTYPES(at::Half)) {
  // In general, sublas regards matrices as column-major.
  // The sublasS/Dgemv usages in sublas::gemv<float>/<double> above
  // require that external blas::gemv callers obey the following convention:
  //
  // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
  // caller interprets it as column-major with shape (summed, output), passes
  // summed and output respectively to our local vars m, n, and requests that sublas
  // internally transpose ("trans") the column-major interpretation of a.
  //
  // There's no such thing as "sublasHalfgemv", so here we hack gemv with a gemm.
  // However, we must allow the same calling convention, because the caller shouldn't
  // have to swap args based on whether it's calling blas::gemv<at::Half> or <float>.

  bool trans_bool = (_sublasOpFromChar(trans) != SUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  // After swap, local vars m, n contain the output and summed sizes respectively,
  // regardless of whether "a" was row-major or column-major in gemv<>'s caller.

  // To handle the possibility incy > 1, interprets vector y as column-major matrix with one row
  // (shape (1, output)) and leading dim incy.
  // trans(a)*x would compute a matrix with one column (shape (output, 1)) which wouldn't match y.
  // So instead, we interpret x similarly to y, as a column-major matrix with one row
  // (shape (1, summed)) and leading dim incx.  The gemm then carries out x*transpose(trans(a)) to
  // produce a matrix with one row (shape (1, output)), matching y.
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::Half>('n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

template <>
void gemv<at::BFloat16>(SUBLAS_GEMV_ARGTYPES(at::BFloat16)) {
  bool trans_bool = (_sublasOpFromChar(trans) != SUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::BFloat16>('n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

/* LEVEL 1 BLAS FUNCTIONS */

template <>
void dot<double>(SUBLAS_DOT_ARGTYPES(double)) {
  TORCH_CHECK(false, "at::sublas::dot does not support double dtype");
}

template <>
void dot<float>(SUBLAS_DOT_ARGTYPES(float)) {
  AT_SUBLAS_CHECK(sublasSdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<c10::complex<double>>(SUBLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CHECK(false, "at::sublas::dot does not support complex double dtype");
}

template <>
void dot<c10::complex<float>>(SUBLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CHECK(false, "at::sublas::dot does not support complex float dtype");
}

template <>
void dot<at::Half>(SUBLAS_DOT_ARGTYPES(at::Half)) {
  AT_SUBLAS_CHECK(sublasDotEx(handle, n, x, SUPA_R_16F, incx, y, SUPA_R_16F, incy, result, SUPA_R_16F, SUPA_R_32F));
}

template <>
void dot<at::BFloat16>(SUBLAS_DOT_ARGTYPES(at::BFloat16)) {
  AT_SUBLAS_CHECK(sublasDotEx(handle, n, x, SUPA_R_16BF, incx, y, SUPA_R_16BF, incy, result, SUPA_R_16BF, SUPA_R_32F));
}

template <>
void vdot<c10::complex<float>>(SUBLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CHECK(false, "at::sublas::vdot does not support complex float dtype")
}

template <>
void vdot<c10::complex<double>>(SUBLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CHECK(false, "at::sublas::vdot does not support complex double dtype")
}
} // namespace at::sublas
