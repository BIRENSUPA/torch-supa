/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/core/supa/SublasContext.h"
using namespace at::native;
namespace at::supa {
static inline int supa_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<int>(value);
  TORCH_CHECK(
      static_cast<int64_t>(result) == value,
      "supa_int_cast: The value of ",
      varname,
      "(",
      (long long)value,
      ") is too large to fit into a int (",
      sizeof(int),
      " bytes)");
  return result;
}

// Some suBLAS batched routines require input to be a device array of pointers to device individual matrices
// 'input' must be a contiguous tensor
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  int64_t input_mat_stride = matrixStride(input);

  // cublas/cusolver interface requires 'int'
  int batch_size = supa_int_cast(batchCount(input), "batch_size");

  // if batch_size==0, then start=0 and end=0
  // if input_mat_stride==0, then step=sizeof(scalar_t)
  return at::arange(
      /*start=*/reinterpret_cast<int64_t>(input_data),
      /*end=*/reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
      /*step=*/static_cast<int64_t>(std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
      input.options().dtype(at::kLong));
}

SUPA_IMPL_FUNC(_linalg_solve_ex)
(const Tensor& A,
 const Tensor& B,
 bool left,
 bool check_errors,
 const Tensor& result,
 const Tensor& LU,
 const Tensor& pivots,
 const Tensor& info) {
  TORCH_CHECK(at::ScalarType::Float == A.scalar_type(), "sublasSmatinvBatched only supports float32");
  TORCH_CHECK(left, "sublasSmatinvBatched only supports left solve");
  TORCH_CHECK(A.size(-2) == A.size(-1), "sublasSmatinvBatched requires square matrices");
  TORCH_CHECK(A.sizes().equals(result.sizes()), "result must have the same shape as A");

  // sublas input shoule be colmajor layout
  auto A_ = A.mT().contiguous();
  auto& result_mut = const_cast<Tensor&>(result);

  auto batch_size = supa_int_cast(batchCount(A_), "batch_size");
  auto n = supa_int_cast(A_.size(-2), "n");
  auto lda = supa_int_cast(std::max<int>(1, n), "lda");

  Tensor a_ptr_array = get_device_pointers<float>(A_);
  auto* a_ptr_array_data = reinterpret_cast<const float**>(a_ptr_array.data_ptr());

  Tensor result_ptr_array = get_device_pointers<float>(result_mut);
  auto* ainv_ptr_array_data = reinterpret_cast<float**>(result_ptr_array.data_ptr());

  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  int info_value = 0;
  AT_SUBLAS_CHECK(
      sublasSmatinvBatched(handle, n, a_ptr_array_data, lda, ainv_ptr_array_data, lda, &info_value, batch_size));
  info.fill_(info_value);
}

} // namespace at::supa
