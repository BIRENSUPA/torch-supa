/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"
#include <ATen/native/SharedReduceOps.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

#include "Reduce.cuh"

namespace at::native {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  constexpr bool is_16_bits = sizeof(scalar_t) == 2;
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  auto op = MeanOps<scalar_t, acc_t, factor_t, out_t>{factor};
  if constexpr (is_16_bits) {
    gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(iter, op);
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    // optimize vt0 based on reduce size
    int64_t reduce_size = iter.num_output_elements() == 0 ? 0 : iter.numel() / iter.num_output_elements();
    if (reduce_size % 1024 == 0) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/8, /*input_vec_size=*/8>(iter, op);
    } else {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/4>(iter, op);
    }
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, op);
  }
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if(iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      mean_kernel_impl<scalar_t>(iter);
    });
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(mean_stub, &mean_kernel_cuda)

} // namespace at::native
