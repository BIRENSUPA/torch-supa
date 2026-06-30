/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/jit_macros.h>
#include <ATen/OpMathType.h>

#include "Reduce.cuh"

namespace at::native {

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    const auto sum_combine = [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return a + b;
    };
    constexpr bool is_16_bits = sizeof(scalar_t) == 2;
    if constexpr (is_16_bits) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      // optimize vt0 based on reduce size
      int64_t reduce_size = iter.num_output_elements() == 0 ? 0 : iter.numel() / iter.num_output_elements();
      if (reduce_size % 1024 == 0) {
        gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/8, /*input_vec_size=*/8>(iter, func_wrapper<out_t>(sum_combine));
      } else {
        gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/4>(iter, func_wrapper<out_t>(sum_combine));
      }
    } else {
      gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    }
  }
};

// jiterated specialization for `complex<Half>`
constexpr char sum_name[] = "sum";
template <>
struct sum_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a + b;
    }
    );
    jitted_gpu_reduce_kernel<sum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }), acc_t{0.});
  }
#endif
};

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

static void sum_kernel_cuda(TensorIterator& iter){
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBool, kComplexHalf, iter.dtype(), "sum_cuda", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };

  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

REGISTER_PRIVATEUSE1_DISPATCH(sum_stub, &sum_kernel_cuda)

} // namespace at::native
