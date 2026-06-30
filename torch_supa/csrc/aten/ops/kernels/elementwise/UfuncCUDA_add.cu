/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"
#include <c10/core/Scalar.h>
#include <c10/macros/Macros.h>
#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"

namespace at::native::ufunc {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self + alpha * other;
}

} // namespace at::native::ufunc

namespace at {

// NB: this is explicitly copied here (via codegen) rather than
// included via NativeFunctions.h to avoid recompiling this file when
// NativeFunctions.h changes
namespace meta {
struct TORCH_API structured_add_Tensor : public TensorIteratorBase {


    void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};
}

namespace native {
struct TORCH_API structured_ufunc_add_CUDA : public at::meta::structured_add_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};


template <typename scalar_t>
struct CUDAFunctorOnSelf_add {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t other_;
opmath_t alpha_;
  CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha) : other_(other), alpha_(alpha) {}
  __host__ __device__ scalar_t operator()(scalar_t self) const {
    return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
  }
};


template <typename scalar_t>
struct CUDAFunctorOnOther_add {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t self_;
opmath_t alpha_;
  CUDAFunctorOnOther_add(opmath_t self, opmath_t alpha) : self_(self), alpha_(alpha) {}
  __host__ __device__ scalar_t operator()(scalar_t other) const {
    return ufunc::add(self_, static_cast<opmath_t>(other), alpha_);
  }
};


template <typename scalar_t>
struct CUDAFunctor_add {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha_;
  CUDAFunctor_add(opmath_t alpha) : alpha_(alpha) {}
  __host__ __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return ufunc::add(static_cast<opmath_t>(self), static_cast<opmath_t>(other), alpha_);
  }
};


using add_fn = void(*)(TensorIteratorBase&, const at::Scalar &);
DECLARE_DISPATCH(add_fn, add_stub)

void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "ufunc_add_CUDA",

AT_DISPATCH_CASE(at::ScalarType::Bool,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Byte,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Char,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Int,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Long,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Short,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Float,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


// AT_DISPATCH_CASE(at::ScalarType::Double,
//   [&]() {
//     using opmath_t = at::opmath_type<scalar_t>;if (false) {}
// else if (iter.is_cpu_scalar(1)) {
//   CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
//   iter.remove_operand(1);
//   gpu_kernel(iter, ufunctor);
// }else if (iter.is_cpu_scalar(2)) {
//   CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
//   iter.remove_operand(2);
//   gpu_kernel(iter, ufunctor);
// }
// else {
//   gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
// }

//   }
// )


AT_DISPATCH_CASE(at::ScalarType::ComplexFloat,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


// AT_DISPATCH_CASE(at::ScalarType::ComplexDouble,
//   [&]() {
//     using opmath_t = at::opmath_type<scalar_t>;if (false) {}
// else if (iter.is_cpu_scalar(1)) {
//   CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
//   iter.remove_operand(1);
//   gpu_kernel(iter, ufunctor);
// }else if (iter.is_cpu_scalar(2)) {
//   CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
//   iter.remove_operand(2);
//   gpu_kernel(iter, ufunctor);
// }
// else {
//   gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
// }

//   }
// )


AT_DISPATCH_CASE(at::ScalarType::BFloat16,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::Half,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)


AT_DISPATCH_CASE(at::ScalarType::ComplexHalf,
  [&]() {
    using opmath_t = at::opmath_type<scalar_t>;if (false) {}
else if (iter.is_cpu_scalar(1)) {
  CUDAFunctorOnOther_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(1), (alpha).to<opmath_t>());
  iter.remove_operand(1);
  gpu_kernel(iter, ufunctor);
}else if (iter.is_cpu_scalar(2)) {
  CUDAFunctorOnSelf_add<scalar_t> ufunctor(iter.scalar_value<opmath_t>(2), (alpha).to<opmath_t>());
  iter.remove_operand(2);
  gpu_kernel(iter, ufunctor);
}
else {
  gpu_kernel(iter, CUDAFunctor_add<scalar_t>((alpha).to<opmath_t>()));
}

  }
)

  );
}
REGISTER_PRIVATEUSE1_DISPATCH(add_stub, &add_kernel)

TORCH_IMPL_FUNC(ufunc_add_CUDA)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_stub(kPrivateUse1, *this, alpha);
}
}} // namespace at::native
