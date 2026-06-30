/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */


#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>

#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"

namespace at::supa {

TORCH_SUPA_API void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kBool, kHalf, kBFloat16, kComplexHalf, iter.common_dtype(), "masked_fill_", [&]() {
        const auto value_ = value.to<scalar_t>();
        at::native::gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

} // namespace at::supa
