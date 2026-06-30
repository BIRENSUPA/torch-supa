/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */


#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#include "torch_supa/csrc/aten/common/EmptyTensor.h"

namespace at::supa {

inline Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions& options) {
  if (strides.empty()) {
    return at::detail::empty_supa(sizes, options);
  }
  return at::detail::empty_strided_supa(sizes, strides, options);
}

inline void resize_out(const Tensor& out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

inline void check_inplace(const Tensor& self, IntArrayRef sizes, const TensorOptions& options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(
      options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ",
      self.dtype(),
      " and output tensor dtype ",
      options.dtype(),
      " should match");
  TORCH_CHECK(
      options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ",
      self.device(),
      " and output tensor device ",
      options.device(),
      " should match");
  TORCH_CHECK(
      sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ",
      self.sizes(),
      " and output tensor size ",
      sizes,
      " should match");
}

inline c10::optional<Tensor> maybe_create_proxy(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (out.strides() != strides) {
    return at::detail::empty_strided_supa(sizes, strides, options);
  }
  return c10::nullopt;
}

} // namespace at::supa
