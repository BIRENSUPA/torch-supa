/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/ops/kernels/kernelDispatch.h"

namespace at::supa {

void masked_fill_kernel(TensorIterator& iter, const Scalar& value);

Tensor& SUPANativeFunctions::masked_fill_(Tensor& self, const Tensor& mask, const Scalar& value) {
  TORCH_CHECK(
      self.device() == mask.device(),
      "expected self and mask to be on the same device, but got mask on ",
      mask.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      mask.scalar_type() == kBool, "masked_fill only supports boolean masks, but got dtype ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of masked_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_const_input(self)
                  .add_const_input(*b_mask)
                  .build();

  masked_fill_kernel(iter, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

} // namespace at::supa
