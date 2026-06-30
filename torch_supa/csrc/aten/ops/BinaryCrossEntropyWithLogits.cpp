/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace at::native {
TORCH_SUPA_API void binary_cross_entropy_with_logits_kernel_supa(
    TensorIteratorBase& iter,
    bool has_pos_weight,
    bool has_weight);
} // namespace at::native

namespace at::supa {

using namespace at::native;

Tensor SUPANativeFunctions::binary_cross_entropy_with_logits(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction) {
  TORCH_CHECK(
      self.sizes() == target.sizes(),
      "Target size (",
      target.sizes(),
      ") must be the same as input size (",
      self.sizes(),
      ")");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "binary_cross_entropy_with_logits is only implemented for floating dtypes");

  auto loss = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  TensorIteratorConfig config;
  config.add_output(loss)
      .add_input(self)
      .add_input(target)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true);

  if (pos_weight_opt.has_value() && pos_weight_opt->defined()) {
    config.add_input(*pos_weight_opt);
  }
  if (weight_opt.has_value() && weight_opt->defined()) {
    config.add_input(*weight_opt);
  }

  auto iter = config.build();
  binary_cross_entropy_with_logits_kernel_supa(
      iter,
      pos_weight_opt.has_value() && pos_weight_opt->defined(),
      weight_opt.has_value() && weight_opt->defined());

  if (reduction == at::Reduction::None) {
    return loss;
  }
  if (reduction == at::Reduction::Mean) {
    return loss.mean();
  }
  if (reduction == at::Reduction::Sum) {
    return loss.sum();
  }
  TORCH_CHECK(false, "invalid reduction enum: ", reduction);
}

} // namespace at::supa
