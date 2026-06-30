/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Normalization.h>

#include <sudnn/sudnn.h>
#include <torch/csrc/autograd/custom_function.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"

namespace at::supa {

using namespace at::native;

enum class BNBackend {
  Native,
  Sudnn,
};

BNBackend _select_batch_norm_backend(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double eps) {
  auto& ctx = at::globalContext();
  bool sudnn_enabled = ctx.userEnabledCuDNN();

  if ((input.device().type() == c10::DeviceType::PrivateUse1) && input.scalar_type() != at::kBFloat16 &&
      weight.scalar_type() != at::kBFloat16 &&
      (input.scalar_type() != at::kHalf || weight.scalar_type() == at::kFloat) && weight.defined() && bias.defined() &&
      ((running_mean.defined() && running_var.defined()) ||
       (!running_mean.defined() && !running_var.defined() && training)) &&
      (input.dim() >= 3) &&
      ((input.sym_size(0) <= 880801 && training) // spatial, training
       || (input.sym_size(0) <= 65535 && !training)) // spatial, eval
      && eps >= SUDNN_BN_MIN_EPSILON && sudnn_enabled &&
      input.sym_numel() <
          std::numeric_limits<std::int32_t>::max() // some cuDNN kernels have 32-bit indexing limitations
  ) {
    return BNBackend::Sudnn;
  }

  return BNBackend::Native;
}

void check_dims_match_num_input_features(const char* arg_name, const SymInt& expected, const SymInt& actual) {
  TORCH_CHECK(actual == expected, arg_name, " should contain ", expected, " elements not ", actual);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> SUPANativeFunctions::_batch_norm_impl_index(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    const std::optional<Tensor>& running_mean_opt /* optional */,
    const std::optional<Tensor>& running_var_opt /* optional */,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());
  const Tensor& running_mean = running_mean_opt.value_or(Tensor());
  const Tensor& running_var = running_var_opt.value_or(Tensor());

  auto num_features = input.sym_sizes()[1];

  if (input.sym_numel() == 0) {
    Tensor reserve = at::empty({0}, input.options().dtype(kByte));
    auto options = input.options().dtype(at::toAccumulateType(input.scalar_type(), input.device().type()));
    auto save_mean = at::empty_symint(c10::SymIntArrayRef({num_features}), options);
    auto save_invstd = at::empty_symint(c10::SymIntArrayRef({std::move(num_features)}), options);

    // don't return view of input, don't return empty tensor because it will break gradient chain
    auto out = input.clone();
    if (weight.defined()) {
      out = out * weight[0];
    }
    if (bias.defined()) {
      out = out + bias[0];
    }
    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(out, save_mean, save_invstd, reserve, 0);
  }

  if (running_mean.defined()) {
    check_dims_match_num_input_features("running_mean", num_features, running_mean.sym_numel());
  } else if (!training) {
    TORCH_CHECK(false, "running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.sym_numel());
  } else if (!training) {
    TORCH_CHECK(false, "running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.sym_numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", std::move(num_features), bias.sym_numel());
  }

  BNBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, training, eps);

  if (backend == BNBackend::Sudnn) {
    auto input_c = input.contiguous(input.suggest_memory_format());
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();
    auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
    auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;

    auto [output, save_mean, save_var, reserve] =
        at::cudnn_batch_norm(input_c, weight_c, bias_c, rmean_c, rvar_c, training, momentum, eps);

    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(output, save_mean, save_var, reserve, 1);
  }

  Tensor reserve = at::empty({0}, input.options().dtype(kByte));

  return std::tuple_cat(
      at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps),
      std::tuple<Tensor>(reserve),
      std::make_tuple(0));
}

std::tuple<Tensor, Tensor, Tensor> SUPANativeFunctions::_batch_norm_impl_index_backward(
    int64_t impl_index,
    const Tensor& input,
    const Tensor& grad_output,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& running_mean_opt /* optional */,
    const std::optional<Tensor>& running_var_opt /* optional */,
    const std::optional<Tensor>& save_mean_opt /* optional */,
    const std::optional<Tensor>& save_var_transform_opt /* optional */,
    bool train,
    double epsilon,
    std::array<bool, 3> output_mask,
    const Tensor& reservedSpace) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& running_mean = running_mean_opt.value_or(Tensor());
  const Tensor& running_var = running_var_opt.value_or(Tensor());
  const Tensor& save_mean = save_mean_opt.value_or(Tensor());
  const Tensor& save_var_transform = save_var_transform_opt.value_or(Tensor());

  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // don't return empty tensor because it will break gradient chain
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (output_mask[2]) {
      grad_bias = grad_output.sum(dims);
    }
    if (output_mask[1]) {
      grad_weight = (grad_output * input).sum(dims);
    }
    if (output_mask[0] && weight.defined()) {
      grad_input = grad_output * weight[0];
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  // backward in inference mode is not supported in cudnn, fallback to native
  if (impl_index == 0 || (!train)) {
    return at::native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var_transform,
        train,
        epsilon,
        output_mask);
  }
  if (impl_index == 1) {
    // TODO: _batch_norm_impl_index_backward is only used in JIT. cudnn NHWC
    // format conversion is done inside cudnn_batch_norm_backward instead
    return at::cudnn_batch_norm_backward(
        input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon, reservedSpace);
  }
  TORCH_INTERNAL_ASSERT(false, "Unsupported impl_index in _batch_norm_impl_index_backward: ", impl_index);
}
} // namespace at::supa