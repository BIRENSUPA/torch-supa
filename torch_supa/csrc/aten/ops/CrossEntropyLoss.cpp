#include <ATen/ATen.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/_log_softmax_backward_data.h>
#include <ATen/ops/cross_entropy_loss_native.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/nll_loss_backward.h>
#include <ATen/ops/nll_loss_forward.h>
#include <torch/csrc/autograd/custom_function.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace at::native {
TORCH_SUPA_API Tensor
cross_entropy_loss_2d_kernel_supa(const Tensor& self, const Tensor& target, int64_t reduction, int64_t ignore_index);
} // namespace at::native

namespace at::supa {

namespace {

bool supported_fast_path(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    double label_smoothing) {
  const bool has_weight = weight_opt.has_value() && weight_opt->defined();
  return self.dim() == 2 && self.scalar_type() == at::kFloat && !has_weight && label_smoothing == 0.0 &&
      (target.scalar_type() == at::kLong || (target.scalar_type() == at::kFloat && target.sizes() == self.sizes())) &&
      (reduction == at::Reduction::None || reduction == at::Reduction::Mean || reduction == at::Reduction::Sum);
}

Tensor cross_entropy_loss_forward(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  if (supported_fast_path(self, target, weight_opt, reduction, label_smoothing)) {
    if (target.scalar_type() == at::kFloat) {
      TORCH_CHECK(ignore_index < 0, "ignore_index is not supported for floating point target");
    }
    return at::native::cross_entropy_loss_2d_kernel_supa(
        self.contiguous(), target.contiguous(), reduction, ignore_index);
  }

  return at::native::cross_entropy_loss_symint(
      self, target, weight_opt, reduction, c10::SymInt(ignore_index), label_smoothing);
}

Tensor unsqueeze_class_dim(const Tensor& input, int64_t class_dim) {
  return input.unsqueeze(class_dim);
}

class CrossEntropyLossFastPathFunction : public torch::autograd::Function<CrossEntropyLossFastPathFunction> {
 public:
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& self,
      const Tensor& target,
      const std::optional<Tensor>& weight_opt,
      int64_t reduction,
      int64_t ignore_index,
      double label_smoothing) {
    TORCH_CHECK(
        supported_fast_path(self, target, weight_opt, reduction, label_smoothing),
        "CrossEntropyLossFastPathFunction expects supported fast path inputs.");
    if (target.scalar_type() == at::kFloat) {
      TORCH_CHECK(ignore_index < 0, "ignore_index is not supported for floating point target");
    }
    const auto class_dim = self.dim() == 1 ? 0 : 1;
    const auto input_dtype = self.scalar_type();
    auto log_probs = at::log_softmax(self, class_dim, self.scalar_type());
    Tensor total_weight;
    if (target.scalar_type() == at::kLong) {
      auto nll_loss = at::nll_loss_forward(log_probs, target, std::nullopt, reduction, ignore_index);
      total_weight = std::get<1>(nll_loss);
    }

    ctx->save_for_backward({target, log_probs, total_weight});
    ctx->saved_data["prob_target"] = target.scalar_type() != at::kLong;
    ctx->saved_data["class_dim"] = class_dim;
    ctx->saved_data["input_dtype"] = static_cast<int64_t>(input_dtype);
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["ignore_index"] = ignore_index;

    return at::native::cross_entropy_loss_2d_kernel_supa(
        self.contiguous(), target.contiguous(), reduction, ignore_index);
  }

  static std::vector<Tensor> backward(torch::autograd::AutogradContext* ctx, std::vector<Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto target = saved[0];
    auto saved_log_probs = saved[1];
    auto total_weight = saved[2];
    const bool prob_target = ctx->saved_data["prob_target"].toBool();
    const int64_t class_dim = ctx->saved_data["class_dim"].toInt();
    const auto input_dtype = static_cast<at::ScalarType>(ctx->saved_data["input_dtype"].toInt());
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    const int64_t ignore_index = ctx->saved_data["ignore_index"].toInt();

    Tensor grad_self;
    Tensor grad_target;

    at::AutoDispatchBelowAutograd guard;
    Tensor grad_log_probs;
    if (prob_target) {
      const auto n_classes = saved_log_probs.size(class_dim);
      Tensor grad_factor;
      if (reduction == at::Reduction::Mean) {
        grad_factor = grad_outputs[0] / (saved_log_probs.numel() / n_classes);
      } else if (reduction == at::Reduction::Sum) {
        grad_factor = grad_outputs[0];
      } else if (reduction == at::Reduction::None) {
        grad_factor = unsqueeze_class_dim(grad_outputs[0], class_dim);
      } else {
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
      }
      grad_log_probs = -target * grad_factor;
      if (ctx->needs_input_grad(1)) {
        grad_target = -saved_log_probs * grad_factor;
      }
    } else {
      grad_log_probs = at::nll_loss_backward(
          grad_outputs[0], saved_log_probs, target, std::nullopt, reduction, ignore_index, total_weight);
    }

    if (ctx->needs_input_grad(0)) {
      grad_self = at::_log_softmax_backward_data(grad_log_probs, saved_log_probs, class_dim, input_dtype);
    }
    return {grad_self, grad_target, Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

} // namespace

Tensor SUPANativeFunctions::cross_entropy_loss(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  if (!supported_fast_path(self, target, weight_opt, reduction, label_smoothing)) {
    return cross_entropy_loss_forward(self, target, weight_opt, reduction, ignore_index, label_smoothing);
  }
  return CrossEntropyLossFastPathFunction::apply(self, target, weight_opt, reduction, ignore_index, label_smoothing);
}

} // namespace at::supa
