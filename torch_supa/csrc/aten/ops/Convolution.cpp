/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <limits>
#include <utility>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/utils/EnvConfig.h"

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/core/GradMode.h>

namespace at::supa {

enum class ConvBackend {
  SupaDepthwise2d,
  SupaDepthwise3d,
  Slow2d,
  SlowDilated2d,
  SlowTranspose2d,
  Sudnn,
  SudnnTranspose,
};

inline at::MemoryFormat sudnn_conv_suggest_memory_format(const at::Tensor& input, const at::Tensor& weight) {
  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();
  auto weight_ndim = weight.ndimension();

  bool can_use_sudnn_channels_last_2d = (weight_ndim == 4) &&
      ((input_memory_format == at::MemoryFormat::ChannelsLast) ||
       (weight_memory_format == at::MemoryFormat::ChannelsLast));
  if (can_use_sudnn_channels_last_2d) {
    return at::MemoryFormat::ChannelsLast;
  }

  bool can_use_sudnn_channels_last_3d = (weight_ndim == 5) &&
      ((input_memory_format == at::MemoryFormat::ChannelsLast3d) ||
       (weight_memory_format == at::MemoryFormat::ChannelsLast3d));
  if (can_use_sudnn_channels_last_3d) {
    return at::MemoryFormat::ChannelsLast3d;
  }

  return at::MemoryFormat::Contiguous;
}

// This struct is templated so that we can run backend selection in a dynamic
// shapes context; all of the real kernel selection in eager mode runs with
// int64_t
template <typename T>
struct ConvParams {
  std::vector<T> stride;
  std::vector<T> padding;
  std::vector<T> dilation;
  bool transposed{};
  std::vector<T> output_padding;
  T groups{};
  bool benchmark{};
  bool deterministic{};
  bool sudnn_enabled{};
  bool allow_tf32{};

  bool is_strided() const {
    bool is_strided = false;
    for (const auto& s : stride) {
      is_strided |= (s != 1);
    }
    return is_strided;
  }

  bool is_dilated() const {
    bool is_dilated = false;
    for (const auto& d : dilation) {
      is_dilated |= (d != 1);
    }
    return is_dilated;
  }

  bool is_padded() const {
    bool is_padded = false;
    for (auto p : padding) {
      is_padded |= (p != 0);
    }
    return is_padded;
  }

  bool is_output_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : output_padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_output_padding_big() const {
    bool is_big = false;
    for (auto i : c10::irange(output_padding.size())) {
      is_big |= (output_padding[i] >= stride[i]);
    }
    return is_big;
  }

  bool is_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_dilation_neg() const {
    bool is_non_neg = false;
    for (const auto& p : dilation) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_stride_nonpos() const {
    bool is_nonpos = false;
    for (const auto& s : stride) {
      is_nonpos |= (s <= 0);
    }
    return is_nonpos;
  }

  void view1d_as_2d() {
    if (stride.size() == 1) {
      stride.insert(stride.begin(), 1);
      padding.insert(padding.begin(), 0);
      dilation.insert(dilation.begin(), 1);
      output_padding.insert(output_padding.begin(), 0);
    }
  }

  bool use_sudnn(const at::Tensor& input, const at::Tensor& weight) const {
    if (!input.device().is_privateuseone() || !sudnn_enabled) {
      return false;
    }

    if (sudnn_conv_suggest_memory_format(input, weight) == at::MemoryFormat::Contiguous) {
      // bypass dilation checks for channels_last convolution
      if (deterministic && is_dilated()) {
        // sudnn doesn't support deterministic dilated convolution fully yet
        return false;
      }
      if (is_dilated()) {
        return !is_output_padding_big();
      }
    }
    return !is_output_padding_big();
  }

  // Use sudnn for depthwise convolutions when the selected math mode expects it.
  bool use_sudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const {
    if (!use_sudnn(input, weight)) {
      return false;
    }
    if (input.ndimension() == 5) {
      return false;
    }
    if (sudnn_conv_suggest_memory_format(input, weight) != at::MemoryFormat::Contiguous) {
      // Always use sudnn_depthwise for channels-last format.
      return true;
    }
    return deterministic && allow_tf32 && input.scalar_type() == at::kFloat;
  }

  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const {
    return input.device().is_privateuseone() && !transposed && (input.ndimension() == 4 || input.ndimension() == 5) &&
        at::symint::size<T>(input, 1) == groups && groups > 1 && // no point if there is only a single group
        at::symint::size<T>(weight, 0) % at::symint::size<T>(input, 1) ==
        0; // output channels must be a multiple of input channels
  }
};

inline Tensor reshape_bias(int64_t dim, const Tensor& bias) {
  std::vector<int64_t> shape(dim, 1);
  shape[1] = -1;
  return bias.reshape(shape);
}

static void check_input_same_type_as_parameters(const Tensor& input, const Tensor& weight, const Tensor& bias) {
  TORCH_CHECK(
      input.options().type_equal(weight.options()),
      "Input type (",
      input.toString(),
      ") and weight type (",
      weight.toString(),
      ") should be the same");
  TORCH_CHECK(
      !bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (",
      input.toString(),
      ") and bias type (",
      bias.toString(),
      ") should be the same");
}

static void check_input_same_type_as_parameters(const Tensor& input, const Tensor& weight) {
  check_input_same_type_as_parameters(input, weight, /*bias=*/Tensor());
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(
      tensor.ndimension() == 3, "expected 3D tensor, got tensor with ", tensor.ndimension(), " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(
      tensor.ndimension() == 4, "expected 4D tensor, got tensor with ", tensor.ndimension(), " dimensions instead");
  return tensor.squeeze(2);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const ConvParams<T>& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride} << "  padding = " << ArrayRef<T>{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation} << "  transposed = " << params.transposed
      << "  output_padding = " << ArrayRef<T>{params.output_padding} << "  groups = " << params.groups
      << "  benchmark = " << params.benchmark << "  deterministic = " << params.deterministic
      << "  sudnn_enabled = " << params.sudnn_enabled << "  allow_tf32 = " << params.allow_tf32 << "}";
  return out;
}

template <typename T>
static void check_shape_forward(
    const at::Tensor& input,
    const c10::ArrayRef<T>& weight_sizes,
    const at::Tensor& bias,
    const ConvParams<T>& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  auto groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");
  TORCH_CHECK(!params.is_dilation_neg(), "dilation should be greater than zero");

  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      at::symint::sizes<T>(input),
      " instead");
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size [",
      weight_sizes,
      "] instead");

  if (!transposed) {
    std::vector<T> input_shape;
    std::vector<T> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(
        at::symint::size<T>(input, 1) == (weight_sizes[1] * groups),
        "Given groups=",
        groups,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        at::symint::sizes<T>(input),
        " to have ",
        (weight_sizes[1] * groups),
        " channels, but got ",
        at::symint::size<T>(input, 1),
        " channels instead");

    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        at::symint::sizes<T>(bias),
        " instead");

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(at::symint::size<T>(input, i) + 2 * padding[i - 2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator;

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(
          false,
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(
        at::symint::size<T>(input, 1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        at::symint::sizes<T>(input),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        at::symint::size<T>(input, 1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        at::symint::sizes<T>(bias),
        " instead");
  }
}

template <typename T>
static void check_shape_backward(
    const at::Tensor& input,
    const c10::ArrayRef<T>& weight_sizes,
    const ConvParams<T>& params) {
  check_shape_forward<T>(input, weight_sizes, /*bias=*/Tensor(), params);
}

template <typename T>
ConvBackend select_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const bool need_backward,
    const ConvParams<T>& params) {
  if (at::symint::numel<T>(input) == 0) {
    TORCH_CHECK(
        false,
        "Only zero batch or zero channel inputs are supported, but got "
        "input shape: ",
        at::symint::sizes<T>(input));
  }

  if (params.is_depthwise(input, weight)) {
    if (params.use_sudnn_depthwise(input, weight)) {
      return ConvBackend::Sudnn;
    }
    if (input.ndimension() == 4) {
      return ConvBackend::SupaDepthwise2d;
    }
    if (input.ndimension() == 5) {
      return ConvBackend::SupaDepthwise3d;
    }
    // unsupported
    TORCH_CHECK(false, "Unsupported conv condition!");
  }

  if (params.use_sudnn(input, weight)) {
    if (params.transposed) {
      return ConvBackend::SudnnTranspose;
    }
    return ConvBackend::Sudnn;
  }

  if (!params.transposed && input.ndimension() == 4 && params.groups == 1) {
    if (params.is_dilated()) {
      return ConvBackend::SlowDilated2d;
    }
    return ConvBackend::Slow2d;
  }

  if (params.transposed && input.ndimension() == 4) {
    return ConvBackend::SlowTranspose2d;
  }
  TORCH_CHECK(false, "unsupported ConvNd parameters");
}

static inline at::MemoryFormat determine_backend_memory_format(
    const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend) {
  at::MemoryFormat backend_memory_format = at::MemoryFormat::Contiguous;
  // auto k = weight.ndimension();
  switch (backend) {
    case ConvBackend::Sudnn:
    case ConvBackend::SudnnTranspose:
      backend_memory_format = sudnn_conv_suggest_memory_format(input, weight);
      break;
    default:
      backend_memory_format = at::MemoryFormat::Contiguous;
  }
  return backend_memory_format;
}

Tensor SUPANativeFunctions::convolution_overrideable(
    const Tensor& input_,
    const Tensor& weight_,
    const optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto input = input_;
  auto weight = weight_;
  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");

  ConvParams<int64_t> params;
  auto& ctx = at::globalContext();
  params.stride = at::native::expand_param_if_needed(stride, "stride", dim);
  params.padding = at::native::expand_param_if_needed(padding, "padding", dim);
  params.dilation = at::native::expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = at::native::expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;
  params.benchmark = ctx.benchmarkCuDNN();
  params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  params.sudnn_enabled = ctx.userEnabledCuDNN();
  params.allow_tf32 = ctx.allowTF32CuDNN();

  check_shape_forward(input, weight.sizes(), bias, params);

  if (k == 3) {
    input = input.contiguous();
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  bool need_backward = GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  ConvBackend backend = select_conv_backend(input, weight, need_backward, params);
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  // Call the backend.
  Tensor output;
  auto kernel_size = weight.sizes().slice(2);
  switch (backend) {
    case ConvBackend::SupaDepthwise2d:
      output = at::_conv_depthwise2d(
          input.contiguous(), weight, kernel_size, bias, params.stride, params.padding, params.dilation);
      break;
    case ConvBackend::SupaDepthwise3d:
      output = at::conv_depthwise3d(
          input.contiguous(), weight, kernel_size, bias, params.stride, params.padding, params.dilation);
      break;
    case ConvBackend::Slow2d:
      output = at::_slow_conv2d_forward(
          input.contiguous(), weight.contiguous(), kernel_size, bias, params.stride, params.padding);
      break;
    case ConvBackend::SlowDilated2d:
      output =
          at::slow_conv_dilated2d(input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
      break;
    case ConvBackend::SlowTranspose2d:
      output = at::slow_conv_transpose2d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
      break;
    case ConvBackend::Sudnn:
      check_input_same_type_as_parameters(input, weight, bias);
      output = at::cudnn_convolution(
          input.contiguous(backend_memory_format),
          weight,
          params.padding,
          params.stride,
          params.dilation,
          params.groups,
          params.benchmark,
          params.deterministic,
          params.allow_tf32);
      if (bias.defined()) {
        output.add_(reshape_bias(input.dim(), bias));
      }
      break;
    case ConvBackend::SudnnTranspose:
      check_input_same_type_as_parameters(input, weight, bias);
      output = at::cudnn_convolution_transpose(
          input.contiguous(backend_memory_format),
          weight,
          params.padding,
          params.output_padding,
          params.stride,
          params.dilation,
          params.groups,
          params.benchmark,
          params.deterministic,
          params.allow_tf32);
      if (bias.defined()) {
        output.add_(reshape_bias(input.dim(), bias));
      }
      break;
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

std::tuple<Tensor, Tensor, Tensor> SUPANativeFunctions::convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
  ConvParams<int64_t> params;
  auto& ctx = at::globalContext();
  params.stride = at::native::expand_param_if_needed(stride, "stride", dim);
  params.padding = at::native::expand_param_if_needed(padding, "padding", dim);
  params.dilation = at::native::expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = at::native::expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;
  params.benchmark = ctx.benchmarkCuDNN();
  params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  params.sudnn_enabled = ctx.userEnabledCuDNN();
  params.allow_tf32 = ctx.allowTF32CuDNN();

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(
      input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of "
      "dimensions, but got: ",
      input.dim(),
      " and ",
      grad_output.dim());

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(
          pad == 0,
          "output_padding is not supported for non-transposed "
          "convolutions; got: ",
          params.output_padding);
    }
  }
  ConvBackend backend = select_conv_backend(input, weight, /*need_backward=*/true, params);
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  // Call the backend.
  Tensor backend_grad_input;
  Tensor backend_grad_weight;
  Tensor backend_grad_bias;
  auto kernel_size = weight.sizes().slice(2);
  switch (backend) {
    case ConvBackend::SupaDepthwise2d: {
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) = at::native::conv_depthwise2d_backward_stub(
          input.device().type(),
          grad_output,
          input,
          weight,
          kernel_size,
          params.stride,
          params.padding,
          params.dilation,
          input_weight_output_mask);
      break;
    }
    case ConvBackend::SupaDepthwise3d:
      TORCH_CHECK(input.ndimension() == 5);
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = at::native::conv_depthwise3d_backward_stub(
          input.device().type(),
          grad_output,
          input,
          weight,
          kernel_size,
          params.stride,
          params.padding,
          params.dilation,
          output_mask);
      break;
    case ConvBackend::Slow2d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = at::_slow_conv2d_backward(
          grad_output.contiguous(),
          input.contiguous(),
          weight.contiguous(),
          kernel_size,
          params.stride,
          params.padding,
          output_mask);
      break;
    case ConvBackend::SlowDilated2d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
          at::native::slow_conv_dilated2d_backward_stub(
              input.device().type(),
              grad_output,
              input,
              weight,
              kernel_size,
              params.stride,
              params.padding,
              params.dilation,
              output_mask);
      break;
    case ConvBackend::SlowTranspose2d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
          at::native::slow_conv_transpose2d_backward_stub(
              input.device().type(),
              grad_output,
              input,
              weight,
              kernel_size,
              params.stride,
              params.padding,
              params.output_padding,
              params.dilation,
              output_mask);
      break;
    case ConvBackend::Sudnn: {
      check_input_same_type_as_parameters(input, weight);
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) = at::native::cudnn_convolution_backward_stub(
          input.device().type(),
          // Only make input contiguous when it is necessary for the backwards
          // computation
          output_mask[1] ? input.contiguous(backend_memory_format) : input,
          grad_output,
          weight,
          params.padding,
          params.stride,
          params.dilation,
          params.groups,
          params.benchmark,
          params.deterministic,
          params.allow_tf32,
          input_weight_output_mask);
      break;
    }
    case ConvBackend::SudnnTranspose: {
      check_input_same_type_as_parameters(input, weight);
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) = at::native::cudnn_convolution_transpose_backward_stub(
          input.device().type(),
          // Only make input contiguous when it is necessary for the backwards
          // computation
          output_mask[1] ? input.contiguous(backend_memory_format) : input,
          grad_output,
          weight,
          params.padding,
          params.output_padding,
          params.stride,
          params.dilation,
          params.groups,
          params.benchmark,
          params.deterministic,
          params.allow_tf32,
          input_weight_output_mask);
      break;
    }
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0]) {
    if (k == 3) {
      backend_grad_input = view3d(backend_grad_input);
    }
  }
  if (output_mask[1]) {
    if (k == 3) {
      backend_grad_weight = view3d(backend_grad_weight);
    }
  }
  if (output_mask[2]) {
    if (!backend_grad_bias.defined()) {
      // Calculate bias gradients outside of the backend for those that don't
      // support it.
      backend_grad_bias = grad_output.sum((dim == 3) ? IntArrayRef{0, 2, 3, 4} : IntArrayRef{0, 2, 3});
    }
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

} // namespace at::supa
