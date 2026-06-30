/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <c10/util/string_view.h>
#include <c10/util/Array.h>
#include <cmath>
#include <cstdint>
#include <functional>

namespace sdp {
    
inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

}