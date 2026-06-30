/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */


#pragma once

#include <ATen/native/TensorIterator.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>

#include <string>

namespace at::supa {

inline std::string format_int_list(c10::IntArrayRef values) {
  std::string s = "[";
  for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
    if (i) {
      s += ", ";
    }
    s += std::to_string(values[i]);
  }
  s += "]";
  return s;
}

inline std::string format_tensor_iterator_operand(const TensorIteratorBase& iter, int tensor_idx) {
  std::string s = (tensor_idx < iter.noutputs()) ? "output(" : "input(";
  s += "shape=" + format_int_list(iter.shape());
  s += ", stride=" + format_int_list(iter.strides(tensor_idx));
  s += ", dtype=";
  s += toString(iter.dtype(tensor_idx));
  s += ", device=" + c10::str(iter.device(tensor_idx));
  s += ", numel=" + std::to_string(iter.numel());
  s += ", data_ptr=" + std::to_string(reinterpret_cast<int64_t>(iter.data_ptr(tensor_idx)));
  s += ")\n";
  return s;
}

inline std::string format_tensor_iterator(const TensorIteratorBase& iter) {
  std::string s = "TensorIterator(shape=" + format_int_list(iter.shape());
  s += ", ndim=" + std::to_string(iter.ndim());
  s += ", numel=" + std::to_string(iter.numel());
  s += ", ntensors=" + std::to_string(iter.ntensors());
  s += ", noutputs=" + std::to_string(iter.noutputs());
  s += ", contiguous=" + std::string(iter.is_contiguous() ? "true" : "false");
  s += ", operands=[\n";
  for (int i = 0; i < iter.ntensors(); ++i) {
    s += format_tensor_iterator_operand(iter, i);
  }
  s += "])";
  return s;
}

} // namespace at::supa
