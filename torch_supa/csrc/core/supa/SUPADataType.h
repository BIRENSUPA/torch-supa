/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/ScalarType.h>

#include <supa.h>
#include <device/library_types.h>

namespace at::supa {

template <typename scalar_t>
supaDataType getSUPADataType() {
  static_assert(false && sizeof(scalar_t), "Cannot convert type to SUPADataType.");
  return {};
}

template<> inline supaDataType getSUPADataType<at::Half>() {
  return SUPA_R_16F;
}
template<> inline supaDataType getSUPADataType<float>() {
  return SUPA_R_32F;
}
template<> inline supaDataType getSUPADataType<c10::complex<c10::Half>>() {
  return SUPA_C_16F;
}
template<> inline supaDataType getSUPADataType<c10::complex<float>>() {
  return SUPA_C_32F;
}
template<> inline supaDataType getSUPADataType<uint8_t>() {
  return SUPA_R_8U;
}
template<> inline supaDataType getSUPADataType<int8_t>() {
  return SUPA_R_8I;
}
template<> inline supaDataType getSUPADataType<int>() {
  return SUPA_R_32I;
}
template<> inline supaDataType getSUPADataType<int16_t>() {
  return SUPA_R_16I;
}
template<> inline supaDataType getSUPADataType<int64_t>() {
  return SUPA_R_64I;
}
template<> inline supaDataType getSUPADataType<at::BFloat16>() {
  return SUPA_R_16BF;
}

inline supaDataType ScalarTypeToSUPADataType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return SUPA_R_8U;
    case c10::ScalarType::Char:
      return SUPA_R_8I;
    case c10::ScalarType::Int:
      return SUPA_R_32I;
    case c10::ScalarType::Half:
      return SUPA_R_16F;
    case c10::ScalarType::Float:
      return SUPA_R_32F;
    case c10::ScalarType::ComplexHalf:
      return SUPA_C_16F;
    case c10::ScalarType::ComplexFloat:
      return SUPA_C_32F;
    case c10::ScalarType::Short:
      return SUPA_R_16I;
    case c10::ScalarType::Long:
      return SUPA_R_64I;
    case c10::ScalarType::BFloat16:
      return SUPA_R_16BF;
    case c10::ScalarType::Float8_e4m3fn:
      return SUPA_R_8F_E4M3;
    case c10::ScalarType::Float8_e5m2:
      return SUPA_R_8F_E5M2;
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to SUPADataType.")
  }
}

} // namespace at::SUPA