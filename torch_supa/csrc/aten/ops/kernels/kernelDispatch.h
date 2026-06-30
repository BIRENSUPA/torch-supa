/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/native/DispatchStub.h>

#include "torch_supa/csrc/utils/EnvConfig.h"

namespace at::native {

template <typename DispatchStub>
struct RegisterPRIVATEUSE1DispatchWithEnv {
  RegisterPRIVATEUSE1DispatchWithEnv(
      DispatchStub& stub,
      typename DispatchStub::FnPtr value) {
    if (!torch_supa::utils::EnvConfig::IsEnableNativeOP()) {
      stub.set_privateuse1_dispatch_ptr(value);
    }
  }
};

} // namespace at::native

#undef REGISTER_PRIVATEUSE1_DISPATCH
#define REGISTER_PRIVATEUSE1_DISPATCH(name, fn)                           \
  static at::native::RegisterPRIVATEUSE1DispatchWithEnv<                 \
      struct name##_DECLARE_DISPATCH_type>                               \
      name##__register(name, fn);
