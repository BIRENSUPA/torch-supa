/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <c10/util/Exception.h>
#include <supa_driver.h>

#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/aten/common/Resize.h"
#include "torch_supa/csrc/core/SUPAStorageImpl.h"
#include "torch_supa/csrc/core/supa/PinnedMemoryAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAHooks.h"
#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"

namespace c10::supa {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, SUPAHooks, SUPAHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, SUPAHooks, SUPAHooksArgs)

#if TORCH_VER >= TORCH_2_8_0
bool SUPAHooks::isAvailable() const {
  return c10::supa::is_available();
}
#endif

#if TORCH_VER >= TORCH_2_6_0
void SUPAHooks::init() const {
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
}
#elif TORCH_VER >= TORCH_2_3_0
void SUPAHooks::initPrivateUse1() const {
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
}
#endif

#if TORCH_VER >= TORCH_2_5_0
const at::Generator& SUPAHooks::getDefaultGenerator(c10::DeviceIndex device_index) const
#else
const at::Generator& SUPAHooks::getDefaultGenerator(c10::DeviceIndex device_index)
#endif
{
  static auto device_gen = at::supa::detail::getDefaultSUPAGenerator(device_index);
  return device_gen;
}

Device SUPAHooks::getDeviceFromPtr(void* data) const {
  supaPointerAttributes attr{};
  C10_SUPA_CHECK(supaPointerGetAttributes(&attr, data));
  return {c10::DeviceType::PrivateUse1, static_cast<DeviceIndex>(attr.device)};
}

#if TORCH_VER >= TORCH_2_3_0
bool SUPAHooks::hasPrimaryContext(c10::DeviceIndex device_index) const {
  TORCH_CHECK(
      device_index >= 0 && device_index < device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      device_index);

  unsigned int ctx_flags = 0;
  int ctx_is_active = 0;
  AT_SUPA_DRIVER_CHECK(suDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

Allocator* SUPAHooks::getPinnedMemoryAllocator() const {
  return at::supa::getPinnedMemoryAllocator();
}

void SUPAHooks::resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const {
  auto* storage_impl = storage.unsafeGetStorageImpl();
  at::supa::resize_bytes_supa(storage_impl, newsize);
}
#endif

#if TORCH_VER >= TORCH_2_5_0
bool SUPAHooks::isPinnedPtr(const void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!c10::supa::is_available()) {
    return false;
  }
  // supaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
  }
  supaPointerAttributes attr{};
  // We do not believe that SUPA needs mutable access to the data
  // here.
  supaError_t err = supaPointerGetAttributes(&attr, data);
  if (err == supaErrorInvalidValue) {
    (void)supaGetLastError(); // clear SUPA error
    return false;
  }
  C10_SUPA_CHECK(err);
  return attr.type == supaMemoryTypeHost;
}
#endif

at::PrivateUse1HooksInterface* get_supa_hooks() {
  static at::PrivateUse1HooksInterface* supa_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] { supa_hooks = new SUPAHooks(); });
  return supa_hooks;
}

} // namespace c10::supa
