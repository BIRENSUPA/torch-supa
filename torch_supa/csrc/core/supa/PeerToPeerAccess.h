/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include <c10/core/Device.h>
#include <cstdint>

namespace at::supa {
namespace detail {
C10_SUPA_API void init_p2p_access_cache(int64_t num_devices);
}

C10_SUPA_API bool get_p2p_access(c10::DeviceIndex source_dev,
                                   c10::DeviceIndex dest_dev);
C10_SUPA_API bool get_fabric_access(c10::DeviceIndex device);

} // namespace at::supa