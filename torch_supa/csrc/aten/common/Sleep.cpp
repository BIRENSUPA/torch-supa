/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/common/Sleep.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace at {
namespace supa {

void sleep_kernel(int64_t cycles, supaStream_t stream);

void sleep(int64_t cycles) {
  sleep_kernel(cycles, c10::supa::getCurrentSUPAStream());
}

} // namespace supa
} // namespace at
