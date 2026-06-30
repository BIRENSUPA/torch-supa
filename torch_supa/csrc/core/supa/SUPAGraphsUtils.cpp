/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace c10::supa {
CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  supaStreamCaptureStatus is_capturing{supaStreamCaptureStatusNone};
  C10_SUPA_CHECK(supaStreamIsCapturing(c10::supa::getCurrentSUPAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

} // namespace c10_npu