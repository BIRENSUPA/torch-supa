/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/core/Tensor.h>
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"

namespace at::supa {
namespace native {

void record_stream(Tensor &self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::supa::SUPACachingAllocator::recordStream(
      self.storage().data_ptr(),
      c10::supa::SUPAStream::unpack3(data.stream_id, data.device_index,
                                     data.device_type));
}
} // namespace native

void SUPANativeFunctions::record_stream(at::Tensor &self, at::Stream s) {
  return at::supa::native::record_stream(self, s);
}

} // namespace at::supa
