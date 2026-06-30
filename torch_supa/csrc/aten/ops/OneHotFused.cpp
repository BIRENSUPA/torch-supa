/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/profiler/supa_profiler.h"

namespace at::supa {

void one_hot_fused_kernel_launch(const int64_t* index, int64_t* output, long long rows, int num_classes);

at::Tensor SUPANativeFunctions::one_hot_fused(const at::Tensor& self, int64_t num_classes) {
  torch_supa::profiler::SUPARecordFunction recorder;
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "one_hot_fused expects a SUPA tensor");
  TORCH_CHECK(self.scalar_type() == at::kLong, "one_hot_fused expects int64 input");
  TORCH_CHECK(num_classes > 0, "one_hot_fused expects num_classes > 0");
  TORCH_CHECK(self.is_contiguous(), "one_hot_fused expects contiguous input");

  auto out_sizes = self.sizes().vec();
  out_sizes.push_back(num_classes);
  auto output = at::empty(out_sizes, self.options());

  auto rows = static_cast<long long>(self.numel());
  if (rows > 0) {
    one_hot_fused_kernel_launch(
        self.data_ptr<int64_t>(), output.data_ptr<int64_t>(), rows, static_cast<int>(num_classes));
  }
  return output;
}

} // namespace at::supa
