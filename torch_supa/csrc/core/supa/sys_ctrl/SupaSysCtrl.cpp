/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"
#include "torch_supa/csrc/core/supa/PeerToPeerAccess.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"

namespace c10::supa {

SupaSysCtrl& SupaSysCtrl::GetInstance() {
  static SupaSysCtrl instance_;
  return instance_;
}

void SupaSysCtrl::supaInit() {
  std::call_once(init_flag_, [&]() {
    numDevices_ = static_cast<int>(c10::supa::device_count_ensure_non_zero());
    // set report_func with nullptr by default. TorchSysCtl should set it to
    // correct
    c10::supa::SUPACachingAllocator::init(numDevices_);
    at::supa::detail::init_p2p_access_cache(numDevices_);
    init_label_ = true;
    auto device = c10::supa::current_device();

    // p2pAccessEnabled_ records if p2p copies are allowed between pairs of
    // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
    // "-1" (unknown).
    p2pAccessEnabled_.assign(numDevices_, std::vector<int>(numDevices_, -1));
    for (int i = 0; i < numDevices_; ++i) {
      p2pAccessEnabled_[i][i] = 1;
    }

    for (int i = 0; i < numDevices_; ++i) {
      c10::supa::set_device(static_cast<c10::DeviceIndex>(i));
    }
    c10::supa::set_device(device);
  });
}

void SupaSysCtrl::supaInit(c10::DeviceIndex device_id) {
  auto device = c10::supa::current_device();
  device_id = (device_id < 0 || device_id >= c10::supa::device_count()) ? device : device_id;
  if (device_id != device) {
    c10::supa::set_device(device_id);
  }
}

} // namespace c10::supa
