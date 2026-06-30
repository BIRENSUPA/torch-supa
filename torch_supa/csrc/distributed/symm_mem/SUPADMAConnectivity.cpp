/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <fmt/printf.h>
#include "torch_supa/csrc/core/supa/DriverAPI.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#include <brml.h>

#if TORCH_VER >= TORCH_2_5_0

#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#elif TORCH_VER >= TORCH_2_5_0
#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>
#endif

namespace {

constexpr int max_blinks = 64;

std::string get_bus_id(int device_idx) {
  supaDeviceProp prop{};
  C10_SUPA_CHECK(supaGetDeviceProperties(&prop, device_idx));
  return fmt::sprintf(BRML_DEVICE_PCI_BUS_ID_FMT, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
}

struct C10_EXPORT BLinkDetector : public c10d::DMAConnectivityDetector {
  c10::intrusive_ptr<c10d::DMAConnectivity> detect() override {
    int num_devices = 0;
    C10_SUPA_CHECK(supaGetDeviceCount(&num_devices));

    std::vector<std::vector<int>> matrix;
    matrix.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      matrix.emplace_back(num_devices, 0);
    }

    // Obtain the bus_id for all visible devices
    std::unordered_map<std::string, int> bus_id_to_device_idx;
    bus_id_to_device_idx.reserve(num_devices);
    std::vector<std::string> bus_ids;
    bus_ids.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      auto bus_id = get_bus_id(i);
      bus_id_to_device_idx.emplace(bus_id, i);
      bus_ids.push_back(std::move(bus_id));
    }

    static constexpr const char* warning_msg = "PyTorch features that use BLinkDetector may assume no BLink presence.";

    auto* driver_api = c10::supa::DriverAPI::get();
    if (driver_api->brmlInit_v2_() != BRML_SUCCESS) {
      LOG(WARNING) << "BLinkDetector: Failed to initialize BRML via brmlInit_v2. " << warning_msg;
      return c10::make_intrusive<c10d::DMAConnectivity>(c10::DeviceType::PrivateUse1, "blink", std::move(matrix));
    }

    // Obtain the brml device for all bus_ids
    std::vector<brmlDevice_t> brml_devices(num_devices, nullptr);
    for (int i = 0; i < num_devices; ++i) {
      auto res = driver_api->brmlDeviceGetHandleByPciBusId_v2_(bus_ids[i].c_str(), &brml_devices[i]);
      if (res != BRML_SUCCESS) {
        LOG(WARNING) << "BLinkDetector: Failed to obtain BRML device via "
                     << "brmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
        return c10::make_intrusive<c10d::DMAConnectivity>(c10::DeviceType::PrivateUse1, "blink", std::move(matrix));
      }
    }

    std::vector<int> switch_link_count(num_devices, 0);
    for (int i = 0; i < num_devices; ++i) {
      for (int link = 0; link < max_blinks; ++link) {
        brmlIntBLinkDeviceType_t deviceType{};
        auto ret = driver_api->brmlDeviceGetBLinkRemoteDeviceType_(brml_devices[i], link, &deviceType);
        if (ret != BRML_SUCCESS) {
          // We've exhausted the BLinks connected to this device. This error
          // is benign. There doesn't seem to be a reliable way to obtain the
          // maximum link value that can be passed to the API. Therefore, we
          // simply increment the link value until the API fails or we reach a
          // predefined maximum value.
          break;
        }
        // Remote device is GPU
        if (deviceType == BRML_BLINK_DEVICE_TYPE_GPU) {
          brmlPciInfo_t pciInfo;
          auto res = driver_api->brmlDeviceGetBLinkRemotePciInfo_v2_(brml_devices[i], link, &pciInfo);
          if (res != BRML_SUCCESS) {
            LOG(WARNING) << "BLinkDetector: Failed to obtain BRML device via "
                         << "brmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
            return c10::make_intrusive<c10d::DMAConnectivity>(c10::DeviceType::PrivateUse1, "blink", std::move(matrix));
          }
          auto it = bus_id_to_device_idx.find(pciInfo.busId);
          if (it != bus_id_to_device_idx.end()) {
            if (i != it->second) {
              matrix[i][it->second] += 1;
            }
          }
          // Remote device is NVSwitch
        } else if (deviceType == BRML_BLINK_DEVICE_TYPE_SWITCH) {
          switch_link_count[i] += 1;
        }
      }
    }

    // Process NVSwitch connections.
    // For simplicity, we assume that all NVSwitches are interconnected.
    for (int i = 0; i < num_devices; ++i) {
      for (int j = 0; j < num_devices; ++j) {
        if (i == j) {
          continue;
        }
        matrix[i][j] += std::min(switch_link_count[i], switch_link_count[j]);
      }
    }

    return c10::make_intrusive<c10d::DMAConnectivity>(c10::DeviceType::PrivateUse1, "blink", std::move(matrix));
  }
};

struct RegisterDetector {
  RegisterDetector() {
    register_dma_connectivity_detector(c10::DeviceType::PrivateUse1, "blink", c10::make_intrusive<BLinkDetector>());
  }
};

RegisterDetector register_detector_;

} // namespace
#endif
