/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPADeviceAssertionHost.h"
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <supa_runtime.h>
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/utils/Utils.h"

#include <memory>
#include <string>
#ifdef TORCH_USE_SUPA_DSA
#include <chrono>
#include <thread>
#endif

#define C10_SUPA_CHECK_WO_DSA(EXPR)                                 \
  do {                                                              \
    const supaError_t __err = EXPR;                                 \
    c10::supa::c10_supa_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        false);                                                     \
  } while (0)

namespace c10::supa {

namespace {

#ifdef TORCH_USE_SUPA_DSA
/// Get current device id
/// We need our own implementation of this function to prevent
/// an infinite initialization loop for SUPAKernelLaunchRegistry
int dsa_get_device_id() {
  c10::DeviceIndex device = -1;
  C10_SUPA_CHECK_WO_DSA(c10::supa::GetDevice(&device));
  return device;
}

/// Get a device's compute capability - note that this dangerously assumes
/// that if one SUPA GPU supports device-side assertions they all do. This is
/// probably fine since the latest SUPA GPU that doesn't support UVM is the
/// K80 released 2014-11-17. Mixing that GPU with a newer one is likely to be
/// rare enough that the defensive
/// We need our own implementation of this function to prevent
/// an infinite initialization loop for SUPAKernelLaunchRegistry
int dsa_get_device_compute_capability(const int device_num) {
  // E01643: simulate h100 for supa device, we could not get correct attribute from supa api
  int compute_capability = 9;
  return compute_capability;
}
#endif

/// Get the number of SUPA devices
/// We need our own implementation of this function to prevent
/// an infinite initialization loop for SUPAKernelLaunchRegistry
int dsa_get_device_count() {
  int device_count = -1;
  C10_SUPA_CHECK_WO_DSA(c10::supa::GetDeviceCount(&device_count));
  return device_count;
}

bool dsa_check_if_all_devices_support_managed_memory() {
#ifdef TORCH_USE_SUPA_DSA
  for (const auto i : c10::irange(dsa_get_device_count())) {
    if (dsa_get_device_compute_capability(i) < 6) {
      return false;
    }
  }
  return true;
#else
  return false;
#endif
}

bool env_flag_set(const char* env_var_name) {
  const auto env_flag = c10::utils::check_env(env_var_name);
  return env_flag.has_value() && env_flag.value();
}

/// Deleter for UVM/managed memory pointers
void uvm_deleter(DeviceAssertionsData* uvm_assertions_ptr) {
  // Ignore error in destructor
  if (uvm_assertions_ptr) {
    C10_SUPA_IGNORE_ERROR(supaFree(uvm_assertions_ptr));
  }
}

} // namespace

/// Check that kernels ran correctly by checking the message buffer. BLOCKING.
std::string c10_retrieve_device_side_assertion_info() {
#ifdef TORCH_USE_SUPA_DSA
  const auto& launch_registry = SUPAKernelLaunchRegistry::get_singleton_ref();
  if (!launch_registry.enabled_at_runtime) {
    return "Device-side assertion tracking was not enabled by user.";
  }
  if (!launch_registry.do_all_devices_support_managed_memory) {
    return "Device-side assertions disabled because not all devices support "
           "managed memory.";
  }

  // Hack that saves a lot of challenging sync logic.
  // The GPU increments the number of errors it's observed and the CPU can see
  // that happening immediately which means we can make it here before the GPU
  // is done writing information about those errors to memory.
  // A short pause gives it time to finish. Since something's gone wrong, this
  // pause shouldn't affect perf.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // The snapshot causes a brief block. That's okay because this function only
  // executes if something's gone wrong such that speed is no longer a priority.
  const auto launch_data = launch_registry.snapshot();
  const auto& assertion_data = launch_data.first;
  const auto& launch_infos = launch_data.second;

  std::stringstream oss;

  oss << "Looking for device-side assertion failure information...\n";

  // Loop over each device that could be managed by the process
  for (const auto device_num : c10::irange(assertion_data.size())) {
    const auto& assertion_data_for_device = assertion_data.at(device_num);

    // Did anything fail?
    const auto failures_found = std::min(assertion_data_for_device.assertion_count, C10_SUPA_DSA_ASSERTION_COUNT);
    if (failures_found == 0) {
      continue;
    }

    // Something failed, let's talk about that
    oss << failures_found << " SUPA device-side assertion failures were found on GPU #" << device_num << "!"
        << std::endl;
    if (assertion_data_for_device.assertion_count > C10_SUPA_DSA_ASSERTION_COUNT) {
      oss << "But at least " << assertion_data_for_device.assertion_count
          << " assertion failures occurred on the device" << std::endl;
      oss << "Adjust `C10_SUPA_DSA_ASSERTION_COUNT` if you need more assertion failure info" << std::endl;
    }

    for (const auto i : c10::irange(failures_found)) {
      const auto& self = *(assertion_data_for_device.assertions + i);
      const auto& launch_info = launch_infos[self.caller % launch_infos.size()];
      oss << "Assertion failure " << i << std::endl;
      oss << "  GPU assertion failure message = " << self.assertion_msg << std::endl;
      oss << "  File containing assertion = " << self.filename << ":" << self.line_number << std::endl;
      oss << "  Device function containing assertion = " << self.function_name << std::endl;
      oss << "  Thread ID that failed assertion = [" << self.thread_id[0] << "," << self.thread_id[1] << ","
          << self.thread_id[2] << "]" << std::endl;
      oss << "  Block ID that failed assertion = [" << self.block_id[0] << "," << self.block_id[1] << ","
          << self.block_id[2] << "]" << std::endl;
      if (launch_info.generation_number == self.caller) {
        oss << "  File containing kernel launch = " << launch_info.launch_filename << ":" << launch_info.launch_linenum
            << std::endl;
        oss << "  Function containing kernel launch = " << launch_info.launch_function << std::endl;
        oss << "  Name of kernel launched that led to failure = " << launch_info.kernel_name << std::endl;
        oss << "  Device that launched kernel = " << launch_info.device << std::endl;
        oss << "  Stream kernel was launched on = " << launch_info.stream << std::endl;
        oss << "  Backtrace of kernel launch site = ";
        if (launch_registry.gather_launch_stacktrace) {
          oss << "Launch stacktracing disabled." << std::endl;
        } else {
          oss << "\n" << launch_info.launch_stacktrace << std::endl;
        }
      } else {
        oss << "  CPU launch site info: Unavailable, the circular queue wrapped around. Increase `SUPAKernelLaunchRegistry::max_size`."
            << std::endl;
      }
    }
  }
  return oss.str();
#else
  return "Compile with `TORCH_USE_SUPA_DSA` to enable device-side assertions.\n";
#endif
}

SUPAKernelLaunchRegistry::SUPAKernelLaunchRegistry()
    : do_all_devices_support_managed_memory(dsa_check_if_all_devices_support_managed_memory()),
      gather_launch_stacktrace(check_env_for_enable_launch_stacktracing()),
      enabled_at_runtime(check_env_for_dsa_enabled()) {
  for ([[maybe_unused]] const auto _ : c10::irange(dsa_get_device_count())) {
    uvm_assertions.emplace_back(nullptr, uvm_deleter);
  }

  kernel_launches.resize(max_kernel_launches);
}

bool SUPAKernelLaunchRegistry::check_env_for_enable_launch_stacktracing() {
  return env_flag_set("PYTORCH_SUPA_DSA_STACKTRACING") || env_flag_set("PYTORCH_CUDA_DSA_STACKTRACING");
}

bool SUPAKernelLaunchRegistry::check_env_for_dsa_enabled() {
  return env_flag_set("PYTORCH_USE_SUPA_DSA") || env_flag_set("PYTORCH_USE_CUDA_DSA");
}

uint32_t SUPAKernelLaunchRegistry::insert(
    const char* launch_filename [[maybe_unused]],
    const char* launch_function [[maybe_unused]],
    const uint32_t launch_linenum [[maybe_unused]],
    const char* kernel_name [[maybe_unused]],
    const int32_t stream_id [[maybe_unused]]) {
#ifdef TORCH_USE_SUPA_DSA
  if (!enabled_at_runtime) {
    return 0;
  }

  const auto backtrace = gather_launch_stacktrace ? c10::get_backtrace() : "";

  const std::lock_guard<std::mutex> lock(read_write_mutex);

  const auto my_gen_number = generation_number++;
  // TODO: It would probably be good to get a stack trace here so that
  // we can better indicate which launch caused the failure.
  kernel_launches[my_gen_number % max_kernel_launches] = {
      launch_filename,
      launch_function,
      launch_linenum,
      backtrace,
      kernel_name,
      dsa_get_device_id(),
      stream_id,
      my_gen_number};
  return my_gen_number;
#else
  return 0;
#endif
}

std::pair<std::vector<DeviceAssertionsData>, std::vector<SUPAKernelLaunchInfo>> SUPAKernelLaunchRegistry::snapshot()
    const {
  // This is likely to be the longest-lasting hold on the mutex, but
  // we only expect it to be called in cases where we're already failing
  // and speed is no longer important
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  std::vector<DeviceAssertionsData> device_assertions_data;
  for (const auto& x : uvm_assertions) {
    if (x) {
      device_assertions_data.push_back(*x);
    } else {
      device_assertions_data.emplace_back();
    }
  }

  return std::make_pair(device_assertions_data, kernel_launches);
}

DeviceAssertionsData* SUPAKernelLaunchRegistry::get_uvm_assertions_ptr_for_current_device() {
#ifdef TORCH_USE_SUPA_DSA
  if (!enabled_at_runtime) {
    return nullptr;
  }

  const auto device_num = dsa_get_device_id();

  // If we've already set up this GPU with managed memory, return a pointer to
  // the managed memory. This is a lock-free quick-return path.
  if (uvm_assertions.at(device_num)) {
    return uvm_assertions.at(device_num).get();
  }

  // Need a lock here so there's not race-condition on creating the new device
  // assertions buffer
  const std::lock_guard<std::mutex> lock(gpu_alloc_mutex);

  // If we've already set up this GPU with managed memory, return a pointer to
  // the managed memory. This locked path ensures that the device memory is
  // allocated only once
  if (uvm_assertions.at(device_num)) {
    return uvm_assertions.at(device_num).get();
  }

  // Otherwise, set up the GPU to be able to use the device-side assertion
  // system
  DeviceAssertionsData* uvm_assertions_ptr = nullptr;

  C10_SUPA_CHECK_WO_DSA(supaMallocManaged(&uvm_assertions_ptr, sizeof(DeviceAssertionsData)));

  C10_SUPA_CHECK_WO_DSA(supaMemAdvise(
      uvm_assertions_ptr, sizeof(DeviceAssertionsData), supaMemAdviseSetPreferredLocation, supaCpuDeviceId));

  // GPU will establish direct mapping of data in CPU memory, no page faults
  // will be generated
  C10_SUPA_CHECK_WO_DSA(
      supaMemAdvise(uvm_assertions_ptr, sizeof(DeviceAssertionsData), supaMemAdviseSetAccessedBy, supaCpuDeviceId));

  // Initialize the memory from the CPU; otherwise, pages may have to be created
  // on demand. We think that UVM documentation indicates that first access may
  // not honor preferred location, which would be bad, if true, because we want
  // this memory on the host so we can access it post-assertion. Initializing
  // this on the CPU helps ensure that that's where the memory will live.
  *uvm_assertions_ptr = DeviceAssertionsData();

  // Ownership and lifetime management of `uvm_assertions_ptr` now passes to the
  // uvm_assertions unique_ptr vector
  uvm_assertions.at(device_num).reset(uvm_assertions_ptr);

  return uvm_assertions_ptr;
#else
  return nullptr;
#endif
}

SUPAKernelLaunchRegistry& SUPAKernelLaunchRegistry::get_singleton_ref() {
  static SUPAKernelLaunchRegistry launch_registry;
  return launch_registry;
}

bool SUPAKernelLaunchRegistry::has_failed() const {
  for (const auto& x : uvm_assertions) {
    if (x && x->assertion_count > 0) {
      return true;
    }
  }
  return false;
}

} // namespace c10::supa
