/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <sstream>

#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

namespace torch_supa::profiler::impl {

using torch::profiler::impl::ProfilerStubs;
using torch::profiler::impl::ProfilerVoidEventStub;
using ProfilerEventStub = std::shared_ptr<SUevent_st>;

namespace {

void supaCheck(supaError_t result, const char* file, int line) {
  if (result != supaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == supaErrorInitializationError) {
      // It is common for users to use DataLoader with multiple workers
      // and the autograd profiler. Throw a nice error message here.
      ss << "SUPA initialization error. "
         << "This can occur if one runs the profiler in SUPA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in SUPA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      ss << supaGetErrorString(result);
    }
    TORCH_CHECK(false, ss.str());
  }
}
#define TORCH_SUPA_CHECK(result) supaCheck(result, __FILE__, __LINE__);

struct SUPAMethods : public ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event, int64_t* cpu_ns) const override {
    if (device) {
      TORCH_SUPA_CHECK(c10::supa::GetDevice(device));
    }
    SUevent_st* supa_event_ptr{nullptr};
    TORCH_SUPA_CHECK(supaEventCreate(&supa_event_ptr));
    *event =
        std::shared_ptr<SUevent_st>(supa_event_ptr, [](SUevent_st* ptr) { TORCH_SUPA_CHECK(supaEventDestroy(ptr)); });
    auto stream = c10::supa::getCurrentSUPAStream();
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }
    TORCH_SUPA_CHECK(supaEventRecord(supa_event_ptr, stream));
  }

  float elapsed(const ProfilerVoidEventStub* event_, const ProfilerVoidEventStub* event2_) const override {
    const auto* event = (const ProfilerEventStub*)(event_);
    const auto* event2 = (const ProfilerEventStub*)(event2_);
    TORCH_SUPA_CHECK(supaEventSynchronize(event->get()));
    TORCH_SUPA_CHECK(supaEventSynchronize(event2->get()));
    float ms = 0;
    TORCH_SUPA_CHECK(supaEventElapsedTime(&ms, event->get(), event2->get()));
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions)
    return ms * 1000.0;
  }

  static void printUnavailableWarning() {
    TORCH_WARN_ONCE("Warning: roctracer isn't available on Windows");
  }
  void mark(const char* name) const override {
    printUnavailableWarning();
  }
  void rangePush(const char* name) const override {
    printUnavailableWarning();
  }
  void rangePop() const override {
    printUnavailableWarning();
  }

  void onEachDevice(std::function<void(int)> op) const override {
    c10::supa::OptionalSUPAGuard device_guard;
    for (const auto i : c10::irange(c10::supa::device_count())) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    TORCH_SUPA_CHECK(supaDeviceSynchronize());
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterSUPAMethods {
  RegisterSUPAMethods() {
    static SUPAMethods methods;
    torch::profiler::impl::registerPrivateUse1Methods(&methods);
  }
};
RegisterSUPAMethods reg;

} // namespace
} // namespace torch_supa::profiler::impl