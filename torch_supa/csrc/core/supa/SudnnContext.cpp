/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SudnnContext.h"
#include "torch_supa/csrc/core/supa/DeviceThreadHandles.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace at::supa {
namespace {

void createSuDNNHandle(sudnnHandle_t* handle) {
  AT_SUDNN_CHECK(sudnnCreate(handle));
}

void destroySuDNNHandle(sudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the cuda context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. @colesbury and I decided to not destroy
  // the handle as a workaround.
  //   - @soumith
  //
  // Further note: this is now disabled globally, because we are seeing
  // the same issue as mentioned above in CUDA 11 CI.
  //   - @zasdfgbnm
  //
  // #ifdef NO_sudnn_DESTROY_HANDLE
  // #else
  //   sudnnDestroy(handle);
  // #endif
}

using sudnnPoolType = at::supa::DeviceThreadHandlePool<sudnnHandle_t, createSuDNNHandle, destroySuDNNHandle>;

} // namespace

sudnnHandle_t getSudnnHandle() {
  c10::DeviceIndex device = 0;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<sudnnPoolType>();
  thread_local std::unique_ptr<sudnnPoolType::PoolWindow> myPoolWindow(pool->newPoolWindow());

  auto* handle = myPoolWindow->reserve(device);
  AT_SUDNN_CHECK(sudnnSetStream(handle, c10::supa::getCurrentSUPAStream()));
  return handle;
}

} // namespace at::supa
