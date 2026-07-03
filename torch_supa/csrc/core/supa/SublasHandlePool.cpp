/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/core/supa/SublasContext.h"
#include "torch_supa/csrc/utils/EnvConfig.h"

#include <ATen/Context.h>

#include "torch_supa/csrc/core/supa/DeviceThreadHandles.h"

#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>

namespace at::supa {

namespace {

void createSublasLtHandle(sublasLtHandle_t* handle) {
  AT_SUBLAS_CHECK(sublasLtCreate(handle));
}

void destroySublasLtHandle(sublasLtHandle_t handle) {
  AT_SUBLAS_CHECK(sublasLtDestroy(handle));
}

void createSublasHandle(sublasHandle_t* handle) {
  AT_SUBLAS_CHECK(sublasCreate(handle));
}

void destroySublasHandle(sublasHandle_t handle) {
  AT_SUBLAS_CHECK(sublasDestroy(handle));
}

using SuBlasLtPoolType = DeviceThreadHandlePool<sublasLtHandle_t, createSublasLtHandle, destroySublasLtHandle>;

using SuBlasPoolType = DeviceThreadHandlePool<sublasHandle_t, createSublasHandle, destroySublasHandle>;

} // anonymous namespace

std::map<std::tuple<void*, void*>, at::DataPtr>& sublas_handle_stream_to_workspace() {
  static auto& instance = *new std::map<std::tuple<void*, void*>, at::DataPtr>;
  return instance;
}

void clearSublasWorkspaces() {
  sublas_handle_stream_to_workspace().clear();
}

size_t parseChosenWorkspaceSize() {
  const char* val = getenv("SUBLAS_WORKSPACE_CONFIG");

  // supaDeviceProp* properties = at::supa::getCurrentDeviceProperties();
  // const bool sm90 = properties != nullptr && properties->major == 9 &&
  // properties->minor == 0; const size_t default_size = sm90 ? 4096 * 8 * 1024
  // : 4096 * 1024 * 2 + 16 * 1024 * 8;

  // TODO: maybe we should get a correct default sublas workspace size?
  const size_t default_size = 4096 * 1024 * 2 + 16 * 1024 * 8;

  if (val) {
    size_t total_size = 0;
    const std::string config(val);
    std::regex exp(":([0-9]+):([0-9]+)");
    std::sregex_iterator next(config.begin(), config.end(), exp);
    std::sregex_iterator end;
    if (next == end) {
      TORCH_WARN(
          "Could not parse SUBLAS_WORKSPACE_CONFIG, using default "
          "workspace size of ",
          default_size,
          " bytes.");
      return default_size;
    }
    while (next != end) {
      std::smatch match = *next;
      TORCH_CHECK(
          match.size() == 3,
          "Expected SUBLAS_WORKSPACE_SPACE_CONFIG "
          "match of size 3 (Format :SIZE:COUNT)");
      size_t curr_size = (size_t)std::stoi(match.str(1));
      size_t count = (size_t)std::stoi(match.str(2));
      total_size += curr_size * 1024 * count;
      next++;
    }
    return total_size;
  }
  return default_size;
}

size_t getChosenWorkspaceSize() {
  size_t pool_size = parseChosenWorkspaceSize();
  return pool_size;
}

at::DataPtr getNewWorkspace() {
  return c10::supa::SUPACachingAllocator::get()->allocate(getChosenWorkspaceSize());
}

sublasHandle_t getCurrentSuBlasHandle() {
  c10::DeviceIndex device = 0;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));

  static auto pool = std::shared_ptr<SuBlasPoolType>(new SuBlasPoolType(), [](SuBlasPoolType* p) {
    // Leak the memory.
  });
  thread_local std::unique_ptr<SuBlasPoolType::PoolWindow> myPoolWindow(pool->newPoolWindow());

  auto* handle = myPoolWindow->reserve(device);
  auto stream = c10::supa::getCurrentSUPAStream();
  AT_SUBLAS_CHECK(sublasSetStream(handle, stream));

  supaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void*>(handle), static_cast<void*>(_stream));
  auto workspace_it = sublas_handle_stream_to_workspace().find(key);
  if (workspace_it == sublas_handle_stream_to_workspace().end()) {
    workspace_it = sublas_handle_stream_to_workspace().insert(workspace_it, {key, getNewWorkspace()});
  }

  AT_SUBLAS_CHECK(sublasSetWorkspace(handle, workspace_it->second.get(), getChosenWorkspaceSize()));

  // To enable TF32, set the math mode of the handle to
  // SUBLAS_TF32_TENSOR_OP_MATH.
  if (!at::NoTF32Guard::should_disable_tf32() &&
#if TORCH_VER >= TORCH_2_10_0
      at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
          at::Float32Precision::TF32
#elif TORCH_VER >= TORCH_2_9_0
      at::globalContext().float32Precision("cuda", "matmul") == "tf32"
#else
      at::globalContext().allowTF32CuBLAS()
#endif
  ) {
    AT_SUBLAS_CHECK(sublasSetMathMode(handle, SUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    AT_SUBLAS_CHECK(sublasSetMathMode(handle, SUBLAS_DEFAULT_MATH));
  }
  return handle;
}

sublasLtHandle_t getCurrentSuBlasLtHandle() {
  return reinterpret_cast<sublasLtHandle_t>(getCurrentSuBlasHandle());
}

} // namespace at::supa
