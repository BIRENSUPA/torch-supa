# Elementwise CUDA → SUPA Mapping Reference

This file summarizes common CUDA → SUPA mappings for elementwise kernel migration, mainly for migrations between these paths:

- Source directory: `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/*.cu`
- Target directory: `torch_supa/csrc/aten/ops/kernels/elementwise/*.cu`

Before performing a concrete migration, check the existing rules in this file first. If a new stable replacement pattern is found, add it here first, then apply it to the target file.

## 0. Migration boundaries and constraints

When migrating elementwise kernels, follow these constraints:

- Prefer reusing the existing migration pattern in `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryMulKernel.cu`.
- Only make the minimal necessary changes: header replacement, CUDA → SUPA mapping replacement, dispatch registration replacement, and necessary jit / jiterator branch handling.
- By default, migration targets should support `BRTB_ENABLE_NATIVE_OP`: any migrated target that needs `PrivateUse1` registration through `REGISTER_PRIVATEUSE1_DISPATCH` should use the override version provided by `torch_supa/csrc/aten/ops/kernels/kernelDispatch.h`, so that registration side effects can be skipped and PyTorch native implementation can be used when `BRTB_ENABLE_NATIVE_OP=true`.
- Do not modify the main source logic. Do not use migration as a reason to rewrite the kernel body, functor semantics, dispatch structure, or computation flow.
- If repository helpers cannot directly support the source implementation, first record the difference and confirm the migration strategy instead of directly rewriting the source implementation into another style.

## 1. Function and API mappings

| CUDA API | SUPA API | Notes |
|----------|----------|-------|
| `at::cuda::getCurrentCUDAStream()` | `c10::supa::getCurrentSUPAStream()` | Get the current stream |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `C10_SUPA_KERNEL_LAUNCH_CHECK()` | Kernel launch check macro |
| `const OptionalDeviceGuard` | `c10::supa::OptionalSUPAGuard` | Optional device guard |
| `OptionalDeviceGuard` | `c10::supa::OptionalSUPAGuard` | Optional device guard |
| `DeviceGuard` | `c10::supa::SUPAGuard` | Device guard |
| `OptionalStreamGuard` | `c10::supa::OptionalSUPAStreamGuard` | Optional stream guard |
| `StreamGuard` | `c10::supa::SUPAStreamGuard` | Stream guard |
| `MultiStreamGuard` | `c10::supa::SUPAMultiStreamGuard` | Multi-stream guard |
| `CUDAStream` | `c10::supa::SUPAStream` | Stream type |
| `c10::cuda::getStreamFromPool()` | `c10::supa::getStreamFromPool()` | Get a stream from the pool |
| `c10::cuda::getDefaultCUDAStream()` | `c10::supa::getDefaultSUPAStream()` | Get the default stream |
| `c10::cuda::setCurrentCUDAStream()` | `c10::supa::setCurrentSUPAStream()` | Set the current stream |
| `at::cuda::getNumGPUs()` | `at::supa::getNumGPUs()` | Get the GPU count |
| `at::cuda::getCurrentDeviceProperties()` | `at::supa::getCurrentDeviceProperties()` | Get current device properties |
| `at::cuda::getDeviceProperties()` | `at::supa::getDeviceProperties()` | Get device properties |
| `at::cuda::get_p2p_access()` | `at::supa::get_p2p_access()` | Query P2P access capability |
| `at::cuda::memcpy_and_sync()` | `c10::supa::memcpy_and_sync()` | Synchronous memcpy wrapper |
| `at::getHostAllocator(at::kCUDA)->record_event(...)` | `at::supa::getCachingHostAllocator()->record_event(...)` | Record host allocator event |
| `at::cuda::warp_size()` | `at::supa::warp_size()` | Get warp size |

## 2. Header mappings

| CUDA header | SUPA / torch_supa header | Notes |
|-------------|------------------------|-------|
| `<c10/cuda/CUDAStream.h>` | `<torch_supa/csrc/core/supa/SUPAStream.h>` | Stream header replacement |
| `<c10/cuda/CUDAGuard.h>` | `<torch_supa/csrc/core/supa/SUPAGuard.h>` | Device and stream guard replacement |
| `<c10/cuda/CUDAException.h>` | `<torch_supa/csrc/core/supa/SUPAException.h>` | Exception header replacement |
| `<ATen/cuda/CUDAEvent.h>` | `<torch_supa/csrc/core/supa/SUPAEvent.h>` | Event header replacement |
| `<ATen/cuda/PeerToPeerAccess.h>` | `<torch_supa/csrc/core/supa/PeerToPeerAccess.h>` | P2P capability query header replacement |
| `<ATen/cuda/CachingHostAllocator.h>` | `<torch_supa/csrc/core/supa/CachingHostAllocator.h>` | Host allocator event recording replacement |
| `<ATen/native/cuda/BinaryInternal.h>` | `"torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h"` | Use this repository's Binary functor implementation |
| `<ATen/native/cuda/Loops.cuh>` | `"torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"` | Use this repository's elementwise loops implementation |
| `<ATen/native/cuda/JitLoops.cuh>` | `"torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"` | Current migration prefers this repository's loops; if jiterator support is missing, only handle the related jit / jiterator branch and do not change the main logic |

## 3. Macro and dispatch registration mappings

| CUDA | SUPA / torch_supa | Notes |
|------|------------------|-------|
| `C10_CUDA_CHECK()` | `C10_SUPA_CHECK()` | SUPA error check |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `C10_SUPA_KERNEL_LAUNCH_CHECK()` | Kernel launch check |
| `C10_CUDA_CHECK_WARN()` | `C10_SUPA_CHECK_WARN()` | SUPA warning check |
| `C10_CUDA_DRIVER_CHECK()` | `C10_SUPA_DRIVER_CHECK()` | Driver error check |
| `AT_CUDA_DRIVER_CHECK()` | `AT_SUPA_DRIVER_CHECK()` | Driver error check |
| `REGISTER_DISPATCH()` | `REGISTER_PRIVATEUSE1_DISPATCH()` | Switch dispatch registration to PrivateUse1 |
| `REGISTER_PRIVATEUSE1_DISPATCH()` | Override version in `kernelDispatch.h` | Override the original macro through `torch_supa/csrc/aten/ops/kernels/kernelDispatch.h`; when `BRTB_ENABLE_NATIVE_OP=true`, do not call `set_privateuse1_dispatch_ptr` |
| `AT_USE_JITERATOR()` | Trim or rewrite case by case | First confirm whether `torch_supa` loops or helpers already have an alternative implementation |

## 4. Symbol export reminder

- If a migrated symbol needs to be referenced by another compilation unit or shared library, explicitly add an export macro such as `TORCH_SUPA_API` or `C10_SUPA_API`.
- Do not rely only on a default function definition. Under the current visibility policy, symbols that are not explicitly exported may become local, causing link or runtime resolution failures.
- When modifying a definition, also check the corresponding declaration and ensure both declaration and definition use consistent export attributes.

## 5. Type mappings

| CUDA type | SUPA type |
|-----------|-----------|
| `cudaDeviceProp` | `supaDeviceProp` |
| `cudaStream_t` | `supaStream_t` |
| `cudaError_t` | `supaError_t` |

## 6. Device type mappings

| CUDA | SUPA |
|------|------|
| `kCUDA` | `kPrivateUse1` |
| `DeviceType::CUDA` | `DeviceType::PrivateUse1` |
| `is_cuda()` | `is_privateuseone()` |

## 7. Namespace and guard mappings

| CUDA | SUPA |
|------|------|
| `c10::cuda` | `c10::supa` |
| `at::cuda` | `at::supa` |
| `OptionalDeviceGuard` | `c10::supa::OptionalSUPAGuard` |

## 8. Usage examples

### Get the current stream

```cpp
// CUDA
auto stream = at::cuda::getCurrentCUDAStream();

// SUPA
auto stream = c10::supa::getCurrentSUPAStream();
```

### Kernel launch check

```cpp
// CUDA
my_kernel<<<grid, block, 0, stream>>>(args);
C10_CUDA_KERNEL_LAUNCH_CHECK();

// SUPA
my_kernel<<<grid, block, 0, stream>>>(args);
C10_SUPA_KERNEL_LAUNCH_CHECK();
```

### Device guard

```cpp
// CUDA
const OptionalDeviceGuard device_guard(device);

// SUPA
const c10::supa::OptionalSUPAGuard device_guard(device);
```

### Stream guard

```cpp
// CUDA
OptionalStreamGuard stream_guard(stream);

// SUPA
c10::supa::OptionalSUPAStreamGuard stream_guard(stream);
```

### Get device properties

```cpp
// CUDA
cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

// SUPA
supaDeviceProp* prop = at::supa::getCurrentDeviceProperties();
```

## 9. Verified example

The following example has verified typical replacement methods for this type of migration:

- Source file: `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/BinaryMulKernel.cu`
- Target file: `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryMulKernel.cu`

This example covers the following migration actions:

- `BinaryInternal` include replacement
- `CUDAGuard` → `SUPAGuard`
- `Loops` / `JitLoops` → this repository's `Loops.cuh`
- `REGISTER_DISPATCH` → `REGISTER_PRIVATEUSE1_DISPATCH`
- Remove the jiterator branch and keep the compilable path in this repository
