---
name: torch-elementwise-adaptation
description: Migrates CUDA elementwise kernels from the torch-supa build directory to torch_supa SUPA / PrivateUse1 implementations. Use when users provide src_file, kernel_entry, CUDA elementwise kernel migration, or ask for minimal CUDA to SUPA adaptation.
depends_on: []
---

# Elementwise Kernel Adaptation Skill

This skill guides migration of CUDA elementwise kernels from the following directory to the corresponding `torch_supa` implementation:

- Source directory: `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/`
- Target directory: `torch_supa/csrc/aten/ops/kernels/elementwise/`

Applicable scenarios:
- The user has provided a source file path and wants to migrate it directly.
- The user only knows the kernel entry name and wants to locate the source file before migration.
- The user wants to batch-migrate similar kernels following the existing elementwise kernel pattern.

If the user's request is not “migrate a kernel” but instead asks to:
- Decide whether the request is operator integration, existing feature extension, or new feature development.
- Extend existing elementwise capabilities such as dtype combinations, broadcast patterns, hybrid vectorized, static cast, or ldcg.
- Understand where to modify an elementwise new feature and which dispatch branch to attach it to.

Then route first to:

- `performance/torch-elementwise-feature/SKILL.md`

## Preconditions

When executing this skill, at least one of the following inputs is required:

- `src_file`: CUDA source file path.
- `kernel_entry`: kernel entry function name or dispatch entry name, for example `mul_kernel_cuda`.

If both are provided, first verify that the entry name actually exists in the given file.

If neither is provided, do not guess the file; ask the user to provide one of them.

## Execution principles

During migration, follow these principles:

1. Locate the source file first.
2. Consult `elementwise_cuda_to_supa_mapping.md` first.
3. Prefer reusing the existing migration pattern from `BinaryMulKernel.cu`.
4. Make only the minimal necessary replacements: headers, CUDA to SUPA mapping replacements, dispatch registration replacements, and necessary jit/jiterator branch handling.
5. Do not modify the main source logic. Do not use migration as a reason to rewrite the kernel structure, functor semantics, dispatch organization, or computation flow.
6. After completion, run a build or minimal validation.

## Source file lookup

### Case 1: User provides `src_file`

Read that file directly.

If `kernel_entry` is also provided, verify that the entry actually appears in that file.

### Case 2: User provides only `kernel_entry`

Search in this directory:

- `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/`

Prefer matching these patterns:

- `void <kernel_entry>(...)`
- `REGISTER_DISPATCH(..., &<kernel_entry>)`
- `REGISTER_DISPATCH(..., <kernel_entry>)`

After locating the source file, continue with the migration flow.

## Target file selection rules

Usually migrate to a file with the same name:

- Source file: `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/<Name>.cu`
- Target file: `torch_supa/csrc/aten/ops/kernels/elementwise/<Name>.cu`

If a same-name target file already exists in `torch_supa`:

- Compare the existing implementation with the source file first.
- Modify the existing file incrementally.
- Do not overwrite the whole file directly.

## Verified migration template: `BinaryMulKernel`

### Reference files

- Source file: `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/BinaryMulKernel.cu`
- Target file: `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryMulKernel.cu`

### Typical replacement patterns

#### 1. Header replacement

```cpp
// CUDA
#include <ATen/native/cuda/BinaryInternal.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

// SUPA / torch_supa
#include "torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h"
#include <torch_supa/csrc/core/supa/SUPAGuard.h>
#include "torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh"
```

Notes:
- `BinaryInternal.h` uses this repository's implementation.
- `CUDAGuard.h` is replaced by `SUPAGuard.h`.
- Loops uniformly use this repository's `Loops.cuh`.
- `JitLoops.cuh` has been removed in the current `BinaryMulKernel` migration implementation.

#### 2. Dispatch registration replacement

```cpp
// CUDA
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda)

// SUPA
REGISTER_PRIVATEUSE1_DISPATCH(mul_stub, &mul_kernel_cuda)
```

#### 3. namespace / guard / stream replacement

Common replacements should first follow `elementwise_cuda_to_supa_mapping.md`. Current known common replacements include:

- `c10::cuda` → `c10::supa`
- `at::cuda` → `at::supa`
- `OptionalDeviceGuard` → `c10::supa::OptionalSUPAGuard`
- `C10_CUDA_KERNEL_LAUNCH_CHECK()` → `C10_SUPA_KERNEL_LAUNCH_CHECK()`

#### 4. jit / jiterator branch handling

The existing `torch_supa` version of `BinaryMulKernel.cu` has removed the `AT_USE_JITERATOR()` branch and keeps only the non-jiterator path:

```cpp
using opmath_t = at::opmath_type<scalar_t>;
opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
    iter, binary_internal::MulFunctor<opmath_t>());
```

When migrating other elementwise kernels:

- If this repository's `Loops.cuh` or `CUDALoops.cuh` already covers the required capability, prefer reusing the existing implementation.
- If the CUDA version strongly depends on jiterator, first check whether `torch_supa` already has an alternative path, then decide whether to keep, trim, or rewrite the corresponding branch.
- Only jit/jiterator-related branches may be handled here. Do not use this process to rewrite the main compute logic.

## Recommended migration flow

### Step 1: Keep the source file read-only

Do not modify or overwrite the source CUDA file, and do not create `.bak` backup files next to upstream/build directories such as `third-party/torch-supa/build/_deps/`.

If a source snapshot is needed during migration, record only the source file path, entry name, and necessary excerpts in the current task notes or generated job artifact. The final diff should contain only target `torch_supa` files and necessary mapping-document updates.

### Step 2: Read the necessary files and confirm differences

Inspect at least the following together:

- Source CUDA file.
- Same-name target file under `torch_supa` if it exists.
- `torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`.
- `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`.
- `torch_supa/csrc/aten/ops/kernels/elementwise/static_binary_kernel_implementation.md`.

Focus on:
- Entry function name.
- Dispatch stub name.
- Functors/helpers used.
- Whether it depends on jiterator, streams, guards, launch checks, or a specific device type.

### Step 3: Apply only the minimal necessary replacements

During migration, modify only the parts directly related to backend switching:

- Include paths.
- `cuda` → `supa` namespaces or APIs.
- `REGISTER_DISPATCH` → `REGISTER_PRIVATEUSE1_DISPATCH`.
- Necessary jit/jiterator branch handling.
- Adjust loops/helper interfaces only when direct reuse is genuinely impossible.

Strictly do not rewrite the main source logic during migration, including but not limited to:

- Rewriting the kernel body.
- Reorganizing the dispatch structure.
- Changing functor semantics.
- Changing the computation flow.
- Replacing the source implementation with a different style merely to “reuse this repository's implementation”.

Do not perform unrelated refactoring opportunistically.

### Step 4: Check elementwise-specific mapping rules first

First check the main mapping document:

- `elementwise_cuda_to_supa_mapping.md`

If a new stable replacement rule is found during migration, such as:

- New header mapping.
- New macro mapping.
- New guard/stream API correspondence.
- New device type, stream, or launch-check replacement pattern.

Handle it in this order:

1. Update `elementwise_cuda_to_supa_mapping.md` first.
2. Apply the new rule to the current migration file.

### Step 5: Build and diagnose issues

Validation should follow the “small before large” principle.

#### Priority 1: Use the command explicitly provided by the user

If the user provides a build or validation command:
- Use the user-provided command exactly.
- Do not replace it with another command without permission.
- If it fails, continue debugging around the first real blocking error.

#### Priority 2: Use single-file or incremental compilation

First check whether an incremental build command can be reused from:

- `build.ninja` or CMake targets in the current build directory.
- The compile command corresponding to an existing object file.

If a single-file compile command can be extracted, prefer using it to validate the target `.cu` file.

#### Priority 3: Default validation approach

If the user did not provide a command, then consider default build entries:

- Root repository: `build.sh`
- torch-supa: `third-party/torch-supa/build.sh`

Notes:
- Root `build.sh` actually runs `python3 setup.py bdist_wheel`.
- `third-party/torch-supa/build.sh` actually runs `python3 -m build --wheel --no-isolation`.
- Both approaches are heavy and should be used only when no smaller-grained validation path is available.

### Compile-error triage order

When compilation fails, investigate in this order:

1. Whether CUDA-specific headers remain.
2. Whether all `c10::cuda` / `at::cuda` names have been replaced with `supa`.
3. Whether `REGISTER_DISPATCH` has been replaced with `REGISTER_PRIVATEUSE1_DISPATCH`.
4. Whether `AT_USE_JITERATOR()` or `jiterator_stringify` remains.
5. Whether helpers in `Loops.cuh` / `CUDALoops.cuh` match the current functor usage.

## Recommended reference files

- `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryMulKernel.cu`
- `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/BinaryMulKernel.cu`
- `torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`
- `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`
- `torch_supa/csrc/aten/ops/kernels/elementwise/static_binary_kernel_implementation.md`
- `elementwise_cuda_to_supa_mapping.md`
- `new-operator/torch-create-new-op/SKILL.md`
- `register_new_operator.md`

## Final output requirements

After executing this skill, the final response should include at least:

1. The actual source file path that was located.
2. The corresponding target file path.
3. Whether the source file stayed read-only; if a source snapshot was saved, state the artifact path.
4. Which CUDA → SUPA replacements were actually completed.
5. Whether the mapping document was updated.
6. Which build or validation command was used.
7. Whether the build passed; if it failed, provide the first real blocking error.
