---
name: torch-base-framework
description: "Quickly understand the torch-supa/torch_supa base framework: a PyTorch PrivateUse1 plugin backend that registers SUPA, initializes torch_supa, wires C++ bindings, dispatch/codegen, operator implementations, kernels, and build outputs. Use when users mention privateuse1, plugin framework, torch_supa initialization, SUPA backend, framework entry points, dispatch registration, or base framework overview."
depends_on: []
---

# torch-supa Base Framework Overview

This skill builds a quick mental model for this repo. `torch_supa` is a PyTorch plugin-style backend that uses `PrivateUse1` as the dispatcher backend and renames it to `supa`. Python code handles autoloading, backend naming, device module registration, and monkey patches. C++ code handles the `_C` extension, runtime bindings, allocator/stream/event abstractions, ATen dispatch registration, operator implementations, and kernel library linkage.

## When to use

Use this skill when the user asks questions like:

- How does this framework plug into PyTorch / PrivateUse1?
- What is the `torch_supa` initialization flow?
- Where is the `supa` backend registered?
- What is the high-level path from `import torch` / `import torch_supa` to operator execution?
- How do `libtorch_supa`, `libtorch_supa_op`, `libtorch_suda_op`, and `c10_supa` relate?
- Which files should I read first to understand the repo?

If the question is about the detailed call stack of a specific operator, use `foundations/torch-base-operator/SKILL.md` for operator call-stack analysis.

## Fast reading order

Read these files first, in this order:

1. `README.md`
   - Project positioning: BRPytorch Plugin lets models run on Biren devices with little or no SUPA-specific code.
   - User-facing entry: importing `torch` can autoload the plugin; devices may appear through `cuda` compatibility or explicit `supa` device usage.

2. `torch_supa/__init__.py`
   - Main Python initialization entry.
   - Key actions:
     - Temporarily disables `TORCH_DEVICE_BACKEND_AUTOLOAD` to avoid circular dependencies.
     - Calls `torch.utils.rename_privateuse1_backend("supa")` to rename PrivateUse1 to `supa`.
     - Calls `torch._register_device_module("supa", torch_supa.supa)` to register the Python device module.
     - Calls `torch.utils.generate_methods_for_privateuse1_backend(...)` to generate tensor/module/storage helper methods.
     - Calls `torch_supa._C._initExtension()` to initialize the C++ extension.
     - Applies profiler, distributed, dynamo, inductor, dlpack, and other integration patches.

3. `torch_supa/csrc/InitSupaBindings.cpp`
   - C++ Python extension entry for `torch_supa._C`.
   - Registers stream/event/mempool/graph bindings, device properties, allocator, runtime APIs, debug/profiler/distributed/transfer bindings.

4. `torch_supa/csrc/aten/supa_native_functions.yaml`
   - Core list for PyTorch native operator codegen integration.
   - `backend: SUPA`, `cpp_namespace: at::supa`.
   - `supported` lists operators registered for the PrivateUse1/SUPA device layer.
   - `autograd` lists operators registered for custom AutogradPrivateUse1 behavior.
   - `custom` defines internal/debug/fused API schemas.

5. `torch_supa/csrc/aten/generated/RegisterSUPANative.cpp`
   - Codegen-generated PrivateUse1 wrappers and `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` registrations.
   - Normal operator wrappers usually call `at::supa::SUPANativeFunctions::<op>`.
   - Structured operator wrappers construct `at::supa::structured_<op>`, run `meta(...)`, then call `impl_supa(...)`.

6. `torch_supa/csrc/aten/RegisterAutogradSUPA.cpp`
   - AutogradPrivateUse1 registration entry.
   - Used for operators that need custom backward or special autograd behavior.

7. `torch_supa/csrc/aten/ops/*.cpp`
   - Hand-written operator implementations.
   - Common namespace: `at::supa`.
   - Normal operators implement `SUPANativeFunctions::<op>`.
   - Structured operators usually implement `SUPA_IMPL_FUNC(op)`.

8. `torch_supa/csrc/aten/ops/kernels/`
   - Lower-level `.su` / `.cu` kernels and kernel helpers.
   - Elementwise logic is usually under `ops/kernels/elementwise/`.

9. `CMakeLists.txt` and `torch_supa/csrc/aten/CMakeLists.txt`
   - Build graph, source grouping, and library splitting.

## Layered mental model

### 1. Python autoload layer

Main file: `torch_supa/__init__.py`

Responsibilities:

- Rename PyTorch `PrivateUse1` backend to `supa`.
- Register `torch_supa.supa` as the Python device module.
- Generate PyTorch PrivateUse1 helper methods.
- Load the C++ extension `_C`.
- Patch profiler, distributed, dynamo, inductor, dlpack, and transfer-to-supa integrations into the PyTorch ecosystem.

Important symbols to inspect:

- `rename_privateuse1_backend("supa")`
- `_register_device_module("supa", torch_supa.supa)`
- `generate_methods_for_privateuse1_backend(...)`
- `torch_supa._C._initExtension()`
- `_autoload()`

### 2. C++ extension / runtime binding layer

Main file: `torch_supa/csrc/InitSupaBindings.cpp`

Responsibilities:

- Defines the Python module `torch_supa._C`.
- Registers SUPA stream/event/mempool/graph objects.
- Registers SUPA device properties, pluggable allocator, runtime API, and debug functions.
- Attaches profiler, distributed, transfer, and related extension functions.

Common follow-up locations:

- `torch_supa/csrc/supa/`: stream/event/graph/module/memory runtime wrappers.
- `torch_supa/csrc/core/` and `torch_supa/csrc/core/supa/`: storage, allocator, guard, driver API, and memory pool logic.
- `torch_supa/supa/*.py`: Python-side `torch.supa` style APIs.

### 3. ATen / dispatcher registration layer

Main files:

- `torch_supa/csrc/aten/supa_native_functions.yaml`
- `torch_supa/csrc/aten/generated/RegisterSUPANative.cpp`
- `torch_supa/csrc/aten/RegisterSUPA.cpp`
- `torch_supa/csrc/aten/RegisterAutogradSUPA.cpp`
- `torch_supa/csrc/aten/core/SUPANativeFunctions.h`
- `torch_supa/csrc/aten/core/SUPAStructuredFunctions.h`

Responsibilities:

- Map PyTorch ATen schemas to `PrivateUse1` / `AutogradPrivateUse1` dispatch keys.
- Generate wrappers, declarations, and registrations from YAML.
- Let wrappers handle common logic such as profiling, device guards, structured output allocation, and proxy outputs.
- Keep hand-written operator implementations focused on the operator-specific behavior.

Typical high-level call path:

```text
Python torch op
  -> PyTorch dispatcher
  -> AutogradPrivateUse1, if custom autograd is registered
  -> PrivateUse1
  -> generated wrapper
  -> at::supa::SUPANativeFunctions::<op> or structured_<op>::impl_supa
  -> native / sudnn / sublas / custom kernel
```

### 4. Operator implementation layer

Main directory: `torch_supa/csrc/aten/ops/`

Common implementation patterns:

- Reuse PyTorch native implementation, for example by calling `at::native_*`.
- Reuse backend libraries such as sudnn, sublas, or sufft.
- Use custom kernels through C++ launch wrappers in `ops/kernels/*.su` or `.cu`.
- Use mixed backend selection based on dtype, shape, training mode, layout, or capability checks.

Example:

- `torch_supa/csrc/aten/ops/BatchNorm.cpp`
  - `_batch_norm_impl_index` selects `Sudnn` or `Native` based on input conditions.
  - The Sudnn path calls `at::cudnn_batch_norm(...)`.
  - The Native path calls `at::native_batch_norm(...)`.
  - Backward uses `impl_index` to choose the matching backend.

### 5. Build and library split layer

Main files:

- Top-level `CMakeLists.txt`
- `torch_supa/csrc/aten/CMakeLists.txt`
- `setup.py`

Main outputs:

- `torch_supa` / `libtorch_supa.so`
  - Main Python extension library containing most C++ glue, registration, and runtime bindings.
- `c10_supa`
  - SUPA core library.
- `torch_supa_op` / `libtorch_supa_op.so`
  - SUPA `.su` kernel library.
- `torch_suda_op` / `libtorch_suda_op.so`
  - SUDA/CUDA-like `.cu` kernel library.
- Optional external `TORCH_SUPA_DIR`
  - When defined, links an existing `libtorch_supa.so` and includes generated registration sources in `torch_supa`.

Important source globs in `torch_supa/csrc/aten/CMakeLists.txt`:

- `*.cpp`, `common/*.cpp`, `ops/*.cpp`, `sudnn/*.cpp`, `ops/*/*.cpp` -> ATen C++ sources.
- `generated/*.cpp` -> generated registration sources.
- `ops/kernels/*.su` -> SUPA kernels.
- `ops/kernels/*.cu`, `ops/kernels/reduce/*.cu`, `ops/kernels/elementwise/*.cu` -> SUDA kernels.

## Standard response template

When the user asks for a quick framework overview, answer with:

1. One-line positioning: this is a PyTorch PrivateUse1 plugin backend that renames `PrivateUse1` to `supa`.
2. Python initialization entry: `torch_supa/__init__.py`.
3. C++ extension entry: `torch_supa/csrc/InitSupaBindings.cpp`.
4. Dispatch/codegen entry: `torch_supa/csrc/aten/supa_native_functions.yaml` plus `generated/RegisterSUPANative.cpp`.
5. Operator implementation entry: `torch_supa/csrc/aten/ops/*.cpp` plus `ops/kernels/`.
6. Build entry: `CMakeLists.txt`, `torch_supa/csrc/aten/CMakeLists.txt`, and `setup.py`.
7. If the user needs a specific operator call stack, route to `foundations/torch-base-operator/SKILL.md`.

## Debugging / reading recipes

### To trace how a Python API reaches SUPA

1. Confirm whether the API is an ATen operator.
2. Search the operator name in `supa_native_functions.yaml`.
3. Search the wrapper in `generated/RegisterSUPANative.cpp`.
4. Check declarations in `SUPANativeFunctions.h` or `SUPAStructuredFunctions.h`.
5. Search the implementation in `ops/*.cpp`.
6. If the implementation launches a kernel, continue into `ops/kernels/`.

### To understand why `.cuda()` can route to SUPA

Read:

- `README.md` getting-started section.
- `_autoload()` in `torch_supa/__init__.py`.
- `torch_supa/contrib/transfer_to_supa.py`.
- `torch_supa/csrc/utils/TransferSupa.cpp`.

### To understand where a new operator should be added

Prefer the existing skill:

- `new-operator/torch-create-new-op/SKILL.md`

First classify the work:

- Register an existing ATen operator in `supa_native_functions.yaml`.
- Add a custom schema / custom op.
- Implement a normal operator or a structured operator.
- Reuse native/sudnn/sublas, or add a new `.su` / `.cu` kernel.

## Important cautions

- Do not confuse `PrivateUse1`, `AutogradPrivateUse1`, and the Python device module. They are dispatcher backend key, autograd key, and Python API surface respectively.
- `supa` is the device name after renaming `PrivateUse1`; the C++ dispatch key is still `PrivateUse1`.
- Generated files are usually not the hand-written source of truth. Operator integration usually starts from YAML and `ops/*.cpp`.
- For structured operators, first identify the real PyTorch schema entry. Many functional variants delegate to `.out` variants.
- Whether a `.su` / `.cu` kernel is compiled into a given library depends on CMake globs and top-level library linkage.
