# 算子和 PrivateUse1 适配

## 算子归属

新增或优化算子前，先判断算子归属。

| 算子类型 | 常见归属 |
| --- | --- |
| 原生 `torch.xxx` 算子，只需要少量源码 patch | `torch-supa` patch 区域；文档也提到 torchvision 算子由这里维护。 |
| 原生 `torch.xxx` 算子，需要 SUPA 特化重写或性能实现 | `br_pytorch2` / `torch_supa` native 实现流程。 |
| 自定义算子或第三方库算子 | 扩展仓库，例如 `torch_supa_ext`。 |

来源：`docs/sphinx/source/op_repo_registration.rst` 和 `docs/user_guider/source/chapter04_torch_supa_features.rst`。

## Native 算子注册流程

SUPA backend 中实现原生算子时，一般流程是：

1. 在 `supa_native_functions.yaml` 中加入算子。
2. 根据用途选择列表：
   - `supported`：注册到 `PrivateUse1` dispatch key；
   - `autograd`：注册到 `AutogradPrivateUse1`，用于自定义 backward；
   - `perf_supported`：已有功能实现，但可能需要性能覆盖的算子。
3. 在 native ops 目录下加入实现。
4. structured operator 使用 `SUPA_IMPL_FUNC` 复用 PyTorch meta function 路径。
5. autograd operator 需要实现 `torch::autograd::Function` 子类。
6. 必要时加入 SUPA kernel，并从 `SUPANativeFunctions` 调用。

最小 YAML 形态：

```yaml
backend: SUPA
cpp_namespace: at::supa
supported:
  - _addmm_activation.out
autograd:
  - renorm
perf_supported:
  - example_op
```

## Patch-native 算子流程

Patch-native 算子按支持的 PyTorch 版本维护源码 patch。用户指南提到的 patch 目录包括：

- `third-party/torch_supa_op/patch_privateuse1`
- `third-party/torch_supa_op/patch_torch`
- `third-party/torch_supa_op/patch_torch28`
- `third-party/torch_supa_op/patch_torch29`

排查 patch 行为时，需要确认期望 patch 是否真的应用到了 PyTorch source tree，并通过 stack trace 或 GDB 确认当前实际执行的实现。

## 开源库 PrivateUse1 适配

对于基于 CUDA 的 PyTorch 库，文档给出的适配思路是：

1. 使用 SUDA 编译 `.cu` 文件，并把 CUDA API 转发到 SUPA API。
2. 在 CMake include 顺序中，把 `br_pytorch2` / Torch SUPA 头文件放在 CPU PyTorch 头文件前面。
3. 必要时用 linker version script 隐藏冲突的 CUDA 符号。
4. 只针对硬件或编译器差异修改 CUDA 源码。

常见源码调整：

- 避免或减少 `double` 使用，因为 complex double 操作不完整或性能较差；
- 使用 `-ffloat-constants` 减少意外 double 常量；
- 当 BRCC 的 host/device 推导和 NVCC 不同时，显式增加 `__host__`；
- 替换 inline PTX，或在合适场景尝试 `-mira`；
- 根据壁仞硬件限制调整 block size 假设。

来源：`docs/sphinx/source/torch_privateuse1_adaptation.rst`。
