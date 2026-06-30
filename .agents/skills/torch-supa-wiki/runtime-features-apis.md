# 运行时能力和 API

## 兼容模型

`torch_supa` 使用 PyTorch `PrivateUse1` 集成来提供较高的原生 PyTorch API 兼容性。主要兼容行为包括：

- `cuda:0` 映射到 `supa:0`。
- 支持范围内的 `torch.cuda.*` 映射到 `torch.supa.*`。
- `tensor.cuda()` 和 `tensor.supa()` 都可以把 tensor 移到 SUPA 设备。
- 适用场景下，NCCL 风格 distributed API 会重定向到 BCCL 风格等价实现。

来源：`docs/user_guider/source/chapter05_torch_supa_api.rst`。

## 核心能力

用户指南列出的主要能力：

| 能力 | 摘要 |
| --- | --- |
| 自动加载和设备转换 | 加载插件，并把 CUDA device 用法映射到 SUPA。 |
| 算子适配 | 注册 native 算子，或集成自定义 C++ extension。 |
| Profiler | 使用 PyTorch profiler 捕获 SUPA activity。 |
| AMP | 支持 autocast 和混合精度训练。 |
| 开源库适配 | 使用 SUDA 和头文件/链接规则适配 CUDA 库。 |
| Graph mode | 支持 `torch.compile()` 和 SUPA Graph 加速路径。 |

来源：`docs/user_guider/source/chapter04_torch_supa_features.rst`。

## API 分组

`torch.supa` 覆盖常见 runtime API 分组：

- device 管理：可用性、设备数量、当前设备、设备属性、同步；
- memory 管理：分配统计、cache 控制、memory summary；
- stream 和 event 管理；
- graph API，包括 SUPA graph capture/replay 概念；
- AMP API，包括 `autocast` 和 `GradScaler` 兼容；
- 通过 BCCL 相关 API 兼容 distributed backend。

## AMP 示例

```python
import torch

a = torch.rand((8, 8), device="supa")
b = torch.rand((8, 8), device="supa")

with torch.autocast(device_type="supa"):
    out = torch.mm(a, b)
    print(out.dtype)
```

AMP 可提升性能，但每个模型仍需要验证收敛和数值精度。

## Profiler 示例

```python
import torch

x = torch.rand(2, 3, device="cuda")
y = torch.rand(2, 3, device="cuda")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.SUPA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    z = x + y

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

## Graph mode

`torch_supa` 支持配合 Inductor backend 使用 `torch.compile()`，并支持 SUPA graph 加速路径。指南说明 Inductor 使用需要 `br-triton` 依赖。

```python
compiled_model = torch.compile(model.supa(), backend="inductor")
output = compiled_model(input)
```
