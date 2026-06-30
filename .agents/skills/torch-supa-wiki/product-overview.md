# 产品概览

## Torch SUPA 提供什么

`torch_supa` 是壁仞面向 BR2xx GPU 的 PyTorch 插件。它通过 PyTorch 的 `PrivateUse1` 机制接入 SUPA backend，让大多数 PyTorch 代码在保持熟悉编程模型的同时，运行在壁仞硬件上，用于训练、推理和优化任务。

插件重点能力：

- 高兼容原生 PyTorch API；
- 自动完成 CUDA 到 SUPA 的设备映射；
- 提供 BR2xx 相关 runtime 和算子优化；
- 支持 profiler、AMP、graph mode、distributed backend 适配和扩展库工作流。

来源：`docs/user_guider/source/chapter01_torch_supa_overview.rst`。

## 版本和包名映射

wheel 命名规则：

```text
torch_supa-<torch_supa_version>.<torch_version>+br2xx-cp310-cp310-linux_x86_64.whl
```

用户指南中的示例：

| torch_supa 版本 | PyTorch 版本 |
| --- | --- |
| `1.0.0.20600` | `v2.6.0` |
| `1.0.0.20800` | `v2.8.0` |
| `1.0.0.20900` | `v2.9.0` |
| `1.0.0.20100` | `v2.10.0` |

安装时应选择与 CPU PyTorch wheel 匹配的 `torch_supa` wheel。

## 文档布局

- `docs/user_guider/source/`：用户指南，包含安装、特性、API、环境变量、工具、FAQ 和附录。
- `docs/sphinx/source/`：开发文档，包含算子注册、dispatch 调试和 PrivateUse1 适配。
- `.agents/skills/`：profiling、性能、算子开发和仓库自动化相关任务工作流。
