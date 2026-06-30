---
name: torch-supa-wiki
description: torch-supa 仓库的中文 Wiki。Use when users ask for torch-supa wiki、仓库概览、安装指南、运行时 API、算子注册、PrivateUse1 适配、profiling 或 agent 工作流说明。
---

# Torch SUPA Wiki

这个 skill 是当前 `torch-supa` 仓库的轻量中文 Wiki。它基于 `docs/` 中的用户/开发文档，以及 `.agents/skills/` 中的 agent 工作流说明整理而成。

## 使用方式

1. 先读 `README.md`，了解模块列表。
2. 根据主题阅读对应页面，获取简明说明和源码/文档入口。
3. 做实现决策前必须再检查当前代码；Wiki 是文档摘要，可能落后于最新源码。

## 模块说明

- `product-overview.md`：Torch SUPA 的定位、版本/包映射和架构概览。
- `installation-quickstart.md`：环境要求、Docker 流程、安装和首次验证。
- `runtime-features-apis.md`：CUDA 到 SUPA 的兼容模型、AMP、profiler、graph mode 和 API 分组。
- `operators-privateuse1.md`：算子归属、注册流程和开源库 PrivateUse1 适配。
- `debugging-profiling-tools.md`：环境变量、环境采集、日志、profiler 排障和精度工具。
- `agent-workflows.md`：本仓库用于 profiler、性能和算子开发的 AI 工作流。

## 维护规则

- 页面保持简短，内容必须能追溯到仓库文档或当前 skill。
- 长表格和细节优先链接到源文档，不在 Wiki 中大段复制。
- 当 `docs/` 或 `.agents/skills/` 发生实质变化时，同步更新对应页面。
