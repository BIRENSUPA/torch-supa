# Agent 工作流

`.agents/skills/` 目录保存本仓库的任务型工作流。这些内容是给 agent 和开发者使用的操作手册，不是产品 API 文档。

## 主路由

`torch-supa` skill 是本仓库的主工作流路由。遇到下面类型的问题时，先从它进入：

- 性能回退或 elementwise 性能优化；
- 新算子接入或算子注册；
- PrivateUse1 和 SUPA 适配；
- 仓库 skill 体系维护。

## Profiler 工作流

`torch-kineto-profiler-br` skill 负责 torch profiler、Kineto、SUPTI activity、GPU trace、graph activity、driver activity、`with_stack`、缺事件和 profiler crash 等问题。

建议排查顺序：

1. 确认 Python profiler 入口真的开启了目标能力。
2. 检查 torch 桥接层或 dispatch 层。
3. 检查 Kineto shim。
4. 检查 SUPTI 或 driver activity 的注册和采集。
5. 最后再看 trace 汇总、导出和展示。

来源：`.agents/skills/torch-kineto-profiler-br/SKILL.md`。

## 性能工作流

`performance/torch-elementwise-feature` 面向 elementwise 性能 triage。只有当请求明确是 elementwise 算子性能、回退或优化 campaign 时，才优先使用它。

来源：`.agents/skills/performance/torch-elementwise-feature/SKILL.md`。
