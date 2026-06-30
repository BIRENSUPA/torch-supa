---
name: torch-elementwise-feature
description: 用于指导 torch_supa 中 elementwise feature 的分类、开发和扩展。Use when users mention elementwise feature, new feature, dtype expansion, broadcast pattern, hybrid vectorized, static cast, contiguous unrolled, ldcg, development flow, or elementwise performance optimization.
depends_on: []
---

# Elementwise Feature 开发指导 Skill

本 Skill 负责 **elementwise feature 分类、开发指导、扩展路线建议**，不负责把 CUDA kernel 从上游直接迁移到 `torch_supa`。

## 什么时候使用

在下列场景优先使用本 Skill：

- 用户问“我想给 elementwise 加一个新 feature，应该改哪里”
- 用户要扩展已有 dtype 组合、broadcast pattern、hybrid vectorized 路径
- 用户要为某个已接入的 elementwise op 做 tuned elements-per-thread 调优
- 用户要判断当前需求应落在 `nocast`、`contiguous + dynamic_casting`、`non-contiguous + dynamic_casting` 哪条分支
- 用户需要理解当前 elementwise 的总体分流逻辑、构建/注册关系、性能约束
- 用户需要基于现有 static cast / hybrid vectorized / contiguous unrolled / ldcg 能力继续开发

## 什么时候不要使用

以下场景不要把本 Skill 当作主入口：

- 用户明确要把某个 CUDA elementwise kernel 迁移到 `torch_supa`
- 用户已经给出了 `src_file` 或 `kernel_entry`，目标是做最小改动的 CUDA → SUPA 适配

这类场景应优先使用：

- `new-operator/torch-elementwise-adaptation/SKILL.md`

本 Skill 可以负责先识别该请求属于“算子接入”，然后明确引导到 `torch-elementwise-adaptation`，但不负责执行迁移本身。

## 执行流程

### 第一步：先分类

先阅读：

- `references/elementwise-development-flow.md`

先把需求归入以下三类之一：

1. **算子接入**
2. **已有 feature 扩展**
3. **新 feature 开发**

如果用户描述不清晰，优先补充以下判断信息：

- 是在迁移新 `.cu` kernel，还是修改现有 `CUDALoops.cuh` / `Loops.cuh` / `BinaryInternal.h`
- 是想接入新算子，还是想补 dtype / broadcast / vectorized 路径
- 是否已经有明确的性能目标或退化场景

### 第二步：算子接入场景

如果分类结果是“算子接入”，输出中必须明确以下事项：

- 推荐优先使用 `new-operator/torch-elementwise-adaptation/SKILL.md`
- 目标文件通常位于 `torch_supa/csrc/aten/ops/kernels/elementwise/`
- 注册关系核心是 `REGISTER_PRIVATEUSE1_DISPATCH`
- hidden/local 与 dispatch 注册是两件事：很多 kernel/helper 可以保持 hidden/local，不需要因为注册就额外导出
- `torch_supa` 初始化晚于 `torch_supa`，因此通常只修改 `torch_supa` 即可覆盖已有注册

必要时建议补充阅读：

- `references/elementwise-development-summary.md`

### 第三步：已有 feature 扩展场景

如果分类结果是“已有 feature 扩展”，先阅读：

- `references/elementwise-development-summary.md`

输出中优先给出：

- 主要修改位置：`torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`、`torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`、`torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h`
- 当前请求更可能落在哪条已有路径：static cast、hybrid vectorized、contiguous unrolled、legacy nocast + ldcg、op 级 tuned elements-per-thread
- 需要验证哪些约束：dtype 命中、stride/broadcast pattern、`element_per_thread` 敏感性、功能正确性与性能数据

应优先复用已有路径，不要为单个 dtype / shape 重新设计一整套基础设施。

### 第四步：新 feature 开发场景

如果分类结果是“新 feature 开发”，先阅读：

- `references/elementwise-development-summary.md`

然后先判断 feature 的落点：

- `needs_dynamic_casting = false`
- `contiguous + dynamic_casting`
- `non-contiguous + dynamic_casting`

再输出建议：

- 优先挂接到已有 vectorized / unrolled / legacy / static-cast 分流
- 还是扩展现有 static cast、hybrid vectorized、tail fallback、ldcg 机制
- 只有在已有框架无法承载时，才建议新增分流节点

如果需要当前 feature 的模板实例化、具体 helper、调用链或性能数据细节，再阅读：

- `references/elementwise-new-feature.md`

如果最终新增了新的 feature，或对现有 feature 做了实质性扩展（例如新增一条主路径、新的触发条件、新的 helper 组合、新的调用链变化、关键性能结论更新），还需要同步补全文档：

- `references/elementwise-new-feature.md`
  - 补充该 feature 的概述、接入方式、执行路径、关键组件、调用链与性能数据；
- `references/elementwise-development-summary.md`
  - 按 Feature 1 / 2 的风格补充开发总结、分流变化、性能结论与总览描述。

## 每类场景的输出要求

### A. 算子接入

输出至少应包含：

- 属于“算子接入”而非 feature 扩展
- 推荐转向 `torch-elementwise-adaptation`
- 目标文件/注册位置
- `REGISTER_PRIVATEUSE1_DISPATCH` 与静态初始化注册关系
- hidden/local 与导出符号的边界

### B. 已有 feature 扩展

输出至少应包含：

- 优先修改位置
- 复用哪条现有 feature 路径
- 需要关注的性能约束
- 是否需要补 benchmark / 回归验证

### C. 新 feature 开发

输出至少应包含：

- 先判断当前改动落在哪条主分流
- 优先挂接已有路径还是新增分流节点
- 对模板膨胀、分支膨胀、维护成本的影响
- 是否需要同步补充文档与开发说明
- 若新增了新 feature 或显著扩展了现有 feature，需同步更新 `references/elementwise-new-feature.md` 与 `references/elementwise-development-summary.md`

## 参考文档的使用顺序

1. **先看** `references/elementwise-development-flow.md`
   - 用于第一步分类：算子接入 / 已有 feature 扩展 / 新 feature 开发
2. **再看** `references/elementwise-development-summary.md`
   - 用于总体分流逻辑、构建关系、符号与注册关系、开发约束、性能结论
3. **按需看** `references/elementwise-new-feature.md`
   - 仅在需要理解 static cast / hybrid vectorized / contiguous unrolled / ldcg 的具体实现细节时阅读
4. **当用户给出 test/op/kernel/shape 并要求判断是否适用 elementwise 优化时，看** `references/elementwise-performance-classification.md`
   - 用于先判断是否 elementwise、提取 shape/stride/dtype、匹配已有优化覆盖，再决定迁移、扩展或新增 feature。

## 与现有 Skill 的分工

- `new-operator/torch-elementwise-adaptation/SKILL.md`
  - 负责 CUDA elementwise kernel 迁移到 `torch_supa`
  - 强调最小必要替换，不改写主逻辑

- `performance/torch-elementwise-feature/SKILL.md`
  - 负责需求分类
  - 负责开发路线建议
  - 负责扩展点判断与 feature 指导
  - 负责在需要时把“算子接入”场景引导到 `torch-elementwise-adaptation`

## 回答风格要求

执行本 Skill 时，回答应尽量按以下结构组织：

1. 先说明当前请求属于哪一类
2. 再说明为什么这样分类
3. 再给出优先查看/修改的文件和路径
4. 最后给出开发约束、验证建议，以及是否应转向 `torch-elementwise-adaptation`
