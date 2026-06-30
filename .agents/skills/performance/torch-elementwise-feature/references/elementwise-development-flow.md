# Elementwise 后续开发流程

## 1. 文档目的

本文档用于说明 `torch_supa` 中 elementwise 的后续开发方式。这里的“开发”主要包括三类：

1. **算子接入**：将新的 `.cu` kernel 正确接入 SUDA 构建，并在符号可见性约束下完成注册与验证，以复用已有优化 feature。这部分算子接入优先使用 `new-operator/torch-elementwise-adaptation/SKILL.md`。
2. **已有 feature 扩展**：在已有实现上继续补充新的 dtype 组合、广播 pattern 或向量化路径。
3. **新 feature 开发**：开发新的优化 feature。这部分优先参考 `torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_development_summary.md` 中的整体流程图与现有分流逻辑。

核心原则：

- **优先复用已有基础设施，不做无关重构**
- **先判断改动属于哪一类，再进入对应流程**
- **先保证构建、注册、符号正确，再看功能与性能**

---

## 2. 第一类：算子接入

### 2.1 适用场景

适用于以下情况：

- 新增一个 elementwise 算子；
- 将 CUDA / torch-supa 中已有算子迁移到当前目录；
- 希望直接复用已有 static cast、hybrid vectorized、等优化路径。

### 2.2 推荐入口

这部分优先使用：

- `new-operator/torch-elementwise-adaptation/SKILL.md`

同时建议配合阅读：

- `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryMulKernel.cu`
- `torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_new_feature.md`
- `new-operator/torch-elementwise-adaptation/elementwise_cuda_to_supa_mapping.md`

### 2.3 关注重点

算子接入时重点确认：

1. 新 `.cu` 文件是否位于 `torch_supa/csrc/aten/ops/kernels/elementwise/`；
2. kernel 注册与 dispatch 是否正确；
3. 新增符号是否满足 `hidden/local` 约束；
4. 是否已经尽量复用现有优化基础设施，而不是重新造一套路径。

---

## 3. 第二类：已有 feature 扩展

### 3.1 适用场景

适用于以下情况：

- 给已有 kernel 补新的 dtype 组合；
- 给已有 binary/unrolled 路径补新的广播 pattern；
- 在现有分流框架上继续补向量化路径或参数调优。

### 3.2 主要修改位置

这类开发通常优先关注：

- `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`
- `torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`
- `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h`

### 3.3 关注重点

扩展已有 feature 时，重点确认：

- 是否命中了已有 static cast 路径；
- 是否适合继续走 hybrid vectorized；
- `element_per_thread` 等参数是否会带来已知退化；
- 新增 dtype / broadcast 场景是否有功能正确性与性能数据支撑。

当前已知结论可直接作为约束：

- 静态类型转换是稳定收益项；
- hybrid vectorized 是当前主优化方向；
- `16/8 element_per_thread` 比 32 更稳；
- 特殊广播 shape 需要重点回归。

---

## 4. 第三类：新 feature 开发

### 4.1 推荐前置阅读

这部分优先参考：

- `torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_development_summary.md`
- `torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_new_feature.md`

其中最重要的是先看 `elementwise_development_summary.md` 中的流程图，明确当前 elementwise kernel 的整体分流逻辑。

### 4.2 开发时先判断落点

在开发新 feature 前，先判断改动落在哪条分支：

- `needs_dynamic_casting = false`
- `contiguous + dynamic_casting`
- `non-contiguous + dynamic_casting`

这一步的目的，是先明确新 feature 是：

- 挂在已有 vectorized / unrolled / legacy 路径上；
- 还是扩展 static cast、hybrid vectorized、tail fallback 一类现有机制；
- 还是确实需要引入新的分流节点。

### 4.3 关注重点

开发新 feature 时，重点关注：

1. 是否有明确的性能动机，而不是只增加实现复杂度；
2. 是否能和现有分流逻辑兼容；
3. 是否会引入新的模板膨胀、分支膨胀或维护成本；
4. 是否需要同步补充文档、流程图和接入方式说明。

---

## 5. 通用检查项

无论属于哪一类开发，最后都建议统一检查以下内容：

### 5.1 构建接入

- `.cu` 文件是否进入 `ATEN_SUDA_SRCS`
- 是否编入 `LIBSUDAOP`
- 是否符合当前 SUDA 构建链路

### 5.2 符号约束

- 内部 helper 是否保持 hidden/local
- 跨库接口是否显式导出
- 是否避免与 native 同名符号冲突

### 5.3 功能验证

- kernel 是否编译通过
- dispatch 是否命中
- 基础 dtype 场景是否正确
- 广播场景是否正确

### 5.4 性能验证

- 是否保留静态类型转换收益
- 是否避免已知敏感 shape 退化
- 是否有明确 benchmark 数据支撑结论

---

## 6. 一句话总结

后续 elementwise 开发建议先判断是**算子接入**、**已有 feature 扩展**还是**新 feature 开发**，然后分别走对应流程；其中算子接入优先复用 `new-operator/torch-elementwise-adaptation/SKILL.md`，新 feature 开发优先对照 `elementwise_development_summary.md` 的整体分流逻辑。