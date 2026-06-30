# Elementwise 当前开发总结

## 1. 文档目的

本文档主要面向当前 `torch_supa` 的 elementwise 新特性开发总结，重点说明四件事：

1. elementwise `.cu` 文件是如何通过 CMake 中的 `SUDA` 路径接入构建的。
2. 为什么在当前构建里需要做符号 local / hidden 处理，以避免与 native 符号冲突。
3. 当前新增 feature 主要是如何围绕 `新增feature md`（`torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_new_feature.md`）落地的。
4. 当前优化带来了哪些性能收益，并直接在本文记录关键性能数据与结论。

为了先说明 `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh` 中 elementwise kernel 的整体分流逻辑，下面先给出**未引入当前新增 feature 前**的原版流程图，再给出**引入当前新增 feature 后**的流程图，方便对比 `launch_vectorized_kernel`、`launch_unrolled_kernel`、`launch_legacy_kernel` 的分流变化。

### 1.1 原版流程图（未引入 static contiguous / static cast feature）

```text
gpu_kernel_impl
├─ needs_dynamic_casting = false
│  └─ gpu_kernel_impl_nocast
│     ├─ iter.is_contiguous = true  → launch_vectorized_kernel
│     │                              ├─ vec_size = 8 / 4 / 2 → vectorized_elementwise_kernel
│     │                              └─ vec_size = 1         → unrolled_elementwise_kernel
│     └─ iter.is_contiguous = false → launch_legacy_kernel → elementwise_kernel
│
└─ needs_dynamic_casting = true
   ├─ iter.is_contiguous = true  → launch_unrolled_kernel → unrolled_elementwise_kernel
   └─ iter.is_contiguous = false → launch_legacy_kernel → elementwise_kernel
```

### 1.2 当前流程图（引入 static contiguous / static cast feature 后）

```text
gpu_kernel_impl
├─ needs_dynamic_casting = false
│  └─ gpu_kernel_impl_nocast
│     ├─ iter.is_contiguous = true  → launch_vectorized_kernel
│     │                              ├─ vec_size = 8 / 4 / 2 → vectorized_elementwise_kernel
│     │                              └─ vec_size = 1         → unrolled_elementwise_kernel
│     └─ iter.is_contiguous = false → launch_legacy_kernel → elementwise_kernel
│
└─ needs_dynamic_casting = true
   ├─ iter.is_contiguous = true
   │  ├─ try_launch_static_contiguous_unrolled_kernel 命中   → launch_unrolled_kernel → unrolled_elementwise_kernel + LoadWithStaticCast
   │  └─ try_launch_static_contiguous_unrolled_kernel 未命中 → launch_unrolled_kernel → unrolled_elementwise_kernel
   │
   └─ iter.is_contiguous = false
      ├─ try_launch_static_binary_kernel 命中
      │  ├─ input matches contiguous output = true  → launch_hybrid_vectorized_kernel → hybrid_vectorized_elementwise_kernel
      │  └─ input matches contiguous output = false → launch_legacy_kernel → elementwise_kernel
      └─ try_launch_static_binary_kernel 未命中     → launch_legacy_kernel → elementwise_kernel
```

---

## 2. CMake 中 SUDA 的引入方式

当前 elementwise `.cu` 文件的构建接入，主要通过两层 CMake 完成。

### 2.1 `torch_supa/csrc/aten/CMakeLists.txt` 负责收集源文件

- `torch_supa/csrc/aten/ops/kernels/elementwise/` 下新增的 `.cu` 文件，默认会被纳入 `ATEN_SUDA_SRCS`。
- 新增 elementwise kernel 时，通常不需要再单独修改这里的文件列表。

### 2.2 根 `CMakeLists.txt` 通过 `suda_target()` 构建 SUDA 动态库

在 `CMakeLists.txt:272` 定义了：

```cmake
function(suda_target)
  include(cmake/suda.cmake)
  list(APPEND SUPA_BRCC_FLAGS ... "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  suda_add_library(${LIBSUDAOP} SHARED ${ATEN_SUDA_SRCS} OPTIONS WITH_SUPA)
  target_compile_options(${LIBSUDAOP} PRIVATE "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  target_include_directories(${LIBSUDAOP} BEFORE PRIVATE ${TORCH_SUPA_DIR}/../include/)
endfunction()
suda_target()
```

这说明：

1. `ATEN_SUDA_SRCS` 中收集到的 `.cu` 文件，会进入 `LIBSUDAOP` 这个共享库；
2. elementwise 的 `.cu` 内核是作为 SUDA 动态库的一部分来编译的；
3. `WITH_SUPA` 选项说明使用了 SUDA 兼容 SUPA 的选项，可以 include SUPA 相关头文件。

## 3. 为什么需要做符号 local / hidden 处理

当前构建在两个位置显式加了：

- `-fvisibility=hidden`
- `-fvisibility-inlines-hidden`

### 3.1 目的：避免与 native 符号冲突

当前 `torch_supa` 的很多 elementwise 实现，本质上是：

- 复用 PyTorch ATen / native 体系的接口与命名习惯；
- 同时在 `PrivateUse1 / SUPA` 路径下重新实现或覆盖相关 kernel。

在这种场景下，如果所有符号默认都对外可见，就可能出现：

- 与 native / 原生同名符号暴露到同一动态链接空间；
- 链接期或运行期符号解析冲突；
- 本应只在 `LIBSUDAOP` 内部使用的 helper / kernel 符号，被外部错误解析或覆盖。

所以这里做 symbol local / hidden 的核心目的，就是：

- **尽量把 SUDA 动态库内部实现细节局部化；**
- **避免和 native 符号或其他共享库中的同名实现冲突；**
- **只导出明确需要对外暴露的接口。**

### 3.2 与导出宏、算子注册的关系

由于默认 visibility 已经收紧，所以如果某个符号确实需要跨编译单元或跨共享库使用，就必须显式导出，例如：

- `TORCH_SUPA_API`

仓库中已经有部分这类声明，例如：

- `TORCH_SUPA_API void direct_copy_kernel_cuda(TensorIteratorBase &iter)`

`direct_copy_kernel_cuda` 在 torch_supa/csrc/aten/common/Copy.cpp 中被调用，所以这个符号需要显式导出。这也是个较为特殊的例子， 主要是因为源代码被分到两个地方实现了。

这里还需要区分 **符号导出** 与 **算子注册**。

对 elementwise kernel 来说，很多接入动作依赖的是 `REGISTER_PRIVATEUSE1_DISPATCH`。这类注册不是通过外部动态符号查找来完成的，而是通过**静态变量初始化**，在动态库加载阶段把 kernel 注册到 PyTorch 全局变量中。所以对很多 elementwise 算子来说：

- kernel/helper 仍然可以保持 hidden/local；
- 只要 dispatch 注册代码在 `torch_supa` 对应动态库中正常执行，就能完成算子注册与覆盖；
- 不应把“需要注册”误解为“必须补导出宏”。

当前还有一个初始化顺序上的关键前提：`torch_supa` 的初始化发生在 `torch_supa` 之后。也就是说，

1. `torch_supa` 中已有的 `REGISTER_PRIVATEUSE1_DISPATCH` 注册会先进入全局 dispatcher；
2. `torch_supa` 后续加载时，可以用自己的注册结果覆盖 `torch_supa` 中已有的同类注册。

因此在迁移新算子到 `torch_supa` 时，一般**无需同步修改 `torch_supa`**。因为最终生效的是 `torch_supa` 在更晚初始化阶段的注册结果。

### 3.4 对 elementwise 开发的直接影响

对 elementwise 新 feature 开发来说，这一策略意味着：

1. **优先把 helper 留在本编译单元或本动态库内部**；
2. **不要假设未导出的函数一定能被其他地方链接到**；
3. **只有确实需要跨库直接调用的公共接口，才需要补导出属性**；
4. **通过 `REGISTER_PRIVATEUSE1_DISPATCH` 完成的注册，通常不需要为 kernel 本身额外导出符号**；
5. **迁移或新增算子时，优先修改 `torch_supa` 即可，通常不需要同步改 `torch_supa`。**

---

## 4. 当前新增 feature 的核心方向

当前 elementwise 侧新增的能力，不只是把已有 CUDA kernel 搬到 SUPA，而是围绕 **static cast 路径** 与 **legacy nocast path** 做了一层更可控的优化扩展。核心参考文档是：

- `torch_supa/csrc/aten/ops/kernels/elementwise/elementwise_new_feature.md:1`

这部分新增 feature 当前主要可以归纳为三类能力：

- **Feature 1**：binary elementwise kernel 的静态 dtype 特化、stride-aware hybrid vectorized 路径与 tail fallback。
- **Feature 2**：contiguous unrolled elementwise kernel 的静态 dtype 特化路径。
- **Feature 3**：legacy nocast path 的 ldcg load 优化路径。

由于当前平台的**指令发射效率相对更低**，且 **vcore 算力相对更弱**，运行时动态 `switch dtype`、kernel 内部 `type cast` 以及大 stride non-contiguous load 带来的访存开销，会比在高吞吐平台上更加敏感。在这种硬件特征下，动态 dtype 分派不仅会增加前端调度负担，也会进一步放大 kernel 内部非计算性开销在总时延中的占比；而对于 legacy nocast path，large stride 输入还更容易引入 L1 容量与 set 冲突，造成额外的缓存访问损耗。

因此，当前新增 feature 的设计重点并不只是增加几条可选优化路径，而是**尽可能将 dtype 判定与 cast 前移到静态特化阶段**，并在 legacy nocast 场景下**通过 ldcg 把不适合留在 L1 的 strided load 切换到 cached global load**，把运行时的动态分派、类型转换和不必要的缓存冲突从主执行路径中剥离。这样做的目的，是让有限的指令发射、vcore 计算资源和缓存资源更多用于有效访存与实际计算，从而在当前平台上获得更稳定、也更可放大的性能收益。

### 4.1 Feature 1：non-contiguous static cast + hybrid vectorized 主路径

Feature 1 的核心入口是 `try_launch_static_binary_kernel`，它负责在 binary elementwise 场景下先做 dtype 匹配，再根据 stride 关系选择 hybrid vectorized 或静态 fallback 路径。

#### 核心流程图

```text
gpu_kernel_impl
  └─ try_launch_static_binary_kernel<InputType1, InputType2, OutputType>
       ├─ 检查 arity / functor / runtime dtypes 是否匹配
       ├─ 检查 input0 / input1 是否与 output stride 对齐
       ├─ 若某个输入 stride 匹配
       │    ├─ make_input_hybrid_offset_calculator
       │    └─ launch_hybrid_vectorized_kernel
       │         └─ hybrid_vectorized_elementwise_kernel
       │              ├─ matched input: vectorized load
       │              ├─ unmatched input: scalar load + offset
       │              └─ output: vectorized store
       └─ 若都不匹配
            └─ invoke_impl_static / legacy static path
```

#### 核心功能组件

- `try_launch_static_binary_kernel`
  - 负责把运行时 dtype 与模板参数做匹配，命中后进入静态类型已知的 binary kernel 路径。
  - 它也是 hybrid vectorized 与静态 fallback 的统一调度入口。

- `input_has_same_strides_as_output`
  - 用于判断某个输入是否与输出具备一致的元素 stride。
  - 命中后该输入就可以优先走向量化加载，而不是继续走完全通用的 offset 路径。

- `make_input_hybrid_offset_calculator` / `HybridOffsetCalculator`
  - 为 hybrid 访存构造 offset 计算器，让匹配输入走 trivial offset，非匹配输入保留真实 stride 计算。
  - 这样既保住可向量化输入的吞吐，也兼容 broadcast 或非连续输入。

- `launch_hybrid_vectorized_kernel`
  - 负责把 stride 匹配信息转成具体 kernel launch，并选择哪个输入作为 matched input。
  - 它把主路径与 tail 路径放到统一入口下，避免在上层分散处理。

- `hybrid_vectorized` policy
  - 负责组织 matched input 的 vectorized load、unmatched input 的 scalar load，以及 output 的 vectorized store。
  - 它本质上定义了“一个输入快路径、另一个输入保守路径”的混合访存策略。

- `invoke_impl_static`
  - 在无法进入 hybrid vectorized 时，提供静态类型已知的调用方式，避免继续依赖运行时类型转换。
  - 它保证即使不做 hybrid 访存，也能保留 static dtype 特化收益。

- tail fallback
  - full block 走 hybrid vectorized 主路径，尾块则回退到 unroll + static cast 的安全路径。
  - 这样优化路径不仅适用于 benchmark 主干，也能覆盖真实 TensorIterator 的尾部数据。

### 4.2 Feature 2：contiguous unrolled static cast 路径

Feature 2 的核心入口是 `try_launch_static_contiguous_unrolled_kernel`，用于在 contiguous unrolled elementwise kernel 中复用静态 dtype 特化能力。

#### 核心流程图

```text
gpu_kernel_impl
  └─ contiguous branch
       └─ try_launch_static_contiguous_unrolled_kernel<InputType1, InputType2, OutputType>
            ├─ 检查 arity / 返回类型 / runtime dtypes 是否匹配
            ├─ 构造输入 TrivialOffsetCalculator
            ├─ 构造输出 TrivialOffsetCalculator
            ├─ LoadWithStaticCast<InputType1, InputType2>
            ├─ StoreWithoutCast
            └─ launch_unrolled_kernel
```

#### 核心功能组件

- `try_launch_static_contiguous_unrolled_kernel`
  - 负责在 contiguous 分支下检查静态 dtype 组合是否命中，命中后直接拉起 unrolled kernel。
  - 它把 contiguous 路径下的静态类型转换独立出来，避免继续走通用 `LoadWithCast`。

- `LoadWithStaticCast`
  - 负责在加载阶段完成编译期已知的静态类型转换。
  - 相比运行时按 dtype 查表转换，这条路径更直接，也更容易被编译器优化。

### 4.3 Feature 3：legacy nocast path 的 ldcg 优化

Feature 3 的核心入口是 `gpu_kernel_impl_nocast`，用于在 legacy nocast path 下按运行时 stride 特征选择是否启用 ldcg load。

#### 核心流程图

```text
gpu_kernel_impl
  └─ needs_dynamic_casting = false
       └─ gpu_kernel_impl_nocast
            ├─ iter.is_contiguous = true  → launch_vectorized_kernel
            └─ iter.is_contiguous = false
                 ├─ legacy_path_all_inputs_support_ldcg<traits>() = false
                 │    └─ launch_legacy_kernel → invoke<false>
                 └─ legacy_path_all_inputs_support_ldcg<traits>() = true
                      ├─ legacy_path_has_large_stride_input_impl<traits>() = true
                      │    └─ launch_legacy_kernel → invoke<true> → invoke_impl_ldcg
                      └─ legacy_path_has_large_stride_input_impl<traits>() = false
                           └─ launch_legacy_kernel → invoke<false>
```

#### 核心功能组件

- `gpu_kernel_impl_nocast`
  - 负责在 nocast 场景下区分 contiguous fast path 与 non-contiguous legacy path。
  - 它也是 Feature 3 的统一入口：只有进入 non-contiguous legacy path 后，才会继续判断是否启用 ldcg。

- `legacy_path_all_inputs_support_ldcg`
  - 负责检查当前 functor 的所有输入类型是否都支持 ldcg，且输入类型不包含 `bool`。
  - 它用于在编译期先过滤掉不适合启用 ldcg 的 dtype 组合。

- `legacy_path_has_large_stride_input_impl`
  - 负责在运行时检查是否至少有一个输入满足 large inner-dim byte stride 条件。
  - 命中后才认为当前 legacy path 值得切换到 ldcg load。

- `input_inner_dim_byte_stride_gt_256`
  - 用于判断单个输入的 inner dim byte stride 是否大于 256。
  - 它体现了 Feature 3 的核心触发条件：优先针对大 stride 的 strided load 场景做缓存访问优化。

- `invoke<true>` / `invoke<false>`
  - 负责作为是否启用 ldcg 的统一分派点。
  - `invoke<true>` 会进入 `invoke_impl_ldcg`，`invoke<false>` 则保留原有普通 load 路径。

- `invoke_impl_ldcg`
  - 负责把所有输入参数的加载统一切换到 ldcg 版本，再调用 functor。
  - 它使 legacy kernel 本身无需重写，只需要在参数加载阶段切换到 cached global load。

- `load_nocast_arg_ldcg`
  - 负责对单个输入参数执行 ldcg 版本的 load，底层通过 `memory::load_global_cached` 读取。
  - 它是 Feature 3 最底层的 cached global load 组件。

### 4.4 新增 feature 的接入方式已经模板化

`elementwise_new_feature.md` 开头给出的“快速指南”已经明确了新增 feature 的推荐接入方式：

在 `CUDALoops.cuh` 中补对应 dtype 组合的 `try_launch_static_binary_kernel` 或 `try_launch_static_contiguous_unrolled_kernel` 进行更多 dtype 的实例化；


## 5. 当前性能数据总结

本节性能数据同时覆盖三部分：

- **4.1 Feature 1：static cast + hybrid vectorized 主路径**
- **4.2 Feature 2：contiguous unrolled static cast 路径**
- **4.3 Feature 3：legacy nocast path 的 ldcg 优化**

### 5.1 Feature 1 性能数据

当前优化路径可以概括为四个阶段：

- Base：动态类型转换 + `elementwise_kernel`
- V1：静态类型转换 + `elementwise_kernel`
- V2：静态类型转换 + burst4 + 32 `element_per_thread` + `hybrid_vectorized_elementwise_kernel`
- V3(主线)：静态类型转换 + burst4 + 16/8 `element_per_thread` + `hybrid_vectorized_elementwise_kernel`

| Shape | DType | Base(μs) | V1(μs) | V1/Base提升 | V2(μs) | V2/Base提升 | V3(μs) | V3/Base提升 |
|-------|-------|----------|--------|-------------|--------|-------------|--------|-------------|
| `(1,2048,24,64) * (1,2048,1,64)` | bf16+float32 | 79.49 | 52.35 | **34.1%** | 28.42 | **64.3%** | 25.22 | **68.3%** |
| `(1,2048,24,64) * (1,2048,1,64)` | float32+bf16 | 77.18 | 55.81 | **27.7%** | 47.49 | **38.5%** | 25.98 | **66.3%** |
| `(1,1024,5120) * (1,1,5120)` | bf16+float32 | 129.28 | 75.90 | **41.3%** | 115.58 | 10.6% | 32.51 | **74.9%** |
| `(1,1024,5120) * (1,1,5120)` | float32+bf16 | 114.94 | 78.34 | **31.8%** | 209.28 | *-82.1%* | 50.56 | **56.0%** |
| `(594,4096) * (594,1)` | bf16+float32 | 53.38 | 32.51 | **39.1%** | 18.56 | **65.2%** | 18.30 | **65.7%** |
| `(594,4096) * (594,1)` | float32+bf16 | 48.26 | 33.79 | **30.0%** | 21.89 | **54.6%** | 18.18 | **62.3%** |
| `(647,4096) * (647,1)` | bf16+float32 | 58.24 | 35.07 | **39.8%** | 21.89 | **62.4%** | 22.02 | **62.2%** |
| `(647,4096) * (647,1)` | float32+bf16 | 52.48 | 36.86 | **29.8%** | 24.83 | **52.7%** | 18.05 | **65.6%** |
| `(679,4096) * (679,1)` | bf16+float32 | 60.67 | 37.25 | **38.6%** | 21.76 | **64.1%** | 22.14 | **63.5%** |
| `(679,4096) * (679,1)` | float32+bf16 | 56.06 | 39.30 | **29.9%** | 31.62 | **43.6%** | 20.22 | **63.9%** |
| **平均** | - | - | - | **34.2%** | - | **49.4%** | - | **64.9%** |

> 注：负值表示性能退化；加粗表示性能提升。

**Feature 1 结论**

- 静态类型转换本身就是稳定收益项；
- hybrid vectorized 是当前 binary 路径的主优化方向；
- `element_per_thread` 不是越大越好，当前 16/8 比 32 更稳。

### 5.2 Feature 2 性能数据

对于 **4.2 Feature 2：contiguous unrolled static cast 路径**，当前已有一组代表性性能数据：

| Shape | DType | 优化前 | 优化后 | 性能提升 |
|-------|-------|--------|--------|----------|
| `(131072, 128)` | `bf16 * fp32` | `0.6 ms` | `0.15 ms` | **75.0%** |

这说明即使在当前实现中只引入了 **static cast**，还没有继续叠加 burst 优化，contiguous 路径本身也已经能拿到比较明显的收益。

**Feature 2 结论**

- 当前只做了 static cast，没有继续做 burst；
- 因此这一路径理论上仍有进一步优化空间；
- 但如果继续叠加 burst 等优化，也会带来更多模板实例化与分支展开，从而增加代码膨胀量。

### 5.3 Feature 3 性能数据

对于 **4.3 Feature 3：legacy nocast path 的 ldcg 优化**，当前已有一组代表性性能数据：

| op name | shape | dtype | stride | br 平均耗时 | nv 平均耗时 | ldcg 耗时 |
|---|---|---|---|---:|---:|---:|
| `add` | `[1, 1024, 5120]` | `float32` | `a=[5242880, 1, 1024]`, `b=[5242880, 5120, 1]` | `1.235 ms` | `0.040 ms` | `0.175 ms` |
| `copy` | `[1, 1024, 5120]` | `float32` | `dst=[5242880, 5120, 1]`, `src=[5242880, 1, 1024]` | `0.594 ms` | `0.036 ms` | `0.150 ms` |
| `copy` | `[1, 2048, 16, 4, 128]` | `bfloat16` | `dst=[16777216, 8192, 512, 128, 1]`, `src=[2048, 1, 262144, 0, 2048]` | `1.456 ms` | `0.105 ms` | `0.440 ms` |

这些数据说明，在 **large stride + nocast + legacy path** 场景下，ldcg 可以显著降低当前 br 路径的 strided load 开销。

**Feature 3 结论**

- 该优化只面向 `needs_dynamic_casting == false` 的 legacy nocast path；
- 触发条件不是 contiguous，而是至少一个输入存在较大的 inner dim byte stride；
- 优化核心是通过 `memory::load_global_cached` 避免 strided load 在 L1 中产生额外冲突；
- 因此它更适合作为 non-contiguous nocast 场景的定向访存优化，而不是通用 fast path。

---

## 6. 当前总结

当前 `elementwise_development_summary.md` 所总结的重点，可以归纳为：

1. 从构建接入看：
   - elementwise `.cu` 通过 `ATEN_SUDA_SRCS` 进入 SUDA 动态库；
   - 当前 CMake 已经把 `ops/kernels/elementwise/*.cu` 纳入构建。

2. 从符号管理看：
   - `-fvisibility=hidden` / `-fvisibility-inlines-hidden` 是当前必要策略；
   - 目的是避免与 native 符号冲突；
   - 对外符号必须通过 `TORCH_SUPA_API` 显式导出。

3. 从新增 feature 看：
   - binary 路径的核心是 static dtype 特化、hybrid vectorized、HybridOffsetCalculator 与 tail fallback；
   - contiguous unrolled 路径的核心是 `try_launch_static_contiguous_unrolled_kernel` 及其静态类型转换加载；
   - legacy nocast 路径新增了基于 large stride 判定的 ldcg load 优化。

4. 从性能数据看：
   - 静态类型转换带来稳定收益；
   - hybrid vectorized 是主方向；
   - `16/8 element_per_thread` 是当前更稳的配置；
   - contiguous static cast 单例 shape 提升达到 **75.0%**；
   - ldcg 在 large stride nocast 场景下可明显降低 strided load 开销；
   - 当前最优平均提升达到 **64.9%**。

一句话说，当前 elementwise 的工作重点已经从“单纯迁移 kernel”转向“在 SUDA 动态库和符号隔离约束下，同时扩展 static cast、contiguous unrolled 与 legacy nocast ldcg 三类优化能力，并把优化收益稳定落到真实 shape 上”。
