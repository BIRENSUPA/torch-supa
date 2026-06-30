## 快速指南：为新算子应用优化 Kernel

如果希望为新的 elementwise 算子应用优化 kernel，可参考以下步骤。对于只是希望后续复用当前 elementwise 优化能力的场景，通常优先完成对应的 elementwise op stub 注册即可；完成 stub 接入后，即可继续沿用现有 static binary / hybrid vectorized 框架，在 `CUDALoops.cuh` 中补充所需 dtype 特化或走 hybrid vectorized 路径，而不需要为每个算子重新设计整套基础设施。

如需迁移已有且需要优化能力的 op，优先参考 `new-operator/torch-elementwise-adaptation/SKILL.md`，直接迁移对应 op stub 所在源文件即可。

---

## Feature 1：elementwise kernel 新增支持静态类型转换及 burst

### 概述

Feature 1 以 `try_launch_static_binary_kernel` 为入口，为 binary elementwise kernel 引入两类能力：

1. **静态类型转换**：在输入 dtype 与 functor 参数类型不完全一致时，仍可基于模板参数命中特化路径。
2. **burst / hybrid vectorized 优化**：当某个输入的 stride 与输出一致时，优先让该输入走向量化加载，另一个输入保留真实 stride 计算。

当前实现主要面向 binary 场景，核心入口位于 `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`。

### 接入方式

如果需要特化新的数据类型组合（如 `bfloat16 + float`），在 `CUDALoops.cuh` 的 `gpu_kernel_impl` 中实例化 `try_launch_static_binary_kernel`：

```cpp
if constexpr (traits::arity == 2 && std::is_same<arg0_t, float>::value) {
  // 特化 InputType1=BFloat16, InputType2=float, OutputType=float 的组合
  if (try_launch_static_binary_kernel<func_t, c10::BFloat16, float, float>(
          iter, f, data, numel, dtypes)) {
    return;
  }
  // 特化 InputType1=float, InputType2=BFloat16, OutputType=float 的组合
  if (try_launch_static_binary_kernel<func_t, float, c10::BFloat16, float>(
          iter, f, data, numel, dtypes)) {
    return;
  }
}
```

模板参数说明：
- `InputType1`: 输入张量 0 的 C++ 类型
- `InputType2`: 输入张量 1 的 C++ 类型
- `OutputType`: 输出张量的 C++ 类型

### 执行路径

#### `try_launch_static_binary_kernel`

```cpp
template <typename func_t, typename InputType1, typename InputType2, typename OutputType>
bool try_launch_static_binary_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const std::array<char*, 3>& data,
    int64_t numel,
    const std::array<ScalarType, 3>& dtypes);
```

##### 功能

尝试以静态已知的输入输出类型启动 kernel。如果实际 dtype 与模板参数匹配，则使用优化的 kernel 执行；否则返回 `false` 让调用者回退到其他实现。

##### 参数说明

- `func_t`: 运算函数类型（如 `[](float a, float b) { return a + b; }`）
- `InputType1`: 输入张量 0 的实际 C++ 类型
- `InputType2`: 输入张量 1 的实际 C++ 类型
- `OutputType`: 输出张量的实际 C++ 类型
- `data`: 数据指针数组 `[output, input0, input1]`
- `dtypes`: 运行时 dtype 数组，用于验证类型匹配

##### 总体流程

```
┌─────────────────────────────────────────────────────────────┐
│           try_launch_static_binary_kernel                    │
├─────────────────────────────────────────────────────────────┤
│ 1. 检查 arity == 2 && dtype 匹配                             │
│    ├─ dtypes[1] == CppTypeToScalarType<InputType1>          │
│    ├─ dtypes[2] == CppTypeToScalarType<InputType2>          │
│    └─ dtypes[0] == CppTypeToScalarType<OutputType>          │
├─────────────────────────────────────────────────────────────┤
│ 2. 检查 stride 匹配情况                                       │
│    ├─ input_matches_contiguous_output<0>(iter)             │
│    ├─ input_matches_contiguous_output<1>(iter)             │
│    └─ 都不匹配 → fallback 到 legacy kernel                   │
├─────────────────────────────────────────────────────────────┤
│ 3. 选择执行路径                                               │
│    ├─ stride 匹配 → hybrid vectorized                        │
│    └─ 无 stride 匹配 → invoke_impl_static                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键组件

#### `input_matches_contiguous_output`

```cpp
template <int input_idx>
bool input_matches_contiguous_output(const TensorIteratorBase& iter);
```

##### 功能

检查指定输入张量的元素 stride 是否与输出张量相同。相同则意味着该输入可以用向量化加载。

##### 实现要点

- 比较的是**元素 stride**（element stride），不是字节 stride（byte stride）
- 因为元素大小可能不同（如 `float` vs `bfloat16`），比较前需要先按元素大小换算
- 这个判断决定 `try_launch_static_binary_kernel` 进入 hybrid vectorized 还是 legacy fallback

#### `launch_hybrid_vectorized_kernel`

```cpp
template <typename func_t, typename array_t, typename inp_calc_t,
          int MatchedInputIdx, typename InputType0, typename InputType1>
static inline void launch_hybrid_vectorized_kernel(
    int64_t N, const func_t& f, array_t data, inp_calc_t input_calc);
```

##### 功能

启动 hybrid vectorized kernel，根据 `MatchedInputIdx` 决定哪个输入走向量化加载。

##### 当前配置

```cpp
constexpr int vec_size = 4;  // 向量化大小 4（128-bit 对齐）
// elems_per_thread 由 io_size 决定：
// - io_size <= 96  -> 16
// - io_size > 96   -> 8
```

##### 双路径设计

- **Main path**：进入 `hybrid_vectorized_elementwise_kernel` 的主路径，匹配输入走 vectorized load，非匹配输入走 scalar load
- **Tail path**：尾块退回到带静态类型转换的 unroll 路径，避免单独实现另一套尾块 kernel

#### `make_input_hybrid_offset_calculator`

```cpp
template<int N, int MatchedArg, bool signed_strides = false>
static HybridOffsetCalculator<N, MatchedArg, uint32_t, signed_strides>
make_input_hybrid_offset_calculator(const at::TensorIteratorBase& iter);
```

##### 功能

创建混合 offset 计算器，仅用于输入张量（排除输出张量）。

##### 作用

- `MatchedArg` 对应的输入直接使用 trivial offset
- 其他输入按真实 stride 计算 offset
- 这样既能保留一个输入的向量化路径，又能兼容另一个输入的 broadcast / 非连续布局

#### `hybrid_vectorized` Policy

```cpp
template <
    int vec_size,
    typename data_t,
    int elems_per_thread,
    typename inp_calc_t,
    int MatchedInputIdx,
    typename InputType0,
    typename InputType1>
struct hybrid_vectorized;
```

##### 功能

定义 hybrid vectorized 的内存访问策略，在同一轮计算中混合使用 vectorized load、scalar load 和 vectorized store。

##### 核心成员

```cpp
static constexpr int loop_size = elems_per_thread / vec_size;  // = 8
static constexpr int block_work_size = elems_per_thread * num_threads();

data_t data;
int remaining;
inp_calc_t input_offset_calculator;
```

##### 加载策略

- 对 `MatchedInputIdx` 对应的输入，`hybrid_vectorized_load_helper` 使用 `load_vector<vec_size>` 做向量化读取，并在需要时做 `static_cast`
- 对未匹配输入，`hybrid_scalar_load_helper` 按相同 element 顺序生成 `linear_index`，再用 `input_offset_calculator.get(...)` 取真实 offset 做标量读取
- 两个 helper 必须保持一致的 index pattern，才能保证同一批元素在 functor 输入上正确对齐
- `detail::static_unroll<load_helper, arity>` 用于在编译期展开每个参数的 load 路径

##### 存储策略

- 输出默认与线性索引匹配，因此 `store(...)` 统一走 vectorized store
- 这使得 hybrid policy 在 input 侧混合加载的同时，output 侧仍保持连续写回

##### 适用场景

- 一个输入与输出 stride 一致，另一个输入可能 broadcast 或非连续
- 希望在 mixed dtype 场景下复用现有 burst / vectorized 基础设施
- 当前本地扩展还会在 `gpu_kernel_impl_nocast` 中复用这条路径，用于一个**窄范围的 masked_fill 风格 `bf16 + bool -> bf16` broadcast 场景**；这不是 generic nocast binary fast path

##### fallback

如果两个输入都不匹配输出 stride，则回退到 legacy kernel，使用 `invoke_impl_static` 做静态类型转换。

#### `invoke_impl_static`

```cpp
template <typename traits, typename func_t, typename index_t, typename... InputTypes, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl_static(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<I...>);
```

##### 功能

使用编译时已知的输入类型调用函数，避免运行时类型转换开销。

##### 与 `invoke` 的对比

```cpp
// invoke: 运行时类型转换
return f(c10::fetch_and_cast<arg_t>(dtypes[I], data[I] + i * strides[I])...);

// invoke_impl_static: 编译时类型转换
```

---
return f(c10::convert<arg_t>(
    c10::load<std::tuple_element_t<I, std::tuple<InputTypes...>>>(
        data[I] + i * strides[I]))...);
```

#### 示例

假设 `InputType1 = bfloat16`, `InputType2 = float`, `arg_t = float`:

```cpp
// invoke_impl_static<..., bfloat16, float>
// I=0: c10::convert<float>(c10::load<bfloat16>(data[0] + ...))
// I=1: c10::convert<float>(c10::load<float>(data[1] + ...))
// 结果: f(float_val_from_bf16, float_val)
```

---

### 性能优化要点

#### 1. 向量化收益

| 配置 | 每线程处理元素 | 内存访问方式 |
|------|--------------|-------------|
| 标准 unroll | 4 | 标量 |
| hybrid_vectorized | 16/8 | 向量化（匹配输入） |

#### 2. 类型转换优化

- **静态类型**: 编译时确定，生成高效的类型转换代码
- **动态类型**: 运行时查表，有额外开销

#### 3. 关键性能数据

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

#### 4. 内存访问模式

```
匹配输入 (vec_size=4):
  Thread 0: load [0, 1, 2, 3], [128, 129, 130, 131], ...
  Thread 1: load [4, 5, 6, 7], [132, 133, 134, 135], ...

非匹配输入:
  使用 offset 计算器，每个元素独立计算地址
```

---

### 完整调用链示例

以 `float + bfloat16` 为例，input0 stride 匹配 output:

```
gpu_kernel_impl
  └─ try_launch_static_binary_kernel<func_t, float, bfloat16, float>
       ├─ dtype 匹配检查: dtypes[1]==Float, dtypes[2]==BFloat16 ✓
       ├─ stride 检查: input_has_same_strides_as_output<0>(iter) ✓
       ├─ hybrid_input_calc = make_input_hybrid_offset_calculator<2, 0>(iter)
       └─ launch_hybrid_vectorized_kernel<..., 0, float, bfloat16>
            └─ hybrid_vectorized_elementwise_kernel<4, ..., 0, float, bfloat16>
                 ├─ Main: hybrid_vectorized<4, ..., 0, float, bfloat16>
                 │    ├─ load arg 0: hybrid_vectorized_load_helper<..., float>
                 │    ├─ load arg 1: hybrid_scalar_load_helper<..., bfloat16>
                 │    └─ store: vectorized store
                 └─ Tail: unroll + LoadWithStaticCast<float, bfloat16>
```
### Feature 1A：nocast path 复用 hybrid vectorized 的窄范围扩展

### 概述

当前代码在 `gpu_kernel_impl_nocast` 中增加了一个**窄范围** special case：当 binary op 满足

- `traits::arity == 2`
- `arg0_t == c10::BFloat16`
- 运行时 dtype 命中 `bf16 + bool -> bf16`

时，会尝试复用 `try_launch_static_binary_kernel<func_t, c10::BFloat16, bool, c10::BFloat16>`，以便在 masked_fill 风格的 broadcast 场景中让 data 输入走 hybrid vectorized load，而让 bool mask 维持真实 stride/scalar load。

### 适用场景

该分支当前主要面向类似如下模式：

- 输出 / data 输入为 `bf16`
- mask 输入为 `bool`
- 一个输入与输出 stride 一致
- mask 可能是 broadcast 或 non-contiguous

### 执行路径

```cpp
if constexpr (traits::arity == 2 && std::is_same<arg0_t, c10::BFloat16>::value) {
  if (try_launch_static_binary_kernel<func_t, c10::BFloat16, bool, c10::BFloat16>(
          iter, f, data, numel, dtypes)) {
    return;
  }
}
```

如果该特化未命中，`gpu_kernel_impl_nocast` 会继续回退到原有 legacy nocast 路径。

### 当前限制

这条 nocast 扩展故意保持很窄，而没有放开成 generic nocast binary hybrid path，原因是当前 hybrid load helper 默认假设存在简单的 `input_t -> arg_t` 标量转换路径。这个假设适用于诸如：

- `bf16 -> float`
- `bool -> bf16`
- 同类型 load

但不适用于 generic opmath / complex functor 参数类型，因此当前实现没有把 hybrid nocast 扩展到所有 binary 算子。


### tuned elements-per-thread 说明

`gpu_kernel(...)` / `gpu_kernel_nocast(...)` 的公开入口保留了 `tuned_elems_per_thread` 参数，默认值为 `0`。当前语义为：

- `0`：关闭 tuned path，继续走默认 heuristic / 默认 vectorized 路径
- `8 / 16 / 24`：启用 `launch_vectorized_kernel_tuned(...)`

因此 `0` 只是“disabled / use default”的哨兵值，并不是一个合法的 tuned 配置值。

## Feature 2：unrolled elementwise kernel 新增支持静态类型转换

当前实现主要面向 contiguous 场景：当 `gpu_kernel_impl` 检测到 `iter.is_contiguous()` 且满足已特化的 binary dtype 组合时，会优先尝试走 `try_launch_static_contiguous_unrolled_kernel`；命中后直接启动 unrolled kernel，否则再回退到原有的 `LoadWithCast` + `launch_unrolled_kernel` 路径。

### 接入方式

如果需要特化新的 contiguous dtype 组合（如 `bfloat16 + float -> float`），在 `CUDALoops.cuh` 的 `gpu_kernel_impl` contiguous 分支中实例化 `try_launch_static_contiguous_unrolled_kernel`：

```cpp
if constexpr (traits::arity == 2 && std::is_same<arg0_t, float>::value) {
  if (try_launch_static_contiguous_unrolled_kernel<func_t, c10::BFloat16, float, float>(
          iter, f, data, numel, dtypes) ||
      try_launch_static_contiguous_unrolled_kernel<func_t, float, c10::BFloat16, float>(
          iter, f, data, numel, dtypes)) {
    return;
  }
}
```

模板参数说明：
- `InputType1`: 输入张量 0 的 C++ 类型
- `InputType2`: 输入张量 1 的 C++ 类型
- `OutputType`: 输出张量的 C++ 类型

### 执行路径

#### `try_launch_static_contiguous_unrolled_kernel`

```cpp
template <typename func_t, typename InputType1, typename InputType2, typename OutputType>
bool try_launch_static_contiguous_unrolled_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const std::array<char*, 3>& data,
    int64_t numel,
    const std::array<ScalarType, 3>& dtypes);
```

##### 功能

尝试以静态已知的输入输出类型启动 contiguous unrolled kernel。如果实际 dtype 与模板参数匹配，则使用静态类型转换的 unrolled 路径；否则返回 `false` 让调用者回退到原有实现。

##### 参数说明

- `func_t`: 运算函数类型
- `InputType1`: 输入张量 0 的实际 C++ 类型
- `InputType2`: 输入张量 1 的实际 C++ 类型
- `OutputType`: 输出张量的实际 C++ 类型
- `data`: 数据指针数组 `[output, input0, input1]`
- `dtypes`: 运行时 dtype 数组，用于验证类型匹配

##### 总体流程

```
┌────────────────────────────────────────────────────────────────┐
│     try_launch_static_contiguous_unrolled_kernel              │
├────────────────────────────────────────────────────────────────┤
│ 1. 检查 arity == 2 && dtype 匹配                               │
│    ├─ dtypes[1] == CppTypeToScalarType<InputType1>            │
│    ├─ dtypes[2] == CppTypeToScalarType<InputType2>            │
│    └─ dtypes[0] == CppTypeToScalarType<OutputType>            │
├────────────────────────────────────────────────────────────────┤
│ 2. 构造 contiguous offset / load / store 组件                  │
│    ├─ TrivialOffsetCalculator<traits::arity>                  │
│    ├─ TrivialOffsetCalculator<1>                              │
│    ├─ memory::detail::LoadWithStaticCast<InputType1,InputType2>│
│    └─ memory::StoreWithoutCast()                              │
├────────────────────────────────────────────────────────────────┤
│ 3. 启动 contiguous unrolled kernel                             │
│    └─ launch_unrolled_kernel                                   │
└────────────────────────────────────────────────────────────────┘
```

### 关键组件

#### `memory::detail::LoadWithStaticCast<InputType1, InputType2>`

##### 功能

按编译期已知的输入类型执行 load，并在 load 后完成静态类型转换。

##### 作用

- 避免 runtime cast 路径
- 让 contiguous unrolled kernel 复用静态 dtype 特化能力
- 与 `launch_unrolled_kernel` 配合完成 contiguous fast path

#### `memory::StoreWithoutCast`

##### 功能

按输出类型直接写回结果，不再经过额外的运行时类型转换。

#### `TrivialOffsetCalculator`

##### 功能

在 contiguous 场景下为输入和输出提供 trivial offset 计算。

##### 作用

- 输入使用 `TrivialOffsetCalculator<traits::arity>`
- 输出使用 `TrivialOffsetCalculator<1>`
- 与 unrolled kernel 的线性访问模式匹配

### 性能优化要点

#### 类型转换优化

Feature 2 的核心收益来自类型转换优化：

- 输入类型在编译期确定，`memory::detail::LoadWithStaticCast<InputType1, InputType2>` 可直接生成静态类型转换代码
- 避免 generic runtime cast 路径
- 在 contiguous unrolled kernel 中复用静态 dtype 特化能力，减少类型转换相关开销

#### 关键性能数据

| Shape | DType | 优化前 | 优化后 | 性能提升 |
|-------|-------|--------|--------|----------|
| `(131072, 128)` | `bf16 * fp32` | `0.6 ms` | `0.15 ms` | **75.0%** |

### 完整调用链示例

以 `float + bfloat16 -> float` 为例：

```
gpu_kernel_impl
  └─ contiguous branch
       └─ try_launch_static_contiguous_unrolled_kernel<func_t, float, bfloat16, float>
            ├─ dtype 匹配检查: dtypes[1]==Float, dtypes[2]==BFloat16, dtypes[0]==Float ✓
            ├─ input_offset_calculator = TrivialOffsetCalculator<2>()
            ├─ output_offset_calculator = TrivialOffsetCalculator<1>()
            ├─ loader = memory::detail::LoadWithStaticCast<float, bfloat16>()
            ├─ storer = memory::StoreWithoutCast()
            └─ launch_unrolled_kernel(...)
```

## Feature 3：legacy nocast path 的 ldcg 优化

### 概述

Feature 3 以 `gpu_kernel_impl_nocast` 为入口，为 legacy nocast path 引入 ldcg load 优化。

当前实现主要面向 non-contiguous 场景：当 `gpu_kernel_impl` 检测到 `needs_dynamic_casting<func_t>::check(iter) == false` 时，会进入 `gpu_kernel_impl_nocast`；若 `iter.is_contiguous()` 为 `true`，则直接走 `launch_vectorized_kernel`。只有在 non-contiguous legacy path 下，且满足 ldcg 条件时，才会切换到 cached global load。

### 接入方式

Feature 3 当前已经内置在 `CUDALoops.cuh` 的 `gpu_kernel_impl_nocast` 中，不需要像 Feature 1 / Feature 2 一样额外实例化 dtype 特化。其接入方式是在 legacy path 中增加 ldcg 判定，并在命中时改为使用 `invoke<true>`：

```cpp
if constexpr (legacy_path_all_inputs_support_ldcg<traits>()) {
  bool use_ldcg = legacy_path_has_large_stride_input_impl<traits>(
      iter,
      std::make_index_sequence<traits::arity>{});
  if (use_ldcg) {
    launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      *out = invoke<true>(f, &data[1], &offsets[1], 1);
    });
    return;
  }
}
```

### 执行路径

#### `gpu_kernel_impl_nocast`

```cpp
template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f);
```

##### 功能

在 nocast 场景下，根据 contiguous / non-contiguous、ldcg 条件以及窄范围 hybrid special case 选择实际执行路径。如果命中 ldcg 条件，则在 legacy kernel 中切换到 cached global load；若 ldcg 未命中，则会再尝试一次 masked_fill 风格的 `bf16 + bool -> bf16` hybrid 特化；否则继续使用普通 load。

##### 总体流程

```
┌─────────────────────────────────────────────────────────────┐
│                   gpu_kernel_impl_nocast                    │
├─────────────────────────────────────────────────────────────┤
│ 1. 检查是否 contiguous                                       │
│    ├─ contiguous → launch_vectorized_kernel(..., tuned)     │
│    └─ non-contiguous → 进入 legacy path                     │
├─────────────────────────────────────────────────────────────┤
│ 2. 检查是否满足 ldcg 条件                                    │
│    ├─ legacy_path_all_inputs_support_ldcg<traits>()        │
│    └─ legacy_path_has_large_stride_input_impl<traits>()    │
├─────────────────────────────────────────────────────────────┤
│ 3. ldcg 命中 → invoke<true>                                  │
├─────────────────────────────────────────────────────────────┤
│ 4. ldcg 未命中时，尝试窄范围 bf16+bool hybrid special case   │
│    └─ try_launch_static_binary_kernel<bf16,bool,bf16>       │
├─────────────────────────────────────────────────────────────┤
│ 5. 仍未命中 → invoke<false> legacy kernel                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键组件

#### `load_nocast_arg_ldcg`

```cpp
template <typename traits, size_t INDEX, typename index_t>
C10_HOST_DEVICE inline typename traits::template arg<INDEX>::type load_nocast_arg_ldcg(
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i);
```

##### 功能

对单个输入参数执行 ldcg 版本的 load。

##### 作用

- 将输入地址解释为对应参数类型指针
- 通过 `memory::load_global_cached<arg_t>(ptr)` 读取数据
- 作为 Feature 3 最底层的 cached global load 组件

#### `invoke_impl_ldcg`

```cpp
template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl_ldcg(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>);
```

##### 功能

将所有输入参数的加载统一切换到 `load_nocast_arg_ldcg`，再调用 functor。

#### `invoke<true>` / `invoke<false>`

##### 功能

作为是否启用 ldcg 的统一分派点。

##### 作用

- `invoke<true>`：调用 `invoke_impl_ldcg`
- `invoke<false>`：调用普通 `invoke_impl`

#### `legacy_path_all_inputs_support_ldcg`

##### 功能

检查所有输入类型是否都支持 ldcg，且所有输入类型都不是 `bool`。

#### `input_inner_dim_byte_stride_gt_256`

##### 功能

判断某个输入的 inner dim byte stride 是否大于 256。

#### `legacy_path_has_large_stride_input_impl` / `legacy_path_use_ldcg`

##### 功能

完成运行时触发判定：只有至少一个输入满足 inner dim byte stride 大于 256 时，才会在 legacy path 中启用 ldcg。

### 性能优化要点

#### 避免 strided load 下的 L1 set 冲突

Feature 3 的核心收益来自 non-contiguous strided load 场景下的缓存访问优化：

- 当输入存在较大的 inner dim byte stride 时，普通 load 更容易受到 L1 容量和 set 冲突影响
- ldcg 路径通过 `memory::load_global_cached` 读取输入，并禁用 L1，从而避免这类 strided load 在 L1 中产生额外冲突
- 因此该优化更适合 **large stride + nocast + legacy path** 场景，而不是 contiguous fast path

| op name | shape | dtype | stride | br 平均耗时 | nv 平均耗时 | ldcg 耗时 |
|---|---|---|---|---:|---:|---:|
| `add` | `[1, 1024, 5120]` | `float32` | `a=[5242880, 1, 1024]`, `b=[5242880, 5120, 1]` | `1.235 ms` | `0.040 ms` | `0.175 ms` |
| `copy` | `[1, 1024, 5120]` | `float32` | `dst=[5242880, 5120, 1]`, `src=[5242880, 1, 1024]` | `0.594 ms` | `0.036 ms` | `0.150 ms` |
| `copy` | `[1, 2048, 16, 4, 128]` | `bfloat16` | `dst=[16777216, 8192, 512, 128, 1]`, `src=[2048, 1, 262144, 0, 2048]` | `1.456 ms` | `0.105 ms` | `0.440 ms` |


### 完整调用链示例

```
gpu_kernel_impl
  └─ gpu_kernel_impl_nocast
       ├─ contiguous ? launch_vectorized_kernel : legacy path
       └─ legacy path
            ├─ legacy_path_all_inputs_support_ldcg && legacy_path_has_large_stride_input_impl
            ├─ launch_legacy_kernel
            └─ kernel lambda 中 invoke<true>
                 └─ invoke_impl_ldcg
                      └─ load_nocast_arg_ldcg
                           └─ memory::load_global_cached
```

## Feature 4：tuned elements-per-thread 与 op 级 elementwise kernel 调优

### 概述

Feature 4 为现有 elementwise kernel 增加了 **op 级可选的 tuned elements-per-thread** 能力。当前默认路径仍由 `calc_io_size` / `elems_per_thread<io_size>()` 自动决定每线程处理元素数；如果某个算子在特定 shape / dtype / 架构下存在更优 thread-work-size，则可以在算子侧显式传入 `tuned_elems_per_thread`，复用现有 vectorized / unrolled / legacy 框架，而无需为单个算子重新实现一套 kernel。

当前已落地的首个案例是 `polar`：

- 调整文件：`torch_supa/csrc/aten/ops/kernels/elementwise/ComplexKernel.cu`
- 调优值：`polar_tuned_elems_per_thread = 8`
- 调用方式：`gpu_kernel(iter, lambda, polar_tuned_elems_per_thread)`

核心入口位于：

- `torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`
- `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`

### 接入方式

如果某个 elementwise op 希望覆写默认 `elems_per_thread`，可在 op kernel 中调用新增重载：

```cpp
constexpr int tuned_elems_per_thread = 8;
gpu_kernel(iter, [] GPU_LAMBDA(...) {
  ...
}, tuned_elems_per_thread);
```

当前支持的 tuned 值为：

- `8`
- `16`
- `24`

非法值会在 `launch_vectorized_kernel_tuned` 中通过 `TORCH_CHECK` 直接报错。

### 执行路径

#### `gpu_kernel(iter, f, tuned_elems_per_thread)`

```cpp
template <typename func_t>
void gpu_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    int tuned_elems_per_thread);
```

##### 功能

为现有 `gpu_kernel` 增加一个带 tuning 参数的重载：

- 若 `iter` 需要 dynamic casting，则仍回退到原有 `gpu_kernel_impl(iter, f)`
- 若 `iter` 不需要 dynamic casting，则进入 `gpu_kernel_impl_nocast_tuned`
- 对大 tensor 仍保留 `with_32bit_indexing()` 拆分逻辑

##### 总体流程

```
gpu_kernel(iter, f, tuned)
  ├─ iter.numel() == 0 → return
  ├─ !iter.can_use_32bit_indexing() → 递归拆分 sub_iter
  ├─ needs_dynamic_casting<func_t>::check(iter) == false
  │    └─ gpu_kernel_impl_nocast_tuned(iter, f, tuned)
  └─ 否则
       └─ gpu_kernel_impl(iter, f)
```

#### `gpu_kernel_impl_nocast_tuned`

```cpp
template <typename func_t>
void gpu_kernel_impl_nocast_tuned(
    TensorIteratorBase& iter,
    const func_t& f,
    int tuned_elems_per_thread);
```

##### 功能

为 `nocast` 路径增加 tuned thread-work-size 支持。

##### 行为

- contiguous 场景：调用 `launch_vectorized_kernel(numel, f, data, tuned_elems_per_thread)`
- non-contiguous 场景：继续复用现有 legacy kernel + ldcg 判断逻辑
- 因此该 feature 主要调整的是 **contiguous nocast fast path** 的 vectorized / tail 行为，不会改变 non-contiguous 的主干逻辑

#### `launch_vectorized_kernel(..., tuned_elems_per_thread)`

```cpp
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    int tuned_elems_per_thread);
```

##### 功能

先尝试命中 tuned vectorized kernel；若未命中，则回退到默认 `elems_per_thread<io_size>()` 路径。

##### 总体流程

```
launch_vectorized_kernel(N, f, data, tuned)
  ├─ tuned > 0 && launch_vectorized_kernel_tuned(...) == true
  │    └─ 直接返回
  └─ 否则
       └─ 继续执行原有默认 vectorized / unrolled 路径
```

#### `launch_vectorized_kernel_tuned`

```cpp
template <typename func_t, typename array_t>
static inline bool launch_vectorized_kernel_tuned(
    int64_t N,
    const func_t& f,
    array_t data,
    int tuned_elems_per_thread);
```

##### 功能

根据：

- `tuned_elems_per_thread`
- `memory::can_vectorize_up_to<func_t>(data)`
- 输出类型大小
- device compute capability

选择合适的模板实例并启动 `vectorized_elementwise_kernel_tuned`。

##### 关键约束

1. `tuned_elems_per_thread` 目前只允许 `{8, 16, 24}`
2. `vec_size` 仍受现有向量化上限约束
3. 非 `sm90/sm100` 架构下，`vec_size` 会被限制到 `<= 4`
4. 若输出类型字节数 `< 2`，`vec_size` 也会被限制到 `<= 4`
5. `tuned=24` 当前只对 `vec_size=4/2` 提供实例，不走 `vec8`

##### 返回值

- `true`：已成功选择并启动 tuned kernel
- `false`：当前组合不支持 tuned vectorization，调用方应回退默认路径

#### `vectorized_elementwise_kernel_tuned`

```cpp
template <int vec_size, int tuned_elems_per_thread, typename func_t, typename array_t>
__global__ void vectorized_elementwise_kernel_tuned(int N, func_t f, array_t data);
```

##### 功能

与现有 `vectorized_elementwise_kernel` 保持同样的两段式结构：

- **full block**：走 `memory::policies::vectorized<vec_size, array_t, tuned_elems_per_thread>`
- **tail block**：走 `memory::policies::unroll<..., tuned_elems_per_thread>`

区别在于 `elems_per_thread` 不再由 `io_size` 自动推导，而是由 op 显式指定。

### `polar` 接入示例

`ComplexKernel.cu` 中新增了 `polar_kernel_cuda`：

```cpp
void polar_kernel_cuda(TensorIterator& iter) {
  constexpr int polar_tuned_elems_per_thread = 8;
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar_cuda", [&]() {
    gpu_kernel(
      iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
      }, polar_tuned_elems_per_thread);
  });
}
```

并通过：

```cpp
REGISTER_PRIVATEUSE1_DISPATCH(polar_stub, &polar_kernel_cuda)
```

完成注册。

### `polar` 调参性能

当前 `polar` 的 tuned elements-per-thread 实测结果为：

| elems_per_thread | 耗时 |
|---|---:|
| `8` | `0.0748 ms` |
| `16` | `0.128 ms` |
| `24` | `0.112 ms` |

可以看到 `elems_per_thread = 8` 是当前最优配置，因此 `ComplexKernel.cu` 最终选择：

```cpp
constexpr int polar_tuned_elems_per_thread = 8;
```

### 完整调用链示例

以 `polar` 为例：

```
polar_stub
  └─ polar_kernel_cuda
       └─ gpu_kernel(iter, lambda, 8)
            ├─ gpu_kernel_impl_nocast_tuned
            │    ├─ contiguous
            │    │    └─ launch_vectorized_kernel(numel, f, data, 8)
            │    │         ├─ launch_vectorized_kernel_tuned(..., 8)
            │    │         │    └─ vectorized_elementwise_kernel_tuned<vec_size, 8>
            │    │         └─ 若未命中则 fallback 到默认 launch_vectorized_kernel
            │    └─ non-contiguous
            │         └─ legacy kernel / ldcg 路径
            └─ 若需要 dynamic casting，则回退 gpu_kernel_impl
```

### 适用场景

适合以下需求：

- 某个已接入的 elementwise op 在默认 `elems_per_thread` 下性能不理想
- 问题主要集中在 contiguous nocast fast path
- 希望做 **算子级小范围调参**，而不是引入新的 memory policy 或新的主分流节点

### 开发约束

1. **优先做 op 级调参，不要先改全局 heuristics**
   - 只有在多个算子表现出一致规律时，才考虑修改 `elems_per_thread<io_size>()`
2. **优先复用现有 vectorized / unroll / legacy 框架**
   - 不要因为某个 op 想改 thread-work-size 就复制一整套 kernel
3. **仅在 nocast contiguous 路径上生效最稳定**
   - dynamic casting 路径当前不会使用 tuned 参数
4. **调优值应配合 benchmark 验证**
   - 至少验证目标 shape、相邻 shape、不同 dtype 与不同架构是否出现退化

### 与前三个 feature 的关系

- **Feature 1 / 2 / 3** 分别关注 static cast + hybrid vectorized、contiguous unrolled static cast、legacy nocast ldcg
- **Feature 4** 关注的是“在既有实现框架内，为单个 op 显式指定 thread-work-size”

因此 Feature 4 更像是一个 **op-level tuning hook**，不是新的 dtype 特化或新的 stride 分流框架。
