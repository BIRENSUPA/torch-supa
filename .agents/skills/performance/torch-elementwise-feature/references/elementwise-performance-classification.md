# Elementwise Performance Optimization Classification and Landing Skill

This skill handles user-provided tests, benchmarks, op names, kernel names, or source files. It determines whether the target belongs to an elementwise op, then checks whether the current general elementwise optimizations already cover it based on shape / stride / dtype. If not covered, guide the work by scenario:

1. Operator not migrated: perform minimal CUDA → SUPA adaptation using `new-operator/torch-elementwise-adaptation/SKILL.md`.
2. Operator already migrated and can reuse general optimizations: extend the existing feature using `performance/torch-elementwise-feature/SKILL.md`.
3. Existing features cannot cover the case: add a new optimization feature and include validation plus performance data.

## Applicable inputs

The user may provide any one or more of the following:

- pytest case, for example `test_xxx.py::TestFoo::test_bar`
- benchmark script or minimal repro
- op name, for example `add`, `mul`, `div`, `copy_`, `where`
- kernel name, for example `elementwise_kernel`, `unrolled_elementwise_kernel`
- CUDA kernel entry, for example `mul_kernel_cuda`
- source file path or target file path
- shape / stride / dtype description

If shape / stride / dtype are incomplete, extract them from the test code first. If they still cannot be determined, ask the user to run a minimal print script to provide them:

```python
print("shape", t.shape)
print("stride", t.stride())
print("dtype", t.dtype)
print("contiguous", t.is_contiguous())
```

## Overall workflow

```text
Input test / op / kernel / file
  ├─ Step 1: Determine whether it is elementwise
  │    ├─ No → output "not applicable" and stop
  │    └─ Yes / high probability → continue
  ├─ Step 2: Extract op, shape, stride, dtype, contiguous, broadcast, dynamic casting
  ├─ Step 3: Match existing general optimization coverage
  ├─ Step 4: Classify the action
  │    ├─ Covered → compile/run/confirm hit and output performance
  │    ├─ op not migrated → use torch-elementwise-adaptation to migrate it
  │    ├─ op migrated but pattern not covered → use torch-elementwise-feature to extend an existing path
  │    └─ existing features cannot cover it → add a new feature and update documentation
  └─ Step 5: Compile, run correctness tests, and compare performance before and after optimization
```

---

## Step 1: Determine whether it is an elementwise op

First determine whether the given case is elementwise. Do not directly perform elementwise optimization before confirming that it is elementwise.

### 1.1 Code and kernel evidence

Prefer code evidence for the decision:

- Kernel name contains any of these patterns:
  - `elementwise_kernel`
  - `unrolled_elementwise_kernel`
  - `vectorized_elementwise_kernel`
  - `hybrid_vectorized_elementwise_kernel`
- Call chain enters:
  - `gpu_kernel_impl`
  - `gpu_kernel_impl_nocast`
  - `launch_vectorized_kernel`
  - `launch_unrolled_kernel`
  - `launch_legacy_kernel`
  - `try_launch_static_binary_kernel`
  - `try_launch_static_contiguous_unrolled_kernel`
- Source file is located in:
  - `torch_supa/csrc/aten/ops/kernels/elementwise/`
  - CUDA elementwise kernel under `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/`
- Uses TensorIterator / TensorIteratorBase to organize element-by-element computation.

### 1.2 Common elementwise op name judgment

If the user only provides an op name or test name, common op names can also be used for an initial judgment.

Common binary / unary elementwise ops include but are not limited to:

```text
add, sub, mul, div, remainder, fmod, pow, rpow,
maximum, minimum, clamp, where,
eq, ne, lt, le, gt, ge,
logical_and, logical_or, logical_xor,
bitwise_and, bitwise_or, bitwise_xor,
abs, neg, reciprocal, sqrt, rsqrt, exp, exp2, log, log2, log10,
sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
sigmoid, erf, erfc, floor, ceil, round, trunc,
copy, copy_, to, masked_fill, fill
```

Note: op name can only be used as an initial judgment. The final decision should, when possible, be confirmed with kernel name, TensorIterator call chain, source path, or profiler results.

### 1.3 Use Claude AI as auxiliary judgment

When the op name is atypical, the test name is unclear, or the code path is complex, Claude AI can be used to help determine whether it is elementwise.

Recommended prompt:

```text
Please determine whether the following PyTorch test/operator is an elementwise op.
Base the decision on op semantics, input/output shape relationships, whether it computes element by element, and whether it may use TensorIterator.
Only output: elementwise / not elementwise / uncertain, and explain the key evidence.

op/test/kernel:
<user input>

Relevant code snippet or profiler kernel name:
<pasted content>
```

Requirements after using Claude AI for judgment:

- If Claude AI outputs `elementwise`, still prefer a second confirmation using local code or profiler kernel name.
- If Claude AI outputs `uncertain`, do not optimize directly. First provide source code, kernel name, or shape/stride/dtype information.
- If Claude AI outputs `not elementwise`, stop this skill unless local kernel evidence clearly proves otherwise.

### 1.4 Classification output

Step 1 output must be one of the following:

```text
A. Confirmed elementwise
B. High probability elementwise, requires profiler/kernel name confirmation
C. Uncertain, more information needed
D. Not elementwise, this skill is not applicable
```

---

## Step 2: Extract shape / stride / dtype

Extract the following for each input and output:

- op name
- input shape / output shape
- input stride / output stride
- input dtype / output dtype
- whether it is contiguous
- whether it uses broadcast
- whether it is binary elementwise
- whether it has `needs_dynamic_casting`
- whether a scalar operand exists
- whether large stride exists, especially whether the inner-dim byte stride is greater than 256

Recommended record format:

```markdown
| tensor | shape | stride | dtype | contiguous | role |
|---|---|---|---|---|---|
| out | ... | ... | ... | ... | output |
| input0 | ... | ... | ... | ... | input |
| input1 | ... | ... | ... | ... | input |
```

---

## Step 3: Match existing general optimization coverage

The current general elementwise optimizations mainly fall into three categories.

### 3.1 Feature 1: non-contiguous static cast + hybrid vectorized

Entry: `try_launch_static_binary_kernel`

Applicable conditions:

- binary elementwise
- dynamic casting or mixed dtype scenario exists
- runtime dtype matches template instantiation dtype
- the element stride of at least one input matches the output stride
- the other input can be broadcast or non-contiguous

Typical verified cases:

| op | shape pattern | dtype | optimization path | conclusion |
|---|---|---|---|---|
| binary elementwise, representative op is mul | `(1,2048,24,64) * (1,2048,1,64)` | `bf16 + fp32 -> fp32` / `fp32 + bf16 -> fp32` | static cast + hybrid vectorized | V3 average is better than base |
| binary elementwise, representative op is mul | `(1,1024,5120) * (1,1,5120)` | `bf16 + fp32 -> fp32` / `fp32 + bf16 -> fp32` | static cast + hybrid vectorized | 16/8 element_per_thread is more stable than 32 |
| binary elementwise, representative op is mul | `(594,4096) * (594,1)` | `bf16 + fp32 -> fp32` / `fp32 + bf16 -> fp32` | static cast + hybrid vectorized | stable gain |
| binary elementwise, representative op is mul | `(647,4096) * (647,1)` | `bf16 + fp32 -> fp32` / `fp32 + bf16 -> fp32` | static cast + hybrid vectorized | stable gain |
| binary elementwise, representative op is mul | `(679,4096) * (679,1)` | `bf16 + fp32 -> fp32` / `fp32 + bf16 -> fp32` | static cast + hybrid vectorized | stable gain |

Known experience:

- static cast is a stable gain item.
- hybrid vectorized is the main optimization direction for binary mixed dtype.
- `element_per_thread = 16/8` is more stable than `32`; special broadcast shapes need focused regression.

### 3.2 Feature 2: contiguous unrolled static cast

Entry: `try_launch_static_contiguous_unrolled_kernel`

Applicable conditions:

- contiguous elementwise
- dynamic casting or mixed dtype scenario
- dtype combination hits template instantiation
- unrolled kernel can execute

Typical verified case:

| op | shape | dtype | optimization path | conclusion |
|---|---|---|---|---|
| binary elementwise | `(131072, 128)` | `bf16 * fp32` | contiguous unrolled static cast | clear gain observed |

Known experience:

- Current main gains come from static cast.
- Further stacking burst/vectorized may improve performance, but increases template instantiation and branch expansion.

### 3.3 Feature 3: legacy nocast + ldcg

Entry: non-contiguous legacy path in `gpu_kernel_impl_nocast`

Applicable conditions:

- `needs_dynamic_casting == false`
- non-contiguous legacy path
- functor input types support ldcg and do not include bool
- at least one input has large inner-dim byte stride; the current key condition is inner dim byte stride `> 256`

Typical verified cases:

| op | shape | dtype | stride | optimization path | conclusion |
|---|---|---|---|---|---|
| add | `[1,1024,5120]` | `float32` | `a=[5242880,1,1024]`, `b=[5242880,5120,1]` | legacy nocast + ldcg | strided load clearly improved |
| copy | `[1,1024,5120]` | `float32` | `dst=[5242880,5120,1]`, `src=[5242880,1,1024]` | legacy nocast + ldcg | strided load clearly improved |
| copy | `[1,2048,16,4,128]` | `bfloat16` | `dst=[16777216,8192,512,128,1]`, `src=[2048,1,262144,0,2048]` | legacy nocast + ldcg | strided load clearly improved |

Known experience:

- ldcg targets only large stride + nocast + legacy path.
- It is not a contiguous fast path and not a dynamic casting optimization.
- The optimization core is changing strided loads that should not stay in L1 into cached global loads.

---

## Step 4: Classify the action

Based on Steps 1-3, output one of the following classifications.

### A. Covered by general optimizations

Conditions:

- elementwise has been confirmed
- op is already registered in `torch_supa` or can already enter the current general elementwise path
- shape / stride / dtype matches Feature 1 / 2 / 3 conditions

Actions:

- compile and run tests
- use profiler to confirm whether the target kernel is hit
- output performance comparison before and after optimization

### B. It is elementwise, but the op has not been migrated

Conditions:

- op semantics and code path are elementwise
- but there is no corresponding implementation under `torch_supa/csrc/aten/ops/kernels/elementwise/` or no `REGISTER_PRIVATEUSE1_DISPATCH`

Actions:

- switch to `new-operator/torch-elementwise-adaptation/SKILL.md`
- locate the source CUDA kernel under `third-party/torch-supa/build/_deps/pytorch-src/aten/src/ATen/native/cuda/`
- migrate it to `torch_supa/csrc/aten/ops/kernels/elementwise/`
- perform only the minimal necessary replacements: include, guard, namespace, launch check, dispatch registration
- do not rewrite the main logic

### C. op has been migrated, but dtype / stride pattern is not covered

Conditions:

- op already has a `torch_supa` elementwise implementation
- but the dtype combination or stride/broadcast pattern does not hit existing optimizations
- existing static cast / hybrid vectorized / contiguous unrolled / ldcg infrastructure can be reused

Actions:

- switch to `performance/torch-elementwise-feature/SKILL.md`
- prefer extending existing paths instead of adding an independent framework
- possible files to modify:
  - `torch_supa/csrc/aten/ops/kernels/elementwise/CUDALoops.cuh`
  - `torch_supa/csrc/aten/ops/kernels/elementwise/Loops.cuh`
  - `torch_supa/csrc/aten/ops/kernels/elementwise/BinaryInternal.h`

Common changes:

- add a new `try_launch_static_binary_kernel<...>` dtype combination
- add a new `try_launch_static_contiguous_unrolled_kernel<...>` dtype combination
- extend hybrid vectorized stride/broadcast checks
- adjust `element_per_thread`
- connect ldcg for large-stride nocast scenarios

### D. Existing features cannot cover it; a new feature is needed

Conditions:

- it is elementwise
- none of the three existing feature categories can reasonably cover it
- clear performance data proves the bottleneck

Actions:

- first decide where the new feature should land:
  - `needs_dynamic_casting = false`
  - `contiguous + dynamic_casting`
  - `non-contiguous + dynamic_casting`
- prefer attaching it to existing vectorized / unrolled / legacy / static-cast branches
- add a new branch node only if the existing framework cannot support it
- update:
  - `performance/torch-elementwise-feature/references/elementwise-new-feature.md`
  - `performance/torch-elementwise-feature/references/elementwise-development-summary.md`

### E. Not elementwise; not applicable

Conditions:

- op is not element-by-element computation
- or the main bottleneck comes from non-elementwise paths such as reduction, matmul, conv, sort, scan, indexing, communication, or memory allocation

Actions:

- stop this skill
- output the reason it is not applicable and the recommended analysis direction

---

## Step 5: Compile, correctness tests, and performance comparison

Any real migration or optimization must be validated.

### 5.1 Compile check

First check the existing build method in the current repository. Do not hardcode commands before confirming.

Checkpoints:

- whether added or modified `.cu` files enter `ATEN_SUDA_SRCS`
- whether they are compiled into `LIBSUDAOP`
- whether there are symbol conflicts
- whether `REGISTER_PRIVATEUSE1_DISPATCH` takes effect
- whether hidden/local visibility and required exports match existing constraints

### 5.2 Correctness test

Prefer running the user-provided test:

```text
pytest <user-provided-test>
```

If the user did not provide a test, construct a minimal repro covering:

- shape
- stride
- dtype
- broadcast
- contiguous / non-contiguous
- CPU / CUDA / SUPA result consistency

### 5.3 Performance test

Compare two data groups before and after optimization:

- baseline: before modification, with the new path disabled, or original kernel
- optimized: after modification, hitting the new path

Record:

- op
- shape
- stride
- dtype
- baseline time
- optimized time
- speedup
- kernel before
- kernel after
- whether static cast / hybrid vectorized / contiguous unrolled / ldcg was hit

Recommended output table:

```markdown
| case | op | shape | stride | dtype | baseline | optimized | speedup | kernel before | kernel after | optimization hit |
|---|---|---|---|---|---:|---:|---:|---|---|---|
```

---

## Final output format

After executing this skill, the final response should include:

```markdown
## Judgment Result

- Is elementwise:
- Evidence:
- op:
- shape:
- stride:
- dtype:
- classification:

## General Optimization Match

- Matched optimization path:
- Conditions not matched:
- Evidence files/functions:

## Suggested Action

- Action type: covered / migrate operator / extend existing feature / add new feature / not applicable
- Files to modify:
- Key changes:

## Validation Result

- Compile:
- Correctness:
- profiler/kernel hit:

## Performance Comparison

| case | baseline | optimized | speedup | kernel | notes |
|---|---:|---:|---:|---|---|
```

## Key principles

- Determine whether it is elementwise before judging optimization.
- op name, common op lists, and Claude AI judgment are only auxiliary; prefer source code, TensorIterator call chain, and profiler kernel name as final evidence.
- First determine whether shape / stride / dtype hits existing general optimizations.
- If only the op is not migrated, use `torch-elementwise-adaptation`; do not rewrite the main logic during migration.
- If the op has been migrated but the pattern is not covered, use `torch-elementwise-feature` to extend an existing path.
- Add a new feature only when the existing framework cannot cover it and performance data supports the need.
- Every real change must include compile validation, correctness validation, and before/after performance comparison.
