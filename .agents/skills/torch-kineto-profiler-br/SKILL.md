---
name: torch-kineto-profiler-br
description: 处理 br_pytorch2 中与 torch profiler、Kineto、SUPTI、GPU trace、graph activity、driver activity、with_stack、profile 缺事件、profiling 崩溃和 profiling 接入路径相关的问题。适用于用户提到 profiler、kineto、supti、trace、activity、graph、with_stack、profile 结果异常、缺 kernel 或相关代码概述的场景。
---

# torch-kineto-profiler-br

用于理解、解释和排查本仓库中与 profiling 相关的实现，重点覆盖 torch profiler、Kineto 子模块、SUPTI activity、GPU trace、graph activity、driver activity、with_stack 以及 profiling 接入路径。

## 适用场景

当用户的问题涉及以下任一主题时使用此 Skill：

- `torch.profiler` 或 torch profiler 行为异常
- Kineto 集成、kineto 子模块升级或适配
- SUPTI activity、callback、graph activity、driver activity
- GPU trace、CPU op profile、`with_stack`
- profile 结果缺事件、trace 不完整、profiling 报错或崩溃
- profiling 相关代码路径概述、模块定位、调用链说明

不要用于与 profiler 基础设施无关的通用性能优化问题。

## 核心排查路径

默认按下面顺序排查，不要一上来就分散到很多方向：

1. **确认 Python 入口是否真的开了目标 profiling 能力**
   - 用户请求的是 CPU、SUPA、driver activity、graph activity，还是 `with_stack`
   - 不要把 Python 层的开关当成底层已经生效

2. **检查 Kineto shim 是否真的把目标 activity 传下去**
   - 重点看 activity 集合是怎么组装的
   - 先查 `kCpuTypes` / `kSupaTypes`
   - 再查 `prepareTrace()` 是否真的把目标 activity 传给 `activityProfiler().prepareTrace(...)`

3. **检查 SUPTI enable/disable 映射是否接通**
   - 重点确认目标 activity 是否被映射到对应的底层 activity kind
   - 不要只看 activity 名字存在，就默认底层真的采到了

4. **最后再看 trace 汇总、导出和展示**
   - 先判断是“没采到”，还是“采到了但结果不对”
   - 不要过早把问题归因到前端展示或 trace export

## 代码理解顺序

如果用户要代码概述，按下面顺序解释：

1. 用户侧 profiler 入口
2. torch 桥接或分发层
3. Kineto 接入点
4. SUPTI / driver activity 注册与采集
5. trace 汇总、导出或展示层

解释时尽量锚定到具体仓库文件，而不是只讲抽象概念。

## 关键目录与文件

```text
torch_supa/
├── profiler/
│   ├── profiler.py                  # Python 侧 torch.profiler 包装入口
│   ├── profiler_kineto.py           # Python 侧 Kineto 生命周期：prepare/start/stop
│   └── profiler_util.py             # event 后处理、table/trace 辅助逻辑
├── csrc/
│   └── profiler/
│       ├── kineto_shim.cpp          # activity 集合组装、prepareTrace、下发到 Kineto
│       ├── profiler_kineto.cpp      # C++ profiler 主 glue code
│       ├── collection.cpp           # event 收集、结果汇总、with_stack 高频落点
│       ├── profiler_python.cpp      # Python/C++ profiler 绑定层
│       ├── init.cpp                 # profiler 初始化入口
│       └── CMakeLists.txt           # profiler 构建接入
└── supa/
    └── profiler.py                  # torch_supa.supa.profiler 入口

third-party/
└── kineto/                          # Kineto 子模块；activity/callback/driver/graph 问题重点检查
```

建议阅读顺序：

- **先看入口**：`torch_supa/profiler/profiler.py`、`torch_supa/profiler/profiler_kineto.py`
- **再看 activity 下发**：`torch_supa/csrc/profiler/kineto_shim.cpp`
- **再看结果汇总**：`torch_supa/csrc/profiler/collection.cpp`
- **最后看子模块**：`third-party/kineto`

如果只是排 activity 缺失，优先顺序通常是：

`profiler.py / profiler_kineto.py` → `kineto_shim.cpp` → `collection.cpp` → `third-party/kineto`

## 关键知识点

### 1. `KINETO_LOG_LEVEL` 很实用

Kineto/Libkineto 支持用环境变量打开日志，初始化时会读取 `KINETO_LOG_LEVEL`。

当怀疑 profiling 适配有问题、activity 没抓到、或者 trace 路径没走通时，优先建议打开这个环境变量辅助定位。

如果还需要更细粒度的日志，可再结合：

- `VERBOSE_LOG_LEVEL`
- `VERBOSE_LOG_MODULES`

但在这个仓库里，日常排查优先用 `KINETO_LOG_LEVEL`，不要先把重点放在 Kineto 配置文件流上。

### 2. graph activity 不等于 graph 内 kernel

当前仓库里 `GPU_GRAPH` 是单独的 activity 类型：

- shim 会把它放进 SUPA 侧 activity 集合
- SUPTI 侧会把它映射到 `GRAPH_TRACE`

但要特别注意：

- **看到了 graph activity，不等于已经抓到了 graph 内 kernel 明细**
- **CUPTI 可以直接抓 kernel**
- **SUPTI 目前暂时不支持把 graph 内 kernel 明细直接抓出来**

所以如果用户看到 graph trace 里有 graph、但没有 graph 内 kernel，不要直接判断成普通 trace 丢失。

### 3. activity 集合是排查第一现场

在这个仓库里，SUPA profiling 的 activity 选择不是抽象概念，而是明确在 Kineto shim 中组装的。

分析缺事件时，优先检查：

- `kCpuTypes` 是否包含目标 activity
- `kSupaTypes` 是否包含目标 activity
- `prepareTrace()` 是否真的把目标 activity 传进了 `activityProfiler().prepareTrace(...)`

#### 适配 activity 的最小例子

假设现在要新增一个 activity，比如 `GPU_GRAPH`，最小适配思路通常不是只改一个地方，而是至少顺着下面几层一起看：

1. **先让 Kineto shim 把它放进 activity 集合**
   - 例如在 `kSupaTypes` 里加入 `libkineto::ActivityType::GPU_GRAPH`
   - 否则即使底层支持，这个 activity 也不会进入 `prepareTrace()`

2. **确认底层 SUPTI enable/disable 路径有映射**
   - 例如把 `GPU_GRAPH` 映射到对应的 `GRAPH_TRACE` kind
   - 否则上层名字存在，但底层不会真的开始采集

3. **确认结果汇总层认识这个 activity**
   - 比如 device type 分类、trace event 转换、table 展示是否需要同步
   - 否则可能出现“采到了但最后结果不对”

4. **最后补最小验证**
   - 跑最小 `torch.profiler` 用例
   - 看 trace 里是否真的出现目标 activity
   - 必要时打开 `KINETO_LOG_LEVEL`

可以把它理解成一条固定链路：

`activity 名称/枚举` → `shim activity 集合` → `SUPTI enable/disable` → `结果汇总/展示`

只改其中一个点，通常不够。

### 4. `with_stack`、driver activity、callback ID 都是易碎点

这些能力通常和版本强耦合：

- `with_stack` 可能因为版本 API 变化失效
- driver activity 可能在采集层支持了，但导出或解析层没跟上
- callback ID 在版本升级后经常需要同步更新
- profiler 升级有时不是单纯改 Python API，还会连带改 `collection.cpp` 这类结果汇总逻辑

遇到这类问题时，不要只看单点文件，要联动看 shim、Kineto 和底层 activity 定义。

另外，从历史改动看，Kineto 相关问题很常伴随子模块升级；如果某个 profiling 能力突然失效，优先确认是不是需要同步更新 `third-party/kineto`，而不是先怀疑业务代码。

## 常见误判

重点避免以下误判：

- Python 层开了 activity，等于 Kineto 一定收到了
- Kineto 能识别 activity 名称，等于 SUPTI 一定真的启用了采集
- trace 不完整，等于一定是展示层问题
- 有 graph activity，等于 graph 内 kernel 已经抓到了
- profile 结果异常，等于一定是上层接口逻辑有问题

## 常见问题模式

重点关注以下几类高频问题：

- activity 枚举新增或变更，但注册路径没同步
- callback ID 在版本升级后变化
- Kineto 子模块升级后，本地适配没有同步更新
- driver activity 在采集层支持了，但导出或解析层没跟上
- `with_stack` 或 CPU op profile 因版本 API 变化失效
- trace 看起来不完整，其实是构建配置或 activity 开关没打开
- Python 层选择了 SUPA profiling，但 Kineto shim 没把对应 activity 放进 `prepareTrace()` 的 activity 集合
- Kineto 能识别 activity 名称，但 SUPTI enable/disable 路径没有同步接上，导致底层没真正采集
- graph activity 出现了，但 graph 内 kernel 明细并没有被 SUPTI 真正展开抓到
- 升级 torch / Kineto 后，结果汇总层或 post-process 逻辑没有同步，导致事件其实采到了，但最终 trace 或 table 不对
- profiler 相关问题表面像功能 bug，实际是子模块版本、CMake 链接或 profiler glue code 没同步

## 修复建议

如果用户要修复建议：

- 优先给最小改动方案
- 明确指出问题更像在 torch 接线、Kineto 同步、SUPTI 定义，还是 trace 汇总层
- 明确说明要补回归的 profiling 场景
- 如果改动点涉及 `third-party/kineto`，默认把它当成一个独立排查方向，不要假设上层 glue code 一定没问题
- 如果问题发生在版本升级后，优先把“子模块升级 + glue code 同步 + 汇总层兼容”一起检查

## 输出格式

默认按下面结构组织回答：

- 现象
- 根因
- 相关模块
- 建议修复方向或下一步检查点
- 验证方式

如果用户要的是代码概述而不是问题排查，则将“根因”替换为“调用链”。

## 验证建议

根据问题类型，优先建议以下验证方式：

- 跑一个最小 `torch.profiler` 用例，确认关键事件是否出现
- 对比 CPU-only 与启用 GPU trace 的结果差异
- 检查 exported trace 中是否包含 driver activity 或 graph activity
- 验证 `with_stack` 输出是否包含预期堆栈
- 打开 `KINETO_LOG_LEVEL`，必要时再结合 `VERBOSE_LOG_LEVEL` 与 `VERBOSE_LOG_MODULES`，观察 Kineto 初始化、activity 注册和 trace 准备过程
- 如果怀疑是底层 SUPTI 开关问题，继续核对目标 activity 是否在 shim 的 activity 集合中出现，以及是否在 SUPTI enable/disable 路径中被映射到对应的 activity kind
- 如果问题与 graph 相关，要单独区分“抓到了 graph activity”与“抓到了 graph 内 kernel”
- 如果问题发生在升级后，补看相关子模块 commit 和 profiler glue code 改动，不要只靠当前代码静态判断

## 可参考的历史主题

搜索历史修改时，可优先关注这些主题：

- support profiler
- support gpu trace
- update submodule kineto
- support driver activity
- support supti graph activity
- update supti callback id
- support triton cpu op profile
- fix with_stack trace

## 依赖

此 Skill 常用以下能力：

- Read
- Bash
- `rg` / `git grep`
- `git log`

分析时优先依赖当前仓库实现和历史修改，不要凭印象判断。
