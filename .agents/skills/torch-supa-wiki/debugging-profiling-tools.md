# 调试、Profiling 和工具

## 环境变量

`torch_supa` 会在进程启动时读取运行时环境变量。需要调试时，应在 import `torch` 之前设置。

常用变量：

| 变量 | 用途 |
| --- | --- |
| `BRTB_LOG_LEVEL` | 日志级别，例如 `info`、`warning`、`error`。 |
| `BRTB_LOG_BACKEND` | 日志后端，例如示例中的 `glog` 或 `stdout`。 |
| `BRTB_LOG_DIR` | 日志输出目录，用户指南中的默认值为 `./logs`。 |
| `BRTB_ENABLE_REALTIME_LOG` | 实时打印日志，便于调试。 |
| `BRTB_ENABLE_SIGNAL_HANDLING` | 进程收到 `SIGSEGV`、`SIGABRT` 等信号时输出诊断信息。 |
| `BRTB_ENABLE_FLUSH_LOG_INSTANTLY` | 每条日志立即 flush；利于调试，但生产环境会更慢。 |
| `BRTB_SUBLAS_PREFERRED_BACKEND` | 选择 `Sublas` 或 `Sublaslt`。 |
| `BRTB_ENABLE_NATIVE_OP` | 优先使用 native operator 实现，常用于兼容性排查。 |
| `BRTB_TRANSFER_SILENCE` | 静默 `transfer_to_supa` 日志。 |

快速调试示例：

```bash
export BRTB_LOG_LEVEL=info
export BRTB_LOG_BACKEND=stdout
export BRTB_ENABLE_REALTIME_LOG=true
python your_script.py
```

来源：`docs/user_guider/source/chapter06_torch_supa_envconfig.rst`。

## 常用工具

### 环境信息采集

```python
import torch_supa.utils.collect_env as collect_env
collect_env.get_pretty_env_info()
```

输出内容包括 PyTorch 版本、`torch_supa` 版本、Python 版本、OS、SUPA driver 版本和设备信息。

### 精度对比

用户指南描述了精度对比工具，可比较 CPU/GPU/CUDA 输出和 SUPA 输出，覆盖 FP32、FP16、BF16，并给出 max error 和 average error 等统计。

### Profiler

可使用 PyTorch profiler 的 `ProfilerActivity.SUPA` 捕获 SUPA activity，并导出 Chrome trace。最小示例见 `runtime-features-apis.md`。

来源：`docs/user_guider/source/chapter07_torch_supa_tools.rst`。

## Profiler 排障路径

Kineto profiler skill 推荐按下面顺序排查缺事件或事件错误：

1. 确认 Python profiler 入口真的启用了目标能力，例如 CPU、SUPA、driver activity、graph activity 或 `with_stack`。
2. 检查 Kineto shim 是否把 activity 传给了 `prepareTrace()`。
3. 检查 SUPTI enable/disable 是否把 activity 映射到底层 activity kind。
4. 最后再检查 trace 汇总、导出和展示。

有用的 profiler 调试环境变量：

```bash
export KINETO_LOG_LEVEL=1
```

必要时可配合 `VERBOSE_LOG_LEVEL` 和 `VERBOSE_LOG_MODULES`。

注意：看到 graph activity 不等于已经采集到 graph 内部 kernel 明细。

来源：`.agents/skills/torch-kineto-profiler-br/SKILL.md`。

## FAQ 和故障排查

用户指南有专门的 FAQ 页面：`docs/user_guider/source/chapter08_torch_supa_faq.rst`。用户可见的已知问题优先看 FAQ；实现级排查仍应以当前源码为准。
