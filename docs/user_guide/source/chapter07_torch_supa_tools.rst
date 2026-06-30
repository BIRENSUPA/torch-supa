常用工具
==============================================================

本章介绍 torch_supa 提供的常用工具，帮助您更好地进行开发、调试和性能优化。

环境信息收集
---------------

torch_supa 提供了环境信息收集工具，用于诊断环境配置问题。建议在提交问题或联系技术支持前收集该信息。

.. code-block:: python

   import torch_supa.utils.collect_env as collect_env

   # 打印环境信息
   print(collect_env.get_pretty_env_info())

**输出信息包括**：

- PyTorch 版本
- torch_supa 版本
- Python 版本
- 操作系统信息
- SUPA 驱动版本
- 设备信息

这些信息可帮助判断版本是否匹配、驱动和运行环境是否正确加载，以及设备是否可见。

性能分析工具
---------------

torch_supa 支持 PyTorch Profiler 进行性能分析，详见 :ref:`profiler-tool` 章节。建议在确认模型可以正常运行后，再使用 Profiler 定位算子耗时、内存使用和数据加载瓶颈。

日志调试
---------------

torch_supa 提供了丰富的日志配置选项，详见 :ref:`环境变量 <BRTB_LOG_LEVEL>` 章节。日志调试常用于以下场景：

- 安装验证失败或导入 ``torch`` 失败
- 算子兼容性问题定位
- 分布式初始化失败定位
- 性能分析前收集运行环境和关键日志

**快速启用调试日志**：

.. code-block:: shell

   # 启用详细日志输出
   export BRTB_LOG_LEVEL=info
   export BRTB_LOG_BACKEND=stdout
   export BRTB_ENABLE_REALTIME_LOG=true

   python your_script.py
