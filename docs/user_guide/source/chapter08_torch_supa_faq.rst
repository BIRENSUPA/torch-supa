FAQ
==============================================================

本章汇总了 torch_supa 使用过程中的常见问题及解决方案。排查问题时，建议同时记录 PyTorch、torch_supa、SUPA SDK、驱动、操作系统和硬件型号等环境信息，便于定位版本配套问题。

编译安装
----------

.. rubric:: Q: 编译或运行时出现动态库找不到错误

**症状**

.. code-block:: text

   ImportError: libcu**.so*: cannot open shared object file: No such file or directory

**可能原因**

- SUPA SDK 或 SUDA 依赖未正确安装。
- 未加载 SDK 环境变量。
- 当前 Python 环境中的 torch_supa 与 SDK 版本不匹配。

**检查命令**

.. code-block:: shell

   # 如 SDK 安装路径不同，请替换为实际路径
   source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh

   python - <<'PY'
   import torch
   print("torch imported")
   print("supa available:", torch.supa.is_available())
   PY

**修复步骤**

1. 确认 SUDA 和 SUPA SDK 来自正式发布包，且与 torch_supa 版本匹配。
2. 在运行程序前执行 SDK 环境配置脚本，例如 ``source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh``。
3. 如果路径不同，请将命令中的 SDK 路径替换为实际安装路径。
4. 重新打开终端或重新启动容器后再次验证。

**验证方法**

``torch.supa.is_available()`` 应返回 ``True``，且可以创建 ``device="supa"`` 的张量。

.. rubric:: Q: 编译过程中出现函数未定义问题

**可能原因**

切换 PyTorch 版本后，旧的编译缓存仍然存在，导致编译时引用了不匹配的符号。

**修复步骤**

.. code-block:: shell

   CLEAN_TORCH_SUPA=1 python3 setup.py clean
   python3 setup.py bdist_wheel

**验证方法**

重新安装生成的 ``whl`` 包后，执行安装验证脚本，确认可以导入 ``torch`` 并调用 ``torch.supa.is_available()``。

训练和推理
------------

.. rubric:: Q: 导入 ``torch`` 时出现警告信息

**症状**

.. code-block:: text

   [W212 07:05:54.828986137 GeneratorForPrivateuseone.cpp:17] Warning: REGISTER_GENERATOR_PRIVATEUSE1 is deprecated.
   ...
   The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.supa and torch.nn.Module.supa now..
   The backend in torch.distributed.init_process_group set to bccl now..

**说明**

这是正常的提示信息，说明 torch_supa 已正确加载并自动切换到 SUPA 后端，无需处理。

**验证方法**

.. code-block:: python

   import torch

   print(torch.supa.is_available())
   print(torch.rand(2, 3, device="supa").device)

.. rubric:: Q: 出现 double 类型不支持的警告

**症状**

.. code-block:: text

   UserWarning: SUPA device does not support double (float64). Please use float32 instead.

**可能原因**

壁仞设备不完全支持 ``torch.float64``。如果代码中未显式指定 dtype，某些张量可能默认使用 ``torch.float64``。

**修复步骤**

建议在代码中显式使用 ``torch.float32``：

.. code-block:: python

   # 不推荐：默认可能为 torch.float64
   x = torch.tensor([1.0, 2.0])

   # 推荐
   x = torch.tensor([1.0, 2.0], dtype=torch.float32)

**验证方法**

检查关键输入张量和模型参数的 dtype：

.. code-block:: python

   print(x.dtype)
   for name, param in model.named_parameters():
       print(name, param.dtype)

.. rubric:: Q: 出现内存不足（OOM）错误

**可能原因**

- ``batch size`` 过大。
- 模型中间激活占用过多显存。
- 数据加载或缓存未及时释放。
- 混合精度未启用，显存压力较高。

**修复步骤**

1. **减小 batch size**。
2. **启用梯度检查点**：

   .. code-block:: python

      from torch.utils.checkpoint import checkpoint

      # 对部分层启用梯度检查点
      output = checkpoint(model.layer, input)

3. **使用混合精度训练**：

   .. code-block:: python

      with torch.supa.amp.autocast(dtype=torch.float16):
          output = model(input)

4. **手动清理缓存**：

   .. code-block:: python

      torch.supa.empty_cache()

**验证方法**

逐步恢复 ``batch size``，观察是否仍出现 OOM；必要时使用 Profiler 或日志工具定位内存峰值。

.. rubric:: Q: torch.jit.script 不支持

**症状**

.. code-block:: text

   RuntimeWarning: torch.jit.script and torch.jit.script_method will be disabled by transfer_to_supa

**解决方案**

torch_supa 当前不支持 ``torch.jit.script``。如果需要使用 JIT 功能，请避免使用 ``transfer_to_supa`` 自动转换，改为手动调用 ``torch.supa`` 接口，或参考对应版本的 API 支持矩阵确认替代方案。

分布式训练
------------

.. rubric:: Q: 分布式训练初始化失败

**可能原因**

- ``MASTER_ADDR``、``MASTER_PORT``、``WORLD_SIZE`` 或 ``RANK`` 设置不一致。
- 多机之间网络不通，或端口被占用。
- BCCL 后端未正确安装或环境变量未加载。
- 每个进程绑定的设备不正确。

**检查命令**

.. code-block:: shell

   export MASTER_ADDR=<master-ip>
   export MASTER_PORT=<master-port>
   export WORLD_SIZE=4
   export RANK=0

   # 检查端口连通性，请根据实际环境选择 nc、telnet 或集群管理工具
   nc -vz <master-ip> <master-port>

**初始化示例**

.. code-block:: python

   import torch.distributed as dist

   dist.init_process_group(backend="bccl")

**修复步骤**

1. 确认所有节点使用相同的 ``MASTER_ADDR``、``MASTER_PORT`` 和 ``WORLD_SIZE``。
2. 确认每个进程的 ``RANK`` 唯一且连续。
3. 确认 SDK 环境变量已加载，且 BCCL 组件来自匹配版本的正式发布包。
4. 检查节点间网络和防火墙策略。
5. 使用小规模单机多卡任务验证后，再扩展到多机任务。

性能优化
----------

.. rubric:: Q: 训练速度较慢

**可能原因**

- 未启用混合精度。
- 数据加载成为瓶颈。
- 模型中存在未优化或回退算子。
- 分布式通信或同步开销较高。

**修复步骤**

1. **启用混合精度训练**：

   .. code-block:: python

      scaler = torch.supa.amp.GradScaler()
      with torch.supa.amp.autocast():
          output = model(input)

2. **使用 SUPA Graph**：

   .. code-block:: python

      g = torch.supa.SUPAGraph()
      with torch.supa.graph(g):
          output = model(input)
      g.replay()

3. **检查数据加载瓶颈**：

   .. code-block:: python

      DataLoader(dataset, batch_size=64, num_workers=4)

4. **使用 Profiler 分析瓶颈**：

   .. code-block:: python

      with torch.profiler.profile() as prof:
          model(input)
      print(prof.key_averages().table())

**验证方法**

记录优化前后的吞吐、step time、显存占用和主要算子耗时，确认优化项确实带来收益。

其他问题
----------

.. rubric:: Q: 如何查看 torch_supa 版本

.. code-block:: python

   import torch_supa
   print(torch_supa.__version__)

.. rubric:: Q: 如何查看设备信息

.. code-block:: python

   import torch

   # 查看设备数量
   print(torch.supa.device_count())

   # 查看设备名称
   print(torch.supa.get_device_name(0))

   # 查看设备属性
   print(torch.supa.get_device_properties(0))

.. rubric:: Q: 如何获取技术支持

如遇到文档中未涵盖的问题，请通过正式客户支持渠道获取支持。提交问题时建议附带以下信息：

1. PyTorch、torch_supa、SUPA SDK、驱动和操作系统版本。
2. 环境信息收集工具输出，详见 :doc:`chapter07_torch_supa_tools`。
3. 最小复现脚本、完整报错日志和运行命令。
4. 使用的硬件型号、设备数量、单机/多机配置。
5. 如问题与性能相关，请附带 Profiler 结果或关键性能指标。
