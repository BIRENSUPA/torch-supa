API 说明
==============================================================

原生 PyTorch API 支持度
-----------------------------

本节提供 torch_supa 对原生 PyTorch API 的限制说明和兼容性映射列表。

原生 PyTorch API 功能和使用方法请参看 `PyTorch 官方 Python API 文档 <https://docs.pytorch.org/docs/stable/pytorch-api.html>`_。 算子限制说明如下：

.. list-table:: API限制说明
   :widths: 60 40
   :header-rows: 1

   * - API
     - 限制
   * - ``scaled_dot_product_attention``
     - ``SDPBackend::flash_attention`` 后端限制： ``head_size`` 仅支持 64、96、128、192、256、378、512; 仅支持 ``BFloat16`` 和 ``FLOAT32`` 。不支持 ``efficient_attention`` 后端。
   * - ``inverse``
     - 仅支持 ``FLOAT32``
   * - ``torch.linalg.norm``
     - 仅支持 ``FLOAT32`` 、 ``FLOAT16`` 
   * - ``torch.linalg.vector_norm``
     - 仅支持 ``FLOAT32`` 、 ``FLOAT16`` 
   * - ``torch.linalg.matrix_norm``
     - 仅支持 ``FLOAT32`` 、 ``FLOAT16``
   * - ``torch.linalg.vecdot``
     - 仅支持 ``FLOAT32``
   * - ``torch.linalg.multi_dot``
     - 仅支持 ``FLOAT32`` 、 ``FLOAT16`` 和 ``BFLOAT16``
   * - ``torch.linalg.det``
     - 不支持
   * - ``torch.linalg.slogdet``
     - 不支持
   * - ``torch.linalg.cond``
     - 不支持
   * - ``torch.linalg.matrix_rank``
     - 不支持
   * - ``torch.linalg.cholesky``
     - 不支持
   * - ``torch.linalg.qr``
     - 不支持
   * - ``torch.linalg.lu``
     - 不支持
   * - ``torch.linalg.lu_factor``
     - 不支持
   * - ``torch.linalg.eig``
     - 不支持
   * - ``torch.linalg.eigvals``
     - 不支持
   * - ``torch.linalg.eigh``
     - 不支持
   * - ``torch.linalg.eigvalsh``
     - 不支持
   * - ``torch.linalg.svd``
     - 不支持
   * - ``torch.linalg.svdvals``
     - 不支持
   * - ``torch.linalg.solve``
     - 不支持
   * - ``torch.linalg.solve_triangular``
     - 不支持
   * - ``torch.linalg.lu_solve``
     - 不支持
   * - ``torch.linalg.lstsq``
     - 不支持
   * - ``torch.linalg.inv``
     - 不支持
   * - ``torch.linalg.pinv``
     - 不支持
   * - ``torch.linalg.matrix_exp``
     - 不支持
   * - ``torch.linalg.matrix_power``
     - 不支持
   * - ``torch.linalg.matmul``
     - 不支持
   * - ``torch.linalg.householder_product``
     - 不支持
   * - ``torch.linalg.tensorinv``
     - 不支持
   * - ``torch.linalg.tensorsolve``
     - 不支持
   * - ``torch.linalg.cholesky_ex``
     - 不支持
   * - ``torch.linalg.inv_ex``
     - 不支持
   * - ``torch.linalg.solve_ex``
     - 不支持
   * - ``torch.linalg.lu_factor_ex``
     - 不支持
   * - ``torch.linalg.ldl_factor``
     - 不支持
   * - ``torch.linalg.ldl_factor_ex``
     - 不支持
   * - ``torch.linalg.ldl_solve``
     - 不支持
   * - ``torch.fft.*``
     - 不支持
   * - ``torch.sparse.*``
     - 不支持
   * - ``torch.ao.quantization.*``
     - 不支持

自定义 API
-------------------

torch_supa 通过 PyBind11 和 CPython 扩展提供了底层 ``C++ API`` 绑定，这些 API 主要位于 ``torch_supa._C`` 模块中，供高级用户和框架开发者使用。普通训练和推理用户通常无需直接调用这些底层接口。

设备管理 API
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 设备管理 C++ API
   :widths: 60 40
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_init()``
     - 初始化 SUPA 扩展
   * - ``torch_supa._C._supa_synchronize()``
     - 同步设备
   * - ``torch_supa._C._supa_setDevice(device)``
     - 设置当前设备
   * - ``torch_supa._C._supa_getDevice()``
     - 获取当前设备索引
   * - ``torch_supa._C._supa_getDeviceCount()``
     - 获取设备数量
   * - ``torch_supa._C._supa_exchangeDevice(device)``
     - 切换设备并返回原设备索引
   * - ``torch_supa._C._supa_maybeExchangeDevice(device)``
     - 条件切换设备
   * - ``torch_supa._C._supa_canDeviceAccessPeer(device, peer_device)``
     - 检查设备间 P2P 访问能力
   * - ``torch_supa._C._supa_getCompiledVersion()``
     - 获取编译时的 SUPA 版本
   * - ``torch_supa._C._supa_hasPrimaryContext(device)``
     - 检查设备是否有主上下文

内存管理 API
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 内存管理 C++ API
   :widths: 60 40
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_emptyCache()``
     - 清空缓存分配器
   * - ``torch_supa._C._host_emptyCache()``
     - 清空主机缓存分配器
   * - ``torch_supa._C._supa_memoryStats(device)``
     - 获取设备内存统计信息
   * - ``torch_supa._C._supa_memorySnapshot()``
     - 获取内存快照
   * - ``torch_supa._C._supa_resetPeakMemoryStats(device)``
     - 重置峰值内存统计
   * - ``torch_supa._C._supa_resetAccumulatedMemoryStats(device)``
     - 重置累积内存统计
   * - ``torch_supa._C._supa_getMemoryFraction(device)``
     - 获取设备内存比例
   * - ``torch_supa._C._supa_setMemoryFraction(fraction, device)``
     - 设置设备内存比例
   * - ``torch_supa._C._supa_attach_out_of_memory_observer(observer)``
     - 附加 OOM 观察者回调

流管理 API
~~~~~~~~~~~~~~~~~~

.. list-table:: 流管理 C++ API
   :widths: 60 40
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_getCurrentStream(device)``
     - 获取当前流（返回元组）
   * - ``torch_supa._C._supa_getCurrentRawStream(device)``
     - 获取当前流指针
   * - ``torch_supa._C._supa_getDefaultStream(device)``
     - 获取默认流
   * - ``torch_supa._C._supa_setStream(stream_id, device_index, device_type)``
     - 设置当前流
   * - ``torch_supa._C._supa_isCurrentStreamCapturing()``
     - 检查当前流是否在捕获

显存分配器 API
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 分配器 C++ API
   :widths: 70 30
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_supaHostAllocator()``
     - 获取主机分配器指针
   * - ``torch_supa._C._supa_supaCachingAllocator_raw_alloc(size, stream)``
     - 原始内存分配
   * - ``torch_supa._C._supa_supaCachingAllocator_raw_delete(ptr)``
     - 原始内存释放
   * - ``torch_supa._C._supa_supaCachingAllocator_enable(enabled)``
     - 启用/禁用缓存分配器
   * - ``torch_supa._C._supa_supaCachingAllocator_set_allocator_settings(settings)``
     - 设置分配器参数
   * - ``torch_supa._C._supa_getAllocatorBackend()``
     - 获取分配器后端名称
   * - ``torch_supa._C._supa_getAllocator()``
     - 获取当前分配器
   * - ``torch_supa._C._supa_changeCurrentAllocator(allocator)``
     - 切换当前分配器

``BLAS API``
~~~~~~~~~~~~~~~~~~~

.. list-table:: BLAS C++ API
   :widths: 70 30
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_getCurrentBlasHandle()``
     - 获取当前 BLAS 句柄
   * - ``torch_supa._C._supa_clearSublasWorkspaces()``
     - 清空 Sublas 工作空间

``DLPack API``
~~~~~~~~~~~~~~~~~~~~

.. list-table:: DLPack C++ API
   :widths: 45 55
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._supa_to_dlpack(tensor)``
     - 将张量转换为 DLPack 格式
   * - ``torch_supa._C._supa_from_dlpack(capsule)``
     - 从 DLPack 格式创建张量

``Profiler API``
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Profiler C++ API
   :widths: 70 30
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._profiler.ProfilerActivity``
     - 性能分析活动类型枚举
   * - ``torch_supa._C._profiler.ProfilerConfig``
     - 性能分析配置类
   * - ``torch_supa._C._profiler._enable_profiler(config, activities)``
     - 启用性能分析器
   * - ``torch_supa._C._profiler._disable_profiler()``
     - 禁用性能分析器
   * - ``torch_supa._C._profiler._prepare_profiler(config, activities)``
     - 准备性能分析器
   * - ``torch_supa._C._profiler._kineto_step()``
     - 执行 kineto 步骤

分布式通信 API
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 分布式通信 C++ API
   :widths: 60 40
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch_supa._C._distributed_c10d.ProcessGroupBCCL``
     - BCCL 进程组类
   * - ``torch_supa._C._distributed_c10d._hash_tensors(tensors)``
     - 计算张量哈希值
   * - ``torch_supa._C._distributed_c10d._dump_bccl_trace(...)``
     - 导出 BCCL 追踪日志
   * - ``torch_supa._C._distributed_c10d._dump_bccl_trace_json(...)``
     - 导出 BCCL 追踪日志（JSON 格式）
   * - ``torch_supa._C._distributed_c10d._get_intra_node_comm_usage_counter()``
     - 获取节点内通信使用计数

自定义算子 API
~~~~~~~~~~~~~~~~~~~~~~~

torch_supa 通过 PyTorch 的自定义算子注册机制提供了额外的算子支持。

.. list-table:: 自定义算子 C++ API
   :widths: 70 30
   :header-rows: 1

   * - API
     - 功能描述
   * - ``torch.ops.custom._get_data_ptr(tensor)``
     - 获取张量的数据指针地址

**使用示例**：

.. code-block:: python

   import torch

   # 获取张量数据指针
   x = torch.randn(10, 10, device='supa')
   ptr = torch.ops.custom._get_data_ptr(x)
   print(f"数据指针: {ptr}")
