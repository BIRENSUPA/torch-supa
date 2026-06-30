API 说明
==============================================================

原生 PyTorch API 支持度
-----------------------------

torch_supa 插件通过 PyTorch 的 PrivateUse1 机制实现与原生 PyTorch API 的高度兼容。用户无需修改现有 PyTorch 代码，只需将 ``cuda`` 设备替换为 ``supa`` 设备即可在壁仞硬件上运行。

具体 API 功能请参看 `PyTorch 官方 Python API 文档 <https://docs.pytorch.org/docs/stable/pytorch-api.html>`_。

本节提供 torch_supa 对原生 PyTorch API 的支持矩阵入口，用于查询不同模块的支持状态和限制说明。实际行为以对应版本的正式发布说明、API 支持矩阵和运行验证结果为准。

torch_supa API 支持度及限制参见：

- :ref:`torch <native-api-torch>`
- :ref:`torch.nn <native-api-torch_nn>`
- :ref:`torch.nn.functional <native-api-torch_nn_function>`
- :ref:`torch.Tensor <native-api-torch_tensor>`
- :ref:`Tensor Views <native-api-tensor_view>`
- :ref:`torch.amp <native-api-torch_amp>`
- :ref:`torch.autograd <native-api-torch_autograd>`
- :ref:`torch.library <native-api-torch_library>`
- :ref:`torch.accelerator <native-api-torch_accelerator>`
- :ref:`torch.cuda <native-api-torch_cuda>`
- :ref:`torch.distributed <native-api-torch_distributed>`
- :ref:`torch.distributed.algorithms.join <native-api-torch_distributed_algorithms_join>`
- :ref:`torch.distributed.elastic <native-api-torch_distributed_elastic>`
- :ref:`torch.distributed.fsdp <native-api-torch_distributed_fsdp>`
- :ref:`torch.distributed.tensor.parallel <native-api-torch_distributed_tensor_parallel>`
- :ref:`torch.distributed.optim <native-api-torch_distributed_optim>`
- :ref:`torch.distributed.pipelining <native-api-torch_distributed_pipelining>`
- :ref:`torch.distributed.symmetric_memory <native-api-torch_distributed__symmetric_memory>`
- :ref:`torch.distributed.checkpoint <native-api-torch_distributed_checkpoint>`
- :ref:`torch.distributions <native-api-torch_distributions>`
- :ref:`torch.compiler <native-api-torch_compiler>`
- :ref:`torch.fft <native-api-torch_fft>`
- :ref:`torch.func <native-api-torch_func>`
- :ref:`torch.futures <native-api-torch_futures>`
- :ref:`torch.fx <native-api-torch_fx>`
- :ref:`torch.fx.experimental <native-api-torch_fx_experimental>`
- :ref:`torch.hub <native-api-torch_hub>`
- :ref:`torch.linalg <native-api-torch_linalg>`
- :ref:`torch.monitor <native-api-torch_monitor>`
- :ref:`torch.signal <native-api-torch_signal>`
- :ref:`torch.special <native-api-torch_special>`
- :ref:`torch.overrides <native-api-torch_overrides>`
- :ref:`torch.package <native-api-torch_package>`
- :ref:`torch.profiler <native-api-torch_profiler>`
- :ref:`torch.nn.init <native-api-torch_nn_init>`
- :ref:`torch.nn.attention <native-api-torch_nn_attention>`
- :ref:`torch.onnx <native-api-torch_onnx>`
- :ref:`torch.optim <native-api-torch_optim>`
- :ref:`Quantization <native-api-quantization>`
- :ref:`Distributed RPC Framework <native-api-distributed_rpc>`
- :ref:`torch.random <native-api-torch_random>`
- :ref:`torch.nested <native-api-torch_nested>`
- :ref:`torch.sparse <native-api-torch_sparse>`
- :ref:`torch.Storage <native-api-torch_storage>`
- :ref:`torch.testing <native-api-torch_testing>`
- :ref:`torch.utils <native-api-torch_utils>`
- :ref:`torch.utils.benchmark <native-api-torch_utils_benchmark>`
- :ref:`torch.utils.checkpoint <native-api-torch_utils_checkpoint>`
- :ref:`torch.utils.cpp_extension <native-api-torch_utils_cpp_extension>`
- :ref:`torch.utils.data <native-api-torch_utils_data>`
- :ref:`torch.utils.dlpack <native-api-torch_utils_dlpack>`
- :ref:`torch.utils.mobile_optimizer <native-api-torch_utils_mobile_optimizer>`
- :ref:`torch.utils.model_zoo <native-api-torch_utils_model_zoo>`
- :ref:`torch.utils.tensorboard <native-api-torch_utils_tensorboard>`
- :ref:`Type Info <native-api-typeinfo>`
- :ref:`Named Tensors <native-api-named_tensors>`
- :ref:`torch.__config__ <native-api-torch_config>`
- :ref:`torch._logging <native-api-torch_logging>`

.. note::

   各 ``native_api`` 页面中的表格用于说明 PyTorch API 在 torch_supa 中的支持情况。其中，“是否支持”列为空表示该 API 当前尚未完全验证；“限制”列为空表示当前未记录额外限制，或尚无已知限制说明。``double`` 和 ``complex`` 数据类型相关能力当前未做充分验证，使用前请结合实际模型和目标版本进行验证。实际支持情况仍以对应版本的正式发布说明、API 支持矩阵和运行验证结果为准。

.. toctree::
   :maxdepth: 2

   native_api/torch
   native_api/torch_nn
   native_api/torch_nn_function
   native_api/torch_tensor
   native_api/tensor_view
   native_api/torch_amp
   native_api/torch_autograd
   native_api/torch_library
   native_api/torch_accelerator
   native_api/torch_cuda
   native_api/torch_distributed
   native_api/torch_distributed_algorithms_join
   native_api/torch_distributed_elastic
   native_api/torch_distributed_fsdp
   native_api/torch_distributed_tensor_parallel
   native_api/torch_distributed_optim
   native_api/torch_distributed_pipelining
   native_api/torch_distributed__symmetric_memory
   native_api/torch_distributed_checkpoint
   native_api/torch_distributions
   native_api/torch_compiler
   native_api/torch_fft
   native_api/torch_func
   native_api/torch_futures
   native_api/torch_fx
   native_api/torch_fx_experimental
   native_api/torch_hub
   native_api/torch_linalg
   native_api/torch_monitor
   native_api/torch_signal
   native_api/torch_special
   native_api/torch_overrides
   native_api/torch_package
   native_api/torch_profiler
   native_api/torch_nn_init
   native_api/torch_nn_attention
   native_api/torch_onnx
   native_api/torch_optim
   native_api/quantization
   native_api/distributed_rpc
   native_api/torch_random
   native_api/torch_nested
   native_api/torch_sparse
   native_api/torch_storage
   native_api/torch_testing
   native_api/torch_utils
   native_api/others
   native_api/torch_backend

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
