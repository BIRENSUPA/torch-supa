框架特性指南
==============================================================

概述
---------

torch_supa 插件最大程度继承了 PyTorch 框架的特性并进行了扩展。用户在使用 torch_supa 时，几乎无需修改原生 PyTorch 的代码风格和接口，即可在壁仞硬件上运行模型。

本章同时包含用户功能说明和开发者参考内容。普通训练、推理和迁移用户建议重点阅读自动加载和设备转换、Profiler 工具、自动混合精度、社区开源库适配和图模式；算子适配、重新注册 Native 算子和 C++ Extension 相关内容主要面向需要扩展算子、适配三方库或参与框架开发的高级用户。


.. _feature-auto-load-device-conversion:

自动加载和设备转换
------------------

在 PyTorch 2.5 及以上版本中，torch_supa 支持在加载 ``torch`` 模块时自动加载插件，并将 CUDA 设备自动转换为 SUPA 设备：

.. code-block:: python

   import torch

   # CUDA 设备自动映射到 SUPA
   x = torch.rand(3, 4).cuda()
   print(x.device)  # 输出: supa:0

   # torch.cuda 接口自动替换为 torch.supa
   print(torch.cuda.is_available())  # 等价于 torch.supa.is_available()

.. _feature-operator-adaptation:

算子适配
----------

torch_supa 支持三种算子接入方式，不同需求应选择适合的接入方法：

.. note::

   本节为高级用户和开发者参考内容，适用于算子扩展、三方库适配和框架开发场景。普通模型训练和推理用户通常无需执行本节步骤。

.. list-table:: 算子接入方式
   :widths: 30 70
   :header-rows: 1

   * - 接入方式
     - 适用场景
   * - Patch Native 算子
     - 通过原生 ``torch.xxx`` 调用的算子，仅需少量源码 ``patch``，归属 torch-supa-op 仓
   * - 重新注册 Native 算子
     - 通过原生 ``torch.xxx`` 调用的算子，因性能等原因需 SUPA 重写，归属 torch-supa 仓
   * - C++ Extension 自定义算子
     - 其他自定义或三方库算子，归属 ``torch_supa_ext`` 仓

Patch Native 算子
~~~~~~~~~~~~~~~~~~~~~~~~~

在源码树的 ``torch_supa_op`` 组件中维护了一系列 ``patch`` 文件夹，用于适配不同版本的 PyTorch 框架。

- ``third-party/torch_supa_op/patch_privateuse1`` 主要是 ``c10_cuda`` 相关的头文件 ``patch``。
- ``third-party/torch_supa_op/patch_torch`` 适配 pytorch v2.6.0 版本的算子 ``patch``。
- ``third-party/torch_supa_op/patch_torch28`` 适配 pytorch v2.8.0 版本的算子 ``patch``。
- ``third-party/torch_supa_op/patch_torch29`` 适配 pytorch v2.9.0 版本的算子 ``patch``。

这些补丁在开发或升级过程中可能出现遗漏或逻辑错误。若遇到相关问题，可通过以下方式定位：

1. 使用`` GDB`` 调试：在运行过程中跟踪调用栈，定位问题代码。

2. 查看源码：直接查看 ``third-party/torch_supa_op/build/_deps/pytorch-src`` 目录下的源码，分析补丁是否被正确应用或是否存在冲突。

重新注册 Native 算子
~~~~~~~~~~~~~~~~~~~~~~

**步骤一：算子注册**

在 ``torch_supa/csrc/aten/supa_native_functions.yaml`` 中添加算子声明：

.. code-block:: yaml
   :linenos:

   backend: SUPA
   cpp_namespace: at::supa
   supported:
      - _addmm_activation.out
      - _pin_memory
      - _cdist_forward
   perf_supported:  # 开发中
   autograd:
      - renorm

**参数说明**：

- ``backend``：算子所属后端，目前仅支持 SUPA
- ``cpp_namespace``：算子实现的 ``C++`` 命名空间，默认 ``at::supa``
- ``supported``：支持的原生算子列表，必须存在于 PyTorch 的 ``native_functions.yaml`` 中
- ``autograd``：支持自动微分的算子列表
- ``perf_supported``：支持性能优化的算子列表（功能开发中）

**步骤二：算子实现**

在 ``torch_supa/csrc/aten/ops/`` 目录下添加算子实现文件。

*类型一: :ref:`Structured OP <structured_op>`*

使用 ``SUPA_IMPL_FUNC`` 宏，复用 PyTorch 原生的 ``meta`` 函数, 如 ``_addmm_activation`` 算子：

.. code-block:: cpp
   :linenos:

   SUPA_IMPL_FUNC(_addmm_activation)
   (const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha, bool use_gelu,
    const Tensor& result) {
       addmm_out_impl(const_cast<Tensor&>(result),
           self, mat1, mat2, beta, alpha,
           use_gelu ? Activation::GELU : Activation::RELU);
   }

*类型二：常规算子*

直接在 ``SUPANativeFunctions`` 命名空间中实现, 如 ``_cdist_forward`` 算子：

.. code-block:: cpp
   :linenos:

   Tensor SUPANativeFunctions::_cdist_forward(
       const Tensor& x1, const Tensor& x2,
       double p, c10::optional<int64_t> compute_mode) {
       auto result = cdist_impl(x1, x2, p, compute_mode);
       return result;
   }

*类型三：自动微分算子*

继承 ``torch::autograd::Function`` 实现反向传播，如 ``Renorm`` 算子：

.. code-block:: cpp

   class RenormFunction : public torch::autograd::Function<RenormFunction> {
   public:
       static Tensor forward(AutogradContext* ctx, const Tensor& self,
                            const Scalar& p, int64_t dim, const Scalar& maxnorm);
       static std::vector<Tensor> backward(AutogradContext* ctx,
                                          std::vector<Tensor> grad_outputs);
   };

C++ Extension 自定义算子
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

具体实现细节请参考《``torch_supa_ext`` 用户指南》。

.. _profiler-tool:
.. _feature-profiler-tool:

Profiler 工具
----------------------

torch_supa 支持 PyTorch 原生的 Profiler 工具，用于性能分析和瓶颈定位。

.. code-block:: python

   import torch

   a = torch.rand(2, 3, dtype=torch.float32, device='cuda')
   b = torch.rand(2, 3, dtype=torch.float32, device='cuda')

   with torch.profiler.profile(
      activities=[
         torch.profiler.ProfilerActivity.CPU,
         torch.profiler.ProfilerActivity.SUPA,
      ],
      record_shapes=True,
      profile_memory=True,
   ) as prof:
      c = a + b

   # 打印性能统计
   print(prof.key_averages().table(row_limit=10))

   # 导出 Chrome Trace 文件
   prof.export_chrome_trace("trace.json")

保存 ``trace.json`` 文件，并打印出结果：

.. code-block:: text

   -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      SUPA Mem  Self SUPA Mem    # of Calls
   -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                  aten::add         0.07%       1.069ms        96.84%        1.515s        1.515s           0 b           0 b         512 b         512 b             1
               supaGetDevice         0.03%     446.208us         0.03%     446.208us      55.776us           0 b           0 b           0 b           0 b             8
            supaLaunchKernel        96.56%        1.510s        96.74%        1.513s        1.513s           0 b           0 b           0 b           0 b             1
            Instrumentation         0.18%       2.846ms         0.18%       2.846ms       1.423ms           0 b           0 b           0 b           0 b             2
      supaDeviceSynchronize         3.16%      49.445ms         3.16%      49.445ms      49.445ms           0 b           0 b           0 b           0 b             1
   -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
   Self CPU time total: 1.564s


更多使用方法请参考 `PyTorch 官方文档 Profiler 章节 <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_。

.. _amp-tool:
.. _feature-amp:

自动混合精度（AMP）
----------------------------

torch_supa 支持自动混合精度训练，包括 ``autocast()`` 和 ``GradScaler()`` 接口：

.. code-block:: python

   import torch
   # Creates some tensors in default dtype (here assumed to be float32)
   a_float32 = torch.rand((8, 8), device="supa")
   b_float32 = torch.rand((8, 8), device="supa")
   c_float32 = torch.rand((8, 8), device="supa")
   d_float32 = torch.rand((8, 8), device="supa")

   with torch.autocast(device_type="supa"):
      # torch.mm is on autocast's list of ops that should run in float16.
      # Inputs are float32, but the op runs in float16 and produces float16 output.
      # No manual casts are required.
      e_float16 = torch.mm(a_float32, b_float32)
      print(f"e_float16.dtype: {e_float16.dtype}")
      # Also handles mixed input types
      f_float16 = torch.mm(d_float32, e_float16)
      print(f"f_float16.dtype: {f_float16.dtype}")

执行脚本会打印出两次调用 ``torch.mm`` 的输出数据类型均为 ``torch.float16``。

**使用限制**：

- 混合精度训练可能提升性能，但不适用于所有模型
- 开启混合精度后，可能出现精度下降或训练不收敛的情况，需根据实际情况调整

更多使用方法请参考 `PyTorch 官方文档 amp 章节 <https://docs.pytorch.org/docs/stable/amp.html>`_。

.. _feature-community-library-support:

社区开源库适配
--------------

开源社区中基于 PyTorch 的计算库需要适配后才能在壁仞设备上运行。通过正式发布包中的 SUDA 工具和 torch_supa 提供的能力，可以进行快速适配。

适配步骤
~~~~~~~~

**1. 修改 CMake 配置**

使用 SUDA 编译 CUDA 源码：

.. code-block:: cmake
   :linenos:

   # 查找 SUDA 包
   execute_process(
       COMMAND ${Python_EXECUTABLE} -c "import suda;print(suda.cmake_prefix_path)"
       OUTPUT_VARIABLE SUDA_PREFIX_PATH
       OUTPUT_STRIP_TRAILING_WHITESPACE
   )
   list(APPEND CMAKE_PREFIX_PATH ${SUDA_PREFIX_PATH})
   find_package(SUDA REQUIRED)

   # 使用 suda_add_library 编译 .cu 文件
   suda_add_library(my_lib SHARED ${CUDA_SOURCES})

**2. 配置头文件优先级**

让编译器优先查找 torch_supa 提供的头文件：

.. code-block:: cmake
   :linenos:

   set(TORCH_SUPA_INCLUDE_DIR ${TORCH_SUPA_INSTALL_PATH}/include/)
   target_include_directories(my_target BEFORE PRIVATE ${TORCH_SUPA_INCLUDE_DIR})

**3. 解决符号冲突**

创建 ``version_script.lds`` 文件：

.. code-block:: text

   my_target {
       local:
           *checkBackend*;
           *getCUDAHooks*;
           *is_cuda*;
   };

在 ``CMake`` 中配置链接选项：

.. code-block:: cmake

   set_target_properties(my_target PROPERTIES
       LINK_FLAGS "-Wl,--version-script=${CMAKE_SOURCE_DIR}/version_script.lds"
   )

**4. cu 源码修改**

部分 CUDA 源码可能需要修改以适应壁仞硬件：

- **double 数据类型**：壁仞设备不完全支持 ``double``，建议使用 ``float``。
- **host/device 语法**：部分函数需显式添加 ``__host__`` 属性
- **内嵌 PTX**：使用 ``-use-mira=true`` 编译选项或替换 PTX 代码

.. _feature-graph-mode:

PyTorch 图模式
--------------------

torch_supa 支持 ``torch.compile()`` 图模式加速。

**核心组件**：

.. list-table:: torch.compile() 核心组件
   :header-rows: 1
   :widths: 20 80

   * - 组件
     - 功能
   * - ``Dynamo``
     - 动态图捕获，将 PyTorch 动态图转化为静态图
   * - Inductor
     - 静态图优化，自动优化模型结构并生成高效计算图
   * - ``SUPAGraphs``
     - 将优化后的计算图转换为 SUPA 运行时代码

**使用示例**：

.. code-block:: python

   import torch

   model = MyModel().supa()

   # 使用 Inductor 后端
   compiled_model = torch.compile(model, backend="inductor")
   output = compiled_model(input)

.. note::

   Inductor 后端需要安装 ``br-triton`` 依赖。