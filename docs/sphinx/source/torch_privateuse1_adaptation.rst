基于 torch 的开源仓适配 PrivateUse1 设备指南
==================================================

概述
------------------------------------

开源社区中基于 pytorch 的计算库需要在 BIREN 设备上重新适配，在 BR2xx 系列产品中，我们可以通过 SUDA 及 `br_pytorch2` 提供的能力进行一些快速适配。

本文档详细说明基于 CMake 编译流程的 Torch PrivateUse1 设备适配核心步骤，重点解决如何使用 SUDA，头文件查找优先级、符号冲突规避、NV 平台代码编译三大关键问题。
主要目的是安全地将原有 CUDA 设备相关代码转发到 `br_pytorch2` 实现的 ``c10_supa`` 库中，同时使用 SUDA 将 CUDA API 转发到 SUPA API 中。
目前修改 cmake 相关配置文件即可实现上述目标。

但由于硬件差异，cu 源码仍需要一定的改造，这部分修改量视情况而定，相关修改规则可参考 :ref:`cu 源码修改`。

参考示例：https://gitlab.birentech.com/E00667/TransformerEngine/-/compare/develop_br200...for_torch_supa?from_project_id=30232

SUDA
--------------------

SUDA 详细介绍见：https://gitlab.birentech.com/software/suda。目前 SUDA 会随 lkg 发布并提供 wheel 包，可以直接安装在当前使用的 python 环境中。
一般我们需要对于所有 cu 文件使用 suda 编译以尽量不修改社区源码。


.. code-block:: cmake
   :linenos:

   # find suda package
   execute_process(
   COMMAND ${Python_EXECUTABLE} -c "import suda;print(suda.cmake_prefix_path)"
   OUTPUT_VARIABLE SUDA_PREFIX_PATH
   ERROR_QUIET
   RESULT_VARIABLE SUDA_RESULT
   OUTPUT_STRIP_TRAILING_WHITESPACE
   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
   )
   list(APPEND CMAKE_PREFIX_PATH ${SUDA_PREFIX_PATH})
   find_package(SUDA REQUIRED)

   # 使用 suda 编译 cu 文件，会自动链接 cuda runtime
   suda_add_library(torch_supa SHARED ${ATen_CUDA_CU_SRCS} ${ATen_CUDA_CPP_SRCS})
   # 添加 cublas 等依赖
   target_link_libraries(torch_supa PRIVATE SUDA::cublas_static)


头文件查找优先级配置
--------------------

CUDA 源码在 PrivateUse1 设备适配时，需让编译器 **优先查找 br_pytorch2 提供的同名头文件**，其余通用头文件（包括 CUDA 相关基础头文件）由 `torch_cpu` 提供，且 `torch_cpu` 已包含完整的 CUDA 部分头文件。

CMake 配置实现
~~~~~~~~~~~~~~~~~~~~~~~~
使用 ``target_include_directories`` 并指定 ``BEFORE`` 关键字，强制 `br_pytorch2` 提供的头文件目录优先级高于 `torch_cpu` 的头文件目录：

.. code-block:: cmake
   :linenos:

   # 1. 定义 PrivateUse1 设备头文件目录（替换为实际路径）
   set(PRIVATEUSE1_INCLUDE_DIR ${PYTORCH_SUPA_INSTALL_PATH}/include/)

   # 2. 配置头文件目录优先级
   target_include_directories(targetxxx # targetxxx 替换为实际编译目标名称（可执行文件/库）
       BEFORE
       PRIVATE
       ${PRIVATEUSE1_INCLUDE_DIR}  # PrivateUse1 设备头文件（仅需修改的部分）
   )

- ``BEFORE``：强制将 PrivateUse1 设备相关头文件目录加入编译器 ``-I`` 参数最前端，保证优先查找；
- ``PRIVATE``：遵循模块化原则，仅当前编译目标可见，不影响其他目标；

符号冲突规避（is_cuda 等函数）
--------------------------------------------

PrivateUse1 设备适配中使用 ``is_cuda``、``check_backend`` 等函数时，易与 `torch_cpu` 原生符号冲突，需通过「本地 API 封装 + 版本脚本」限制符号暴露。

is_cuda 函数修改
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is_cuda 已经由 `br_pytorch2` 提供的 `c10/core/Device.h` 修改定义, 修改后代码如下， 用户无需额外处理。

.. code-block:: c
   :linenos:

   // bool is_cuda() const noexcept {
      // return type_ == DeviceType::CUDA;
   __attribute__((always_inline)) bool is_cuda() const noexcept {
    return type_ == DeviceType::PrivateUse1;
   }
   /// Return true if the device is of PrivateUse1 type.


配置版本脚本（限制符号暴露）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过 ``set_target_properties`` 设置链接选项，使用 ``version_script.lds`` 控制目标符号为 local 符号， 避免与 `torch_cpu` 原生符号冲突。

* 创建版本脚本文件（version_script.lds）

.. code-block:: text
   :linenos:

    targetxxx {
    local:
        *checkBackend*;
        *getCUDAHooks*;
        *is_cuda*;
    };


* CMake 配置链接标志

.. code-block:: cmake
   :linenos:

   # 为目标配置链接标志，应用版本脚本规避符号冲突
   set_target_properties(targetxxx
       PROPERTIES
       LINK_FLAGS "-Wl,--version-script=${CMAKE_SOURCE_DIR}/version_script.lds"
   )

- 版本脚本仅在 Linux 平台生效（``-Wl,--version-script`` 是 GNU ld 链接器参数）；
- 目前仅发现 ``is_cuda``、``check_backend``、 ``getCUDAHooks`` 函数需本地化，可按使用情况进行配置；


NV 平台代码编译
------------------

编译 NV 平台（CUDA 后端）代码时，直接使用 Torch 原生 CUDA 头文件即可，无需额外配置自定义头文件。

.. _cu 源码修改:

cu 源码修改规则
------------------------------------------------

cu 源码得编译需要 SUDA 支持，但软硬件本身得兼容性差异可能会要求 cu 源码作出一定得修改以适应 Biren 设备。
大部分情况可以参加针对 torch aten 算子源码的 patch 修改, 可参考
https://gitlab.birentech.com/software/torch-supa-op/-/tree/develop_br200/patch_torch/。

目前修改项主要有以下几点：

double 数据类型支持不完整
~~~~~~~~~~~~~~~~~~~~~~~~~~

目前 double 数据类型是通过软件模拟的方式支持，仅支持普通的加减乘除运算，复杂运算可能会导致为定义行为。
且 double 数据类型的软件模拟性能会较差，建议尽可能屏蔽掉 double 数据类型的使用。

- brcc 编译器添加 ``-ffloat-constants`` flag，可将源码中 ``1.`` 等常量数值从默认 double 数据类型
  改为默认 float 数据类型从而减少 double 数据的产生， 需在 CMakeLists.txt 文件中添加如下配置：

   .. code-block:: cmake
      :linenos:

         list(APPEND SUPA_BRCC_FLAGS "-ffloat-constants")

- double 的复杂运算需要规避:

   .. code-block:: cpp
      :linenos:

         -__forceinline__ __device__ double device_sqrt(scalar_t val) {
         -  return std::sqrt(val);
         +__forceinline__ __device__ float device_sqrt(float val) {
         +  return std::sqrtf(val);
         }

host/device 语法差异
~~~~~~~~~~~~~~~~~~~~~~~

host/device 属性推导的语法 brcc 和 nvcc 行为有差异，需手动给部分函数添加 host 属性以支持 host 端调用

   .. code-block:: cpp
         :linenos:

            -  __device__ __forceinline__ scalar_t operator() () const {
            +  __host__ __device__ __forceinline__ scalar_t operator() () const {
               return value;
               }

内嵌 ptx
~~~~~~~~~~~~~~~~~~~~~~~

内嵌 ptx 的文件需要进行对 ptx 进行替换，或者使用 ``-mira`` 编译选项尝试支持：

   .. code-block:: cmake
         :linenos:

            list(APPEND SUPA_BRCC_FLAGS "-mira")

其他
~~~~~~~~~~~~~~

部分算子配置如 blocksize limit 存在差异，需要修改源码

