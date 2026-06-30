附录
==============================================================

名词解释
---------

.. list-table:: 名词解释
   :widths: 25 75
   :header-rows: 1

   * - 名词
     - 说明
   * - torch_supa
     - Torch SUPA 软件包名称，可通过 ``import torch_supa`` 使用
   * - SUPA
     - 可扩展统一并行架构（Scalable Unified Parallel Architecture），壁仞推出的通用并行计算平台和编程模型
   * - PrivateUse1
     - PyTorch 为新后端设备预留的 DispatchKey，用于新硬件后端验证和集成
   * - BCCL
     - Biren Collective Communication Library，壁仞集合通信库，用于分布式训练
   * - SUDA
     - SUPA CUDA 兼容层，支持用户在壁仞 GPU 上运行 CUDA 应用程序
   * - AMP
     - Automatic Mixed Precision，自动混合精度训练技术
   * - BLAS
     - Basic Linear Algebra Subprograms，基本线性代数子程序库
   * - Sublas
     - 壁仞优化的 BLAS 实现库
   * - Inductor
     - PyTorch 2.0+ 的编译器后端，用于图优化和代码生成
   * - Dynamo
     - PyTorch 2.0+ 的动态图捕获组件

核心概念
---------

本节为高级用户和开发者参考内容，适用于算子开发、框架适配、问题定位和性能调优场景。普通训练和推理用户可以跳过本节，不影响 torch_supa 的日常使用。

.. _structured_op:

Structured OP
~~~~~~~~~~~~~~~~~~~~~~~~

Structured OP 是在 ``native_functions.yaml`` 中通过 ``structured: True`` 标记定义的算子类型，主要应用于 ``out`` 变体。
该设计实现了以下核心优势：

1. **逻辑分离**：将形状推断（``shape inference``）逻辑与实际计算内核解耦
2. **统一接口**：提供标准化的算子接口规范
3. **自动推导**：支持自动化的形状推导和类型检查

定义与结构
^^^^^^^^^^
以 ``sgn`` 算子为例，其在 ``native_functions.yaml`` 中的定义如下：

.. code-block:: yaml
   :linenos:

   # native_functions.yaml
   - func: sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
     structured: true
     structured_inherits: TensorIteratorBase
     dispatch:
       CPU, CUDA: sgn_out
       MPS: sgn_out_mps
       SparseCPU, SparseCUDA, SparseMPS: sgn_sparse_out
       SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sgn_sparse_csr_out

执行流程
^^^^^^^^
Structured OP 的执行分为两个阶段：

1. **meta 阶段**：负责形状推断和预处理（如广播、类型转换、张量连续化等）
2. **impl 阶段**：执行实际的计算操作

对应的代码实现如下：

.. code-block:: cpp
   :linenos:
   :caption: Structured OP 的调用流程

   // torch_supa/csrc/aten/generated/RegisterSUPANative_0.cpp
   at::Tensor & wrapper_PrivateUse1_sgn_out_out(const at::Tensor & self, at::Tensor & out) {
      std::optional<Device> common_device = std::nullopt;
      (void)common_device; // Suppress unused variable warning
      c10::impl::check_and_update_common_device(common_device, out, "wrapper_PrivateUse1_sgn_out_out", "out");
      c10::impl::check_and_update_common_device(common_device, self, "wrapper_PrivateUse1_sgn_out_out", "self");
      structured_sgn_out_out op(out);
      op.meta(self);
      op.impl(self, op.maybe_get_output(0));
      if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
      return out;
   }

- ``meta`` 函数实现

``meta`` 函数通常位于 ``pytorch/aten/src/ATen/native/xxxOps.cpp`` 文件中，可通过 ``TORCH_META_FUNC(op)`` 宏进行搜索：


.. code-block:: cpp
   :linenos:
   :caption: meta 函数实现示例

   // pytorch/aten/src/ATen/native/UnaryOps.cpp
   #define CREATE_UNARY_META_FUNC(func)                     \
      TORCH_META_FUNC(func) (const Tensor& self) {          \
      build_borrowing_unary_op(maybe_get_output(), self);   \
   }
   CREATE_UNARY_META_FUNC(sgn)

- ``impl`` 函数实现

``impl`` 函数的实现分为两种情况：

情况一：CPU/CUDA 共享实现

当 ``CPU`` 和 CUDA 设备使用相同的实现时，可通过 ``TORCH_IMPL_FUNC(op)`` 宏搜索，需要通过 :ref:`DispatchStub` 实现设备二次分发：
如 ``pytorch/aten/src/ATen/native/UnaryOps.cpp`` 中， ``sgn``根据条件选择 ``sgn_stub`` 或 ``sign_stub`` 分发到不同设备：

.. code-block:: cpp
   :linenos:

   TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
      if (self.is_complex()) {
         sgn_stub(device_type(), *this);
      } else {
         sign_stub(device_type(), *this);
      }
   }

情况二：设备特定实现

当不同设备需要特定优化时， ``impl`` 函数分别定义在对应的设备目录中，如 ``pytorch/aten/src/ATen/native/cuda/Loss.cu``：

第二类则是定义在  ``pytorch/aten/src/ATen/native/cuda/xxxOps.cu`` 中，则可以直接定位到设备上的算子实现。
``nll_loss_forward.out`` 算子在 CUDA 和 CPU 分别 ``dispatch`` 到不同的 ``struct impl`` 函数。

.. code-block:: yaml
   :linenos:
   :caption: 设备特定实现的配置示例

   # native_functions.yaml
   - func: nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
   python_module: nn
   structured: True
   dispatch:
      CPU: nll_loss_forward_out_cpu
      CUDA: nll_loss_forward_out_cuda
      MPS: nll_loss_forward_out_mps

对应的 CUDA 实现：

.. code-block:: cpp
   :linenos:
   :caption: 设备特定的 impl 函数实现

   // pytorch/aten/src/ATen/native/cuda/Loss.cu
   TORCH_IMPL_FUNC(nll_loss_forward_out_cuda)
   (const Tensor& self,
   const Tensor& target,
   const OptionalTensorRef weight_opt,
   int64_t reduction,
   int64_t ignore_index,
   const Tensor& output,
   const Tensor& total_weight) {
   const Tensor& weight = weight_opt.getTensorRef();
   nll_loss_forward_out_cuda_template(
         output, total_weight, self, target, weight, reduction, ignore_index);
   }

.. _dispatchstub:

``DispatchStub``
~~~~~~~~~~~~~~~~~~~~~~~~

每个算子各自有继承于 ``DispatchStub`` 的结构体声明和唯一定义，并向其注册不同 ``device type`` 的算子实现。

.. code-block:: cpp
   :linenos:

   // 声明见 pytorch/aten/src/ATen/native/UnaryOps.h
   DECLARE_DISPATCH(unary_fn, sgn_stub)

   // 定义见 pytorch/aten/src/ATen/native/UnaryOps.cpp
   DEFINE_DISPATCH(sgn_stub);

   // cpu 实现注册见 pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
   ALSO_REGISTER_AVX512_DISPATCH(sgn_stub, &CPU_CAPABILITY::sgn_kernel)

   // gpu 实现注册见 pytorch/aten/src/ATen/native/cuda/UnarySignKernels.cu
   REGISTER_DISPATCH(sgn_stub, &sgn_kernel_cuda)

torch_supa 会将 ``REGISTER_DISPATCH`` 宏进行改写，将原本注册在 ``cuda device type`` 上的函数改为注册到 ``privateuse1`` 设备上。
所以我们现在找到了 ``native`` 算子 ``sgn`` 最终的设备算子实现 ``sgn_kernel_cuda``。