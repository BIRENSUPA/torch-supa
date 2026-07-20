算子适配
==============================================================

.. note::

   本节为高级用户和开发者参考内容，适用于算子扩展、三方库适配和框架开发场景。普通模型训练和推理用户通常无需阅读本章节。

.. list-table:: 算子接入方式
   :widths: 30 70
   :header-rows: 1

   * - 接入方式
     - 适用场景
   * - :ref:`重新注册 Native 算子 <register_native_op>`
     - 通过原生 ``torch.xxx`` 调用的算子，因性能等原因需 SUPA 重写
   * - :ref:`C++ Extension 自定义算子 <cpp_extension>`
     - 其他自定义或三方库算子

torch_supa 支持两种算子接入方式，不同需求应选择适合的接入方法：

.. _register_native_op:

方式一: 重新注册 Native 算子
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   perf_supported:
      - cross_entropy_loss
   autograd:
      - renorm
   custom:
      - _get_data_ptr(Tensor self) -> int

**参数说明**：

- ``backend``：算子所属后端，目前仅支持 SUPA
- ``cpp_namespace``：算子实现的 ``C++`` 命名空间，默认 ``at::supa``
- ``supported``：支持的原生算子列表，必须存在于 PyTorch 的 ``native_functions.yaml`` 中
- ``autograd``：支持自动微分的算子列表
- ``perf_supported``：支持性能优化的算子列表
- ``custom``: 自定义算子列表

**步骤二：算子实现**

在 ``torch_supa/csrc/aten/ops/`` 目录下添加算子实现文件。

类型一: :ref:`Structured 算子 <structured_op>`

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

类型二：常规算子

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

.. _cpp_extension:

方式一: C++ Extension 自定义算子
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

功能描述请参考官方 `torch.utils.cpp_extension <https://docs.pytorch.org/docs/2.12/cpp_extension.html#torch-utils-cpp-extension>`_ 。

试用示例：

.. code-block:: python
   :linenos:

   import os
   import sys
   from setuptools import setup
   from torch_supa.utils.cpp_extension import CppExtension, BuildExtension, SupaExtension, SudaExtension
   CXX_FLAGS = ["-g"]
   USE_NINJA = os.getenv("USE_NINJA") == "1"

   setup(
      name='supa_extension',
      description=usage,
      version="0.1.0",
      packages=["supa_extension"],
      cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
      ext_modules = [
         CppExtension(
            "supa_extension.cpp", ["supa_extension.cpp"], extra_compile_args=CXX_FLAGS, define_macros=[("DEBUG", None)]
         ),
         SupaExtension(
            "supa_extension.supa",
            [
                  "supa_extension.cpp",
                  "supa_extension.su",
            ],
            extra_compile_args={"cxx": CXX_FLAGS, "brcc": ["-O2"]},
         ),
         SudaExtension(
            "supa_extension.suda",
            [
                  "supa_extension.cpp",
                  "supa_extension.cu",
            ],
            # -D_SUPA_CUDA_ is only for reuse code of supa_extension.cpp.
            extra_compile_args={"cxx": ["-D_SUPA_CUDA_"], "nvcc": ["-O2"]},
            libraries=["torch_supa_op"],
         ),
      ]

      cmdclass={
         'build_ext': BuildExtension
      })


.. _structured_op:

Structured算子介绍
~~~~~~~~~~~~~~~~~~~~~~~~

Structured 算子是在 ``native_functions.yaml`` 中通过 ``structured: True`` 标记定义的算子类型，主要应用于 ``out`` 变体。
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
Structured 算子的执行分为两个阶段：

1. **meta 阶段**：负责形状推断和预处理（如广播、类型转换、张量连续化等）
2. **impl 阶段**：执行实际的计算操作

对应的代码实现如下：

.. code-block:: cpp
   :linenos:
   :caption: Structured 算子的调用流程

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