Torch_supa native 算子调用及调试指南
=========================================================

概述
----
`br_pytorch2` 粗略包括两类算子，一类是由 `torch-supa-op` 提供的 ``native 算子``，另一类则是 torch_supa/csrc/aten/supa_native_functions.yaml 中定义的 ``biren 算子``。
本文档将详细说明 `torch-supa-op` 提供的 **native 算子的调用流程及相关 Debug 调试辅助信息**。

`torch-supa-op` 将 pytorch Aten CUDA 源码改造为 PrivateUse1 设备算子，并作为算子库 `libtorch_supa.so`
供 `br_pytorch2` 链接并调用。

其流程可以参考: https://gitlab.birentech.com/software/torch-supa-op/-/tree/develop_br200/patch_privateuse1?ref_type=heads
其中 PrivateUse1 相关修改细节可通过上述目录生成 doxygen 文档并查询。

Pytorch 算子调用通过 Pybind 将 Python API 绑定到相应的 cpp API，再经由 Pytorch 设计的 Dispatcher 机制一步步经由不同 DispatchKey 分发设备相关的
DispatchKey 上，如 CUDA、CPU、PrivateUse1 等。在设备代码之前的分发都由官方发布的 CPU 版本 torch 实现，这里我们不做任何修改。

`br_pytorch2` 中的算子调用则是从 torch_supa/csrc/aten/generated/Register*.cpp 开始。该文件夹下的文件均通过 ``script/codegen_native`` 解析环境中 torch
提供的 native_functions.yaml 生成，为防止和 biren 算子重复注册，codegen_native 会过滤 biren 算子并生成新的 build/aten/ATen/native/native_functions.yaml。


Torch-supa Native OP 调用流程
----------------------------------

我们以 torch_supa/csrc/aten/generated/RegisterSUPANative.cpp 为例，这里面包含了所有 **注册在 PrivateUse1 DispatchKey** 上的 native 算子。
RegisterSUPANative 文件是解析 native_functions.yaml 生成，yaml 文件里面会定义算子名称、 overload 版本以及在各个设备上的计算函数。
以 adaptive_avg_pool3d 算子为例，我们可以通过 yaml 找到在 CUDA 上的实现函数。但现在 pytorch 更多的算子正在往 :ref:`Structured OP` 的方向迭代。

.. code-block:: yaml
   :linenos:

   # native_functions.yaml
   - func: adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!)
      out) -> Tensor(a!)
   python_module: nn
   dispatch:
      CPU: adaptive_avg_pool3d_out_cpu
      CUDA: adaptive_avg_pool3d_out_cuda
      QuantizedCPU: adaptive_avg_pool3d_out_quantized_cpu

.. code-block:: cpp
   :linenos:

   // pytorch/aten/src/ATen/native/cuda/AdaptiveAveragePooling3d.cu
   Tensor& adaptive_avg_pool3d_out_cuda(const Tensor& input,
      IntArrayRef output_size,
      Tensor& output) {
      adaptive_avg_pool3d_out_cuda_template(output, input, output_size);
      return output;
   }


.. _structured op:

Structured OP
~~~~~~~~~~~~~~~~~~~~~~~~

Structured OP 表示一类 native_functions.yaml 中通过  ``structured: True`` 标记定义，通常在 out 变体上标记。
它将形状推断逻辑与实际计算内核分离，提供统一的接口和自动化的形状推导能力。

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

.. code-block:: cpp
   :linenos:

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

Structured OP 使用 meta 成员进行形状推断和一些预处理，如广播，TypeCast，转连续张量等, 而 impl 成员进行实际计算。
meta 实现一般在 pytorch/aten/src/ATen/native/xxxOps.cpp 中，也可以尝试通过宏 ``TORCH_META_FUNC(op)`` 搜索。

.. code-block:: cpp
   :linenos:

   // pytorch/aten/src/ATen/native/UnaryOps.cpp
   #define CREATE_UNARY_META_FUNC(func)                     \
      TORCH_META_FUNC(func) (const Tensor& self) {          \
      build_borrowing_unary_op(maybe_get_output(), self);   \
   }
   CREATE_UNARY_META_FUNC(sgn)

impl 实现则有两类情况，一类是定义在 pytorch/aten/src/ATen/native/xxxOps.cpp 中，也可以尝试通过宏 ``TORCH_IMPL_FUNC(op)`` 搜索。
这意味着该 impl 函数也会被 CPU 调用，所以会需要经过 struct :ref:`DispatchStub` 实现设备二次分发。
这个在 native_functions.yaml 中也可以获取到 sgn.out 算子在 CUDA 和 CPU 都是 dispatch 到 sgn_out 同一个 struct impl 函数。

.. code-block:: cpp
   :linenos:

   // pytorch/aten/src/ATen/native/UnaryOps.cpp
   TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
   if (self.is_complex()) {
      sgn_stub(device_type(), *this);
   } else {
      sign_stub(device_type(), *this);
   }

第二类则是定义在  pytorch/aten/src/ATen/native/cuda/xxxOps.cu 中，则可以直接定位到设备上的算子实现。
这个在 native_functions.yaml 中也可以获取到 nll_loss_forward.out 算子在 CUDA 和 CPU 分别 dispatch 到不同的 struct impl 函数。

.. code-block:: yaml
   :linenos:

   # native_functions.yaml
   - func: nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
   python_module: nn
   structured: True
   dispatch:
      CPU: nll_loss_forward_out_cpu
      CUDA: nll_loss_forward_out_cuda
      MPS: nll_loss_forward_out_mps

.. code-block:: cpp
   :linenos:

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

DispatchStub
~~~~~~~~~~~~~~~~~~~~~~~~

每个算子各自有继承于 ``DispatchStub`` 的结构体声明和唯一定义，并向其注册不同 device type 的算子实现。

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

`torch-supa-op` 会将 ``REGISTER_DISPATCH`` 宏进行改写，将原本注册在 cuda device type 上的函数改为注册到 privateuse1 设备上。
所以我们现在找到了 native 算子 sgn 最终的设备算子实现 ``sgn_kernel_cuda``。


Debug 调试
--------------------

通过 DEBUG 编译 `br_pytorch2`, ``DEBUG=on python3 setup.py build`` 我们可以 gdb 断点到上述算子调用流程中的任意阶段以分析代码行为。
DEBUG=on 可以透传到子仓库 `torch_supa` 中，但如果 `torch_supa` 预先编译好，即存在 third-party/torch-supa-op/torch_supa/lib/libtorch_supa.so 文件时，
``DEBUG=on python3 setup.py build`` 只会编译  `br_pytorch2` 仓库。如果需要重新 DEBUG 编译 `torch_supa`，
需要先执行 ``CLEAN_TORCH_SUPA=on python3 setup.py clean`` 后再执行上述编译命令。

对于 Torch-supa 中的 Native op 而言，大部分使用的是源码或者补丁级别修改后的源码，所以错误的来源主要是由以下几类：

CUDA/Privateuse1 代码分支不同
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pytorch 源码中会有很多 device 相关的 check 或者分支逻辑，在 `br_pytorch2` 运行中的张量的设备类型是 Privateuse1 而不是 CUDA，这个差异会导致代码走向不同的分支。
如果这个代码分支是在 ``libtorch_cpu.so`` 中实现的，我们没有办法侵入式修改，目前是通过重新注册来解决。

如 `br_pytorch2` 中重新注册的 renorm 算子，是因为 ``toAccumulateType`` 对于 Privateuse1 设备是按照 CPU 规则做类型提升，导致把 float 提升到 double。
而 pytorch cuda 源码中也没有 double kernel 所不支持报错。

还有一些更为直观的例子，如 index_put，对于 Privateuse1 设备且 accumulate=True 时并不能走到 ``index_put_with_sort_stub`` 而报错，
也需要 `br_pytorch2` 中重新注册。 类似的目前还有 std、var、pdist、cdist 等。

.. code-block:: cpp
   :linenos:

   // pytorch/aten/src/ATen/native/TensorAdvancedIndexing.cpp
   if ((self.device().type() == DeviceType::CUDA ||
       self.device().type() == DeviceType::XPU) &&
      (accumulate ||
       (globalContext().deterministicAlgorithms() && value_.numel() > 1))) {
    TORCH_CHECK(
        value_.device() == self.device(),
        "expected device ",
        self.device(),
        " but got device ",
        value_.device(),
        " for value tensor");
    index_put_with_sort_stub(
        self.device().type(), self, indices, value_, accumulate, unsafe);
    return self;
   }

如果类似的因设备判断相关的逻辑分支是在 ``libtorch_cuda.so`` 中或者在 cu 文件中，我们则可以通过 patch 直接修改源码解决。

.. code-block:: text
   :linenos:

   // torch_supa/patch_privateuse1/aten/src/ATen/native/cuda/Resize.cpp.patch
   @@ -30,7 +30,11 @@ void resize_bytes_cuda(StorageImpl* stor
      c10::cuda::CUDAGuard guard(device.index());
      at::DataPtr data = allocator->allocate(size_bytes);
      if (storage->data_ptr()) {
   +#ifdef USE_PRIVATEUSE1
   +    at::globalContext().lazyInitPrivateUse1();
   +#elsetorch_supa/
      at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
   +#endif


torch-supa-op Patch 逻辑
~~~~~~~~~~~~~~~~~~~~~~~~~~

`torch_supa` 仓中维护了一系列 patch 文件夹，torch_supa/patch_privateuse1 主要是 ``c10_cuda`` 相关的头文件 patch，
torch_supa/patch_torch2x 主要是不同 pytorch 版本的算子 patch。这些 patch 可能出现遗漏或者逻辑错误。可以通过 gdb 或者查看源码的方式到
torch_supa/build/_deps/pytorch-src 定位问题代码位置。

br_pytorch2 基建算子
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`br_pytorch2` 仓库中实现并注册了一些基建算子，如 ``_pin_memory``、``copy_``、``empty``、``record_stream`` 等。这其中 ``copy_`` 实现了包括
H2D，D2H，D2D，P2P，TypeCast, Contiguous等多个复杂功能。这部分实现逻辑由 `br_pytorch2` 开发并维护，可能存在一些潜在的问题。
