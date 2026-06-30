算子归属及注册流程
=========================================================

算子归属
----------

任何算子优化或者算子新增需求，均需要先明确算子所属仓库，目前算子归属有以下三类：

1. 通过原生 torch.xxx 调用的算子，如仅需少量源码 patch 则归属 ``torch-supa`` 仓, 特别地目前 torchvision 算子也在 `torch-supa` 仓维护
2. 通过原生 torch.xxx 调用的算子，因性能等原因需 supa 或者算子库重写则归属 ``br_pytorch2`` 仓
3. 其他自定义或者三方库算子归属 ``torch_supa_ext`` 仓

其中 `torch-supa` 仓算子仅维护 patch 文件，无须额外注册，算子注册由 `br_pytorch2` 仓解析 native_functions.yaml 生成相应的 RegisterSUPANative.cpp 文件完成。
`torch_supa_ext` 仓算子则遵循该仓库规则即可。本文档只阐述 `br_pytorch2` 仓算子注册流程。

br_pytorch2 仓算子注册
---------------------------

supa_native_functions.yaml 算子注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

注册算子到 supa_native_functions.yaml，算子名称需存在于 native_functions.yaml。

其中 ``supported`` 列表中的算子正常注册到 PrivateUse1 DispatchKey，``autograd`` 列表中的算子注册到 AutogradPrivateUse1 DispatchKey 从而允许重写反向实现。
相比于 `br_pytorch` 旧框架新增了 ``perf_supported`` 列表，该列表中所有算子均在 ``torch-supa`` 中有完整的功能实现，
仅因性能原因需要覆盖原有实现，后续可通过只读环境变量进行高性能 SUPA 算子和 Native 算子切换 [``功能开发中``]。

.. code-block:: yaml
   :linenos:

   # torch_supa/csrc/aten/supa_native_functions.yaml
   backend: SUPA
   cpp_namespace: at::supa
   supported:
      - _addmm_activation.out
      - _pin_memory
      - _cdist_forward
   perf_supported:  # 开发中
      - xxxx
   autograd:
      - renorm


at::supa::SUPANativeFunctions 算子定义
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_supa/csrc/aten/ops/ 路径下添加 at::supa::SUPANativeFunctions::xxxOp 的函数定义，:ref:`Structured OP` 类型可以使用 ``SUPA_IMPL_FUNC``，
则可复用 torch 原生的 meta 函数进行算子预处理。``autograd`` 列表中的算子由于需要定义反向，需要继承 ``torch::autograd::Function`` 去实现自动微分。

三种定义方式示例：

.. code-block:: cpp
   :linenos:

   // torch_supa/csrc/aten/ops/Blas.cpp
   SUPA_IMPL_FUNC(_addmm_activation)
   (const Tensor& self,
   const Tensor& mat1,
   const Tensor& mat2,
   const Scalar& beta,
   const Scalar& alpha,
   bool use_gelu,
   const Tensor& result) {
   addmm_out_impl(
         const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha, use_gelu ? Activation::GELU : Activation::RELU);
   }

   // torch_supa/csrc/aten/ops/Distance.cpp
   Tensor SUPANativeFunctions::_cdist_forward(const Tensor &x1, const Tensor &x2,
                                    const double p,
                                    c10::optional<int64_t> compute_mode) {
      auto maybe_outnames = namedinference::compute_cdist_outnames(x1, x2);
      auto result = [&]() {
         NoNamesGuard guard;
         return cdist_impl(x1, x2, p, compute_mode);
      }();
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
   }

   // torch_supa/csrc/aten/ops/Normalization.cpp
   class RenormFunction : public torch::autograd::Function<RenormFunction> {
   public:
   static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const Tensor &self, const Scalar &p, int64_t dim,
                              const Scalar &maxnorm);
   static std::vector<at::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                           std::vector<at::Tensor> grad_outputs);
   };

   Tensor SUPANativeFunctions::renorm(const Tensor &self, const Scalar &p,
                                    int64_t dim, const Scalar &maxnorm) {
   return RenormFunction::apply(self, p, dim, maxnorm);
   }


supa kernel 实现
~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_supa/csrc/aten/ops/kernels 下面添加相应的 SUPA kernel 实现，br_pytorch2 暂未计划迁移 1xx 框架上的 Ptfrontend 和 kernel selection 功能。
at::supa::SUPANativeFunctions 下的算子可以直接调用 supa kernel 实现。
