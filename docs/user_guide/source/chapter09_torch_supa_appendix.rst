附录
==============================================================

.. _glossary:

术语表
---------------

.. list-table:: 术语表
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
     - SUPA Adaptor for CUDA，SUPA兼容CUDA桥接工具
   * - AMP
     - Automatic Mixed Precision，自动混合精度训练技术
   * - BLAS
     - Basic Linear Algebra Subprograms，基本线性代数子程序库
   * - Sublas
     - 壁仞优化的 BLAS 实现库
   * - CPU
     - Central Processing Unit，中央处理器
   * - DDP
     - Distributed Data Parallel，分布式数据并行
   * - FFT
     - Fast Fourier Transform，快速傅里叶变换
   * - Flash Attention
     - Flash Attention融合算子是一种应用于模型加速的具有IO感知的精确注意力算法
   * - GDB
     - GNU Debugger，GNU工程调试器
   * - OOM
     - Out Of Memory，内存不足
   * - OP
     - 算子（Operator，简称OP），是深度学习算法中执行特定数学运算或操作的基础单元
