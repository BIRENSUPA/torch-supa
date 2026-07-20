概述
==============================================================

简介
---------

Torch SUPA 是壁仞科技为其 GPU 生态适配的 PyTorch 框架插件（以下简称 torch_supa），旨在为 PyTorch 用户提供壁仞 GPU 的高性能计算支持。
torch_supa 通过 PyTorch 社区提供的 PrivateUse1 后端集成机制接入 SUPA 后端，继承了 PyTorch 的大部分特性，并且针对壁仞 BR2xx 系列芯片的特点进行了优化。

.. only:: html

   .. figure:: images/torch_supa_framework.svg
      :align: center
      :alt: torch_supa架构图

      torch_supa架构图

.. only:: latex

   .. figure:: images/torch_supa_framework.pdf
      :align: center
      :alt: torch_supa架构图

      torch_supa架构图

.. only:: not html and not latex

   .. figure:: images/torch_supa_framework.png
      :align: center
      :alt: torch_supa架构图

      torch_supa架构图

CUDA兼容性
------------

torch_supa 提供 CUDA 兼容能力，目标是在尽量少修改 PyTorch CUDA 代码的前提下，将计算迁移到壁仞 SUPA 后端执行。对于已有训练或推理脚本，常见的 ``cuda`` 设备写法会被兼容映射到 ``supa`` 设备，例如 ``torch.device("cuda")``、 ``device="cuda"``、 ``Tensor.cuda()`` 和 ``Module.cuda()`` 等接口在 torch_supa 环境中可自动转换为 SUPA 设备。

该兼容机制主要覆盖以下场景：

- **设备写法兼容**：保留原有 ``cuda`` 设备字符串和 ``.cuda()`` 调用方式，运行时自动映射到 ``supa:0``、 ``supa:1`` 等 SUPA 设备。
- **接口命名兼容**：常用 ``torch.cuda.*`` 接口会映射到 ``torch.supa.*`` 对应实现。
- **算子调用兼容**：通过 PyTorch PrivateUse1 后端注册机制，将 PyTorch 原生算子调度到 SUPA 后端实现。PyTorch原生算子可保持原有 ``torch.xxx``、 ``Tensor.xxx`` 和 ``torch.nn`` 调用方式。
- **生态迁移兼容**：对于依赖 CUDA 写法的社区库，可配合 SUDA 工具和 torch_supa 的Extension功能快速迁移。

对于新开发代码，推荐显式使用 ``device="supa"`` 或 ``torch.device("supa")``，以便清晰表达运行后端；对于已有 CUDA 代码，可先依赖兼容性映射快速验证。

关键功能特性
-------------

- **适配壁仞GPU设备**：基于开源PyTorch，适配壁仞GPU设备，提供原生Python接口。
- **框架基础功能**：PyTorch动态图、自动微分、Profiling、优化器等。
- **自定义算子开发**：支持在PyTorch框架中添加自定义算子。
- **分布式训练**：支持原生分布式数据并行训练，包含单机多卡、多机多卡场景支持的集合通信原语，如Broadcast、AllReduce等。