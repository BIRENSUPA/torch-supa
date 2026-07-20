快速开始
==============================================================

本章节提供两个示例，帮助您快速验证 torch_supa 环境并体验将 GPU 模型脚本迁移到壁仞 SUPA 平台的流程：

- **最小 SUPA 示例**：不依赖网络和外部数据集，显式使用 ``torch.device("supa")``，用于快速验证安装是否可用。
- **MNIST 迁移示例**：保留 CUDA 兼容写法，用于展示已有 PyTorch CUDA 脚本迁移到 SUPA 平台的方式。

准备工作
-----------

按照 :ref:`安装环境 <installation-environment>` 章节的介绍，准备好壁仞 SUPA 环境，并安装好相应的依赖包。

最小 SUPA 示例
---------------

创建 ``quick_check.py`` 脚本，写入以下代码：

.. code-block:: python
   :linenos:

   import torch

   def main():
       print(f"torch.supa.is_available(): {torch.supa.is_available()}")
       device = torch.device("supa")
       x = torch.randn(4, 4, device=device)
       w = torch.randn(4, 4, device=device)
       y = x + w
       print(f"device: {y.device}")

   if __name__ == "__main__":
       main()

执行脚本：

.. code-block:: shell

   python quick_check.py

预期输出类似如下：

.. code-block:: text

   torch.supa.is_available(): True
   device: supa:0

MNIST 迁移示例
---------------

本示例展示如何将已有 PyTorch CUDA 训练脚本迁移到 SUPA 平台。示例保留 ``torch.device("cuda")`` 写法，用于展示 torch_supa 对 CUDA 设备写法的兼容映射。

.. warning::

   该示例会下载 MNIST 数据集并在当前目录下创建 ``data`` 目录和 ``model.pth`` 文件。离线环境请提前准备数据集，或先运行上文的最小 SUPA 示例验证环境。

创建 ``train.py`` 脚本，写入以下基于 MNIST 数据集的训练代码：

.. code-block:: shell
   :linenos:

   git clone https://github.com/pytorch/examples.git
   cd examples/mnist
   pip install -r requirements.txt
   python3 main.py

训练过程中会打印训练损失和验证准确率。训练完成后，脚本会自动保存模型到 ``model.pth`` 文件。
