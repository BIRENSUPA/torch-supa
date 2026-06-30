快速开始
==============================================================

本章节提供两个示例，帮助您快速验证 torch_supa 环境并体验将 GPU 模型脚本迁移到壁仞 SUPA 平台的流程：

- **最小 SUPA 示例**：不依赖网络和外部数据集，用于快速验证安装是否可用。
- **MNIST 迁移示例**：保留 ``cuda`` 兼容写法，用于展示已有 PyTorch CUDA 脚本迁移到 SUPA 平台的方式。

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
       x = torch.randn(4, 4, device=device, requires_grad=True)
       w = torch.randn(4, 4, device=device, requires_grad=True)

       y = x @ w
       loss = y.square().mean()
       loss.backward()

       print(f"device: {y.device}")
       print(f"loss: {loss.item():.6f}")
       print(f"x.grad is not None: {x.grad is not None}")
       print(f"w.grad is not None: {w.grad is not None}")


   if __name__ == "__main__":
       main()

执行脚本：

.. code-block:: shell

   python quick_check.py

预期输出类似如下：

.. code-block:: text

   torch.supa.is_available(): True
   device: supa:0
   loss: 3.141592
   x.grad is not None: True
   w.grad is not None: True

.. note::

   最小示例推荐新代码显式使用 ``device="supa"``，语义更清晰。如果当前环境中 torch_supa 未自动加载，请先确认安装版本是否匹配；对于需要手动加载的旧环境，可在脚本开头导入 torch_supa。

MNIST 迁移示例
---------------

本示例展示如何将已有 PyTorch CUDA 训练脚本迁移到 SUPA 平台。示例保留 ``torch.device("cuda")`` 写法，用于展示 torch_supa 对 CUDA 设备写法的兼容映射。

.. warning::

   该示例会下载 MNIST 数据集并在当前目录下创建 ``data`` 目录和 ``model.pth`` 文件。离线环境请提前准备数据集，或先运行上文的最小 SUPA 示例验证环境。

创建 ``train.py`` 脚本，写入以下基于 MNIST 数据集的训练代码：

.. code-block:: python
   :linenos:

   import os
   import torch
   from torch.utils.data import DataLoader
   from torchvision.datasets import mnist
   from torch import nn
   from torch import optim
   from torchvision import transforms
   from torch.optim.lr_scheduler import StepLR
   import torch.nn.functional as F


   class Net(nn.Module):
       """简单的 CNN 模型"""

       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x = F.relu(x)
           x = F.max_pool2d(x, 2)
           x = self.dropout1(x)
           x = torch.flatten(x, 1)
           x = self.fc1(x)
           x = F.relu(x)
           x = self.dropout2(x)
           x = self.fc2(x)
           output = F.log_softmax(x, dim=1)
           return output


   def train(model, train_loader, optimizer, epoch, device):
       """模型训练"""
       model.train()
       for batch_idx, (data, target) in enumerate(train_loader):
           data, target = data.to(device), target.to(device)
           optimizer.zero_grad()
           output = model(data)
           loss = F.nll_loss(output, target)
           loss.backward()
           optimizer.step()
           if batch_idx % 100 == 0:
               print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                     f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


   def validate(model, test_loader, device):
       """模型验证"""
       model.eval()
       test_loss = 0
       correct = 0
       with torch.no_grad():
           for data, target in test_loader:
               data, target = data.to(device), target.to(device)
               output = model(data)
               test_loss += F.nll_loss(output, target, reduction='sum').item()
               pred = output.argmax(dim=1, keepdim=True)
               correct += pred.eq(target.view_as(pred)).sum().item()

       test_loss /= len(test_loader.dataset)
       print(f'\nTest set: Average loss: {test_loss:.4f}, '
             f'Accuracy: {correct}/{len(test_loader.dataset)} '
             f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


   def main():
       # 数据预处理
       transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])

       # 加载数据集。离线环境请提前将 MNIST 数据放入 dataset_dir。
       dataset_dir = './data'
       os.makedirs(dataset_dir, exist_ok=True)

       train_dataset = mnist.MNIST(dataset_dir, train=True, download=True, transform=transform)
       test_dataset = mnist.MNIST(dataset_dir, train=False, transform=transform)

       train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
       test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

       # 创建模型并移动到设备。cuda 写法会由 torch_supa 兼容映射到 supa 设备。
       device = torch.device("cuda")
       model = Net().to(device)
       optimizer = optim.Adadelta(model.parameters(), lr=1.0)
       scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

       # 训练
       epochs = 1
       for epoch in range(1, epochs + 1):
           train(model, train_loader, optimizer, epoch, device)
           validate(model, test_loader, device)
           scheduler.step()

       # 保存模型
       torch.save({
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'epoch': epochs,
       }, 'model.pth')
       print('Model saved to model.pth')


   if __name__ == '__main__':
       main()

执行训练
-----------

执行以下命令启动训练：

.. code-block:: bash

   python train.py

训练过程中会打印训练损失和验证准确率。训练完成后，脚本会自动保存模型到 ``model.pth`` 文件。

代码说明
-----------

**关键迁移点**：

1. **新代码推荐显式使用 SUPA 设备**：新编写的脚本建议使用 ``torch.device("supa")`` 或 ``device="supa"``，便于清楚表达代码运行在 SUPA 后端。

2. **CUDA 兼容迁移**：在当前支持的 PyTorch 版本中， ``.cuda()`` 和 ``torch.device("cuda")`` 会自动映射到 SUPA 设备，原有 CUDA 代码通常无需大幅修改即可运行。

3. **设备检查**：可以通过以下方式确认当前使用的设备：

   .. code-block:: python

      >>> import torch
      >>> x = torch.rand(3, 4, device="cuda")
      >>> print(x.device)
      supa:0

4. **接口兼容**： ``torch.cuda.*`` 接口会自动替换为 ``torch.supa.*`` 接口，保持 API 兼容。若遇到行为差异，请参考 :doc:`chapter05_torch_supa_api` 中的支持矩阵和限制说明。

.. note::

   对于需要手动加载兼容转换能力的环境，可在脚本开头添加以下导入：

   .. code-block:: python

      import torch_supa
      from torch_supa.contrib import transfer_to_supa
