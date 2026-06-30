# 安装和快速开始

## 环境要求

用户指南面向基于 Docker 的部署方式，依赖壁仞 SUPA 软件栈。

基础要求：

- Docker `>= 20.10.7`。
- Ubuntu 22.04 或 Ubuntu 24.04。
- 文档中的 wheel 矩阵使用 Python 3.10。
- 安装 CPU PyTorch wheel，GPU 执行由 SUPA backend 提供。

用户指南列出的 PyTorch / torchvision 组合：

| PyTorch | torchvision | Python |
| --- | --- | --- |
| `2.10.0` | `0.25.0` | `3.10` |
| `2.9.0` | `0.24.0` | `3.10` |
| `2.8.0` | `0.23.0` | `3.10` |
| `2.6.0` | `0.21.0` | `3.10` |

来源：`docs/user_guider/source/chapter02_torch_supa_environments.rst`。

## Docker 流程

1. 加载离线壁仞 PyTorch Docker 镜像：

```bash
docker load -i <birensupa-sdk>.tar
docker image list
```

2. 挂载壁仞设备并启动容器：

```bash
docker run -itd --device /dev/biren/card_X --privileged --name <container> \
  <biren-full-stack-image>:<lkg-version>
```

3. 进入容器：

```bash
docker exec -it <container> bash
```

4. 安装 SUDA 并加载 SDK 环境：

```bash
cd <your_lkg_pack>/full-stack
pip install suda-<suda_version>-linux_x86_64.whl
source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
```

## 安装 torch_supa

使用 SDK 包里的 release wheel：

```bash
cd <your_lkg_pack>/full-stack
pip install torch_supa-1.0.0.<torch_version>+br2xx-cp310-cp310-linux_x86_64.whl
```

也可以从源码构建：

```bash
git clone --recursive https://github.com/birentech/torch-supa.git
cd torch-supa
pip install -r requirements.txt
suda init
[DEBUG=on][BUILD_WITHOUT_BCCL=on] python3 setup.py bdist_wheel
pip install dist/torch_supa*.whl
```

## 验证

```python
import torch
print(torch.supa.is_available())
```

成功 import 后，可能看到 `torch.Tensor.cuda`、`torch.nn.Module.cuda`、`torch.cuda.*` 和 distributed NCCL 风格 API 被替换或重定向到 SUPA/BCCL 等价实现的提示。

## 快速迁移

PyTorch 2.5 及之后版本中，已有 CUDA 风格代码通常可以继续保留 CUDA device 写法：

```python
import torch

x = torch.rand(3, 4, device="cuda")
print(x.device)  # supa:0
```

关键迁移点：

- `torch.device("cuda")` 会映射到 SUPA 设备。
- `.cuda()` 会把 tensor/module 移到 SUPA。
- `torch.cuda.*` API 会重定向到 `torch.supa.*` 兼容 API。
- 对 PyTorch 2.5 之前版本，需要按用户指南显式 import `torch_supa` 和 `transfer_to_supa`。

来源：`docs/user_guider/source/chapter03_torch_supa_quickstart.rst`。
