安装环境
==============================================================

.. _installation-environment:

为帮助您快速使用壁仞 torch_supa 进行深度学习任务，壁仞提供了包含完整 SUPA 软件栈的发布包和 Docker 镜像。建议优先使用正式发布渠道提供的 Docker 镜像完成环境部署；已有基础环境的用户也可以安装对应版本的 ``whl`` 包。

在开始安装torch_supa前，请确认已正确安装壁仞 SUPA 软件栈，具体安装步骤参见《BIRENSUPA SDK安装指南》。

基础环境要求
--------------

操作系统
~~~~~~~~~~

- Ubuntu 22.04
- Ubuntu 24.04

Docker 版本
~~~~~~~~~~~~~

- **Docker 版本**： ≥ 20.10.7
- **参考文档**：`Docker 官方安装手册 <https://docs.docker.com/desktop/setup/install/linux/>`_

完成 Docker 环境安装后，需将非 ``root`` 用户添加到 Docker 用户组：

.. code-block:: shell
   :linenos:

   # 将非 root 用户添加到 Docker 用户组
   sudo groupadd docker
   sudo gpasswd -a ${USER} docker
   sudo systemctl restart docker

   # 重新登录使权限生效
   newgrp docker

**PyTorch 版本要求**

torch_supa 当前支持的 PyTorch 与 torchvision 版本如下：

torch_supa 包命名为 ``torch_supa-<torch_supa_version>.<torch_version>+br2xx-cp310-cp310-linux_x86_64.whl``，其中：

- ``<torch_supa_version>`` 为 torch_supa 版本号，如 ``1.0.0``；
- ``<torch_version>`` 为 PyTorch 社区版本号转换。一般情况下按主版本、次版本和修订版本转换，例如 ``v2.6.0`` 转换成 ``20600``，``v2.9.0`` 转换成 ``20900``；

.. _pytorch-version-compatibility-table:

.. list-table:: PyTorch 版本配套表
   :header-rows: 1
   :widths: 15 15 15 10 45

   * - PyTorch Version
     - torch_supa Version
     - torchvision Version
     - Python Version
     - 安装命令
   * - v2.12.0
     - 1.0.0.21200
     - 0.27.0
     - 3.10
     - ``pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu``
   * - 2.11.0
     - 1.0.0.21100
     - 0.26.0
     - 3.10
     - ``pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu``
   * - 2.10.0
     - 1.0.0.21000
     - 0.25.0
     - 3.10
     - ``pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu``
   * - 2.9.0
     - 1.0.0.20900
     - 0.24.0
     - 3.10
     - ``pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu``
   * - 2.8.0
     - 1.0.0.20800
     - 0.23.0
     - 3.10
     - ``pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu``
   * - 2.6.0
     - 1.0.0.20600
     - 0.21.0
     - 3.10
     - ``pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu``

.. important::

   必须安装 **CPU 版本** 的 PyTorch，GPU 计算由 torch_supa 的 SUPA 后端提供。torch_supa whl 包版本应与 PyTorch 版本严格匹配，版本对应关系请参考 :doc:`chapter01_torch_supa_overview` 章节中的版本配套说明和正式发布包名称。

方式一：使用官方 Docker 镜像（推荐）
------------------------------------

适用场景
~~~~~~~~

该方式适用于希望快速获得完整 SUPA 软件栈的用户。发布镜像通常已包含匹配的驱动运行库、SUPA SDK、SUDA 依赖和 torch_supa 运行环境，可减少手动配置步骤。

获取镜像
~~~~~~~~

请从壁仞正式发布渠道获取 Docker 镜像包或镜像仓库地址。常见交付形式包括：

- 离线镜像包：``<birensupa-sdk>.tar``
- 镜像仓库地址：``<official-registry>/<image-name>:<sdk-version>``

.. note::

   ``<sdk-version>``、``<image-name>`` 和镜像校验信息以正式发布说明或客户交付包为准。请勿混用不同版本的驱动、SUPA SDK、PyTorch 和 torch_supa。

加载镜像
~~~~~~~~

如果获取的是离线镜像包，执行以下命令加载镜像：

.. code-block:: shell
   :linenos:

   docker load -i <birensupa-sdk>.tar
   docker image list

预期可以在 ``docker image list`` 输出中看到对应的 ``REPOSITORY``、``TAG`` 和 ``IMAGE ID``。

启动容器
~~~~~~~~

.. code-block:: shell
   :linenos:

   # 替换 card_X 为实际设备号，如 card_0；替换镜像名和 tag 为正式发布包中的名称
   docker run -itd --device /dev/biren/card_X --privileged --name <your-container-name> \
       <birensupa-docker-image>:<sdk-version>

进入容器：

.. code-block:: shell
   :linenos:

   docker exec -it <your-container-name> bash

.. note::

   ``--device /dev/biren/card_X`` 参数用于挂载壁仞设备到容器中， ``card_X`` 需要替换为实际设备编号。如需挂载多张卡，请按交付说明添加对应设备参数。

方式二：在已有环境中安装 whl
-----------------------------

适用场景
~~~~~~~~

该方式适用于已完成 SUPA SDK 和驱动部署，并希望在已有 Python 环境中安装 torch_supa 的用户。

安装 PyTorch CPU 版本
~~~~~~~~~~~~~~~~~~~~~

根据版本配套表选择对应 PyTorch 和 torchvision 版本。例如：

.. code-block:: shell
   :linenos:

   pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu

安装 SUDA 依赖并配置 SDK 环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从正式发布包中找到 SUDA whl 和环境配置脚本，执行：

.. code-block:: shell
   :linenos:

   cd <release-package>/full-stack
   pip install suda-<suda_version>-linux_x86_64.whl

   # 如 SDK 安装路径不同，请替换为实际路径
   source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh

安装 torch_supa
~~~~~~~~~~~~~~~~

在正式发布包中找到与 PyTorch 版本匹配的 torch_supa whl 包，使用 ``pip`` 安装：

.. code-block:: shell
   :linenos:

   cd <release-package>/full-stack
   pip install torch_supa-1.0.0.<torch_version>+br2xx-cp310-cp310-linux_x86_64.whl

其中 ``<torch_version>`` 需与 :ref:`PyTorch 版本配套表 <pytorch-version-compatibility-table>` 一致。例如 PyTorch v2.10.0 当前发布包使用 ``21000`` 编码。

方式三：源码构建安装（开发者参考）
----------------------------------

该方式面向需要二次开发、问题定位或自定义构建的高级用户和开发者。普通训练和推理用户建议优先使用官方 Docker 镜像或正式发布的 ``whl`` 包。

请从壁仞正式发布源码包或授权代码仓库获取 torch-supa 源码，然后执行：

.. code-block:: shell
   :linenos:

   # 1. 获取源码
   git clone --recursive <official-torch-supa-repository-url>
   cd torch-supa

   # 2. 安装依赖
   pip install -r requirements.txt
   suda init

   # 3. 编译
   python3 setup.py bdist_wheel

   # 4. 安装
   pip install dist/torch_supa*.whl

.. note::

   - ``DEBUG=on``：默认值未off, 编译 Debug 版本，包含调试符号。
   - ``USE_FLASH_ATTENTION=off``：默认值为on, 不编译 FLASH_ATTENTION 算子库。
   - 源码构建依赖完整的编译工具链和匹配版本的 SUPA SDK，如无二次开发需求，不建议作为首选安装方式。

验证安装
---------

安装完成后，可以通过以下方式验证安装是否成功：

.. code-block:: python
   :linenos:

   import torch

   print(torch.supa.is_available())
   x = torch.rand(2, 3, device="supa")
   print(x.device)
   print(x + 1)

预期输出中 ``torch.supa.is_available()`` 返回 ``True``，张量设备显示为 ``supa:0`` 或等价 SUPA 设备。

如果导入 ``torch`` 时显示以下信息，说明 torch_supa 已正确加载并自动切换到 SUPA 后端：

.. code-block:: text

   The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.supa and torch.nn.Module.supa now..
   The backend in torch.distributed.init_process_group set to bccl now..
   The torch.cuda.* and torch.cuda.amp.* are replaced with torch.supa.* and torch.supa.amp.* now..

.. warning::

   上述提示信息是正常输出，表示 torch_supa 已正确初始化，无需担心。

常见检查项
-----------

如果验证失败，请优先检查：

- PyTorch、torchvision 和 torch_supa 版本是否匹配。
- 是否安装 CPU 版本 PyTorch。
- 是否已执行 ``brsw_set_env.sh``，且路径与实际 SUPA SDK 安装位置一致。
- 容器是否正确挂载 ``/dev/biren/card_X`` 设备。
- 当前用户是否具备访问 Docker 和壁仞设备的权限。
- 正式发布包中的 SUDA、SUPA SDK、驱动和 torch_supa 是否来自同一配套版本。
