环境变量
==============================================================

torch_supa 提供了一系列环境变量用于控制日志、性能和行为配置。这些环境变量通常在程序启动时读取，建议在启动 Python 进程前，或至少在导入 ``torch`` / torch_supa 前设置。运行期间修改环境变量通常不会影响已初始化的组件。

.. list-table:: 环境变量说明
   :header-rows: 1
   :widths: 50 50

   * - 环境变量
     - 功能
   * - :ref:`BRTB_LOG_LEVEL`
     - 设置日志级别，如 ``info``、 ``warning``、 ``error``、 ``debug``、 ``empty``。
   * - :ref:`BRTB_LOG_BACKEND`
     - 设置日志后端，如 ``glog``、 ``stdout``、 ``empty``。
   * - :ref:`BRTB_LOG_DIR`
     - 设置日志文件保存路径，默认为 ``./logs``。
   * - :ref:`BRTB_ENABLE_REALTIME_LOG`
     - 设置是否实时打印日志，默认为 ``false``。
   * - :ref:`BRTB_REALTIME_LOG_SIZE`
     - 设置实时打印日志的大小阈值，单位为 ``KB``，默认为 12800。
   * - :ref:`BRTB_ENABLE_SIGNAL_HANDLING`
     - 设置是否开启信号处理，Release 模式默认为 ``false``。
   * - :ref:`BRTB_ENABLE_FLUSH_LOG_INSTANTLY`
     - 设置是否立即刷新日志，Release 模式默认为 ``false``。
   * - :ref:`BRTB_ASYNC_LOGGER_QUEUE_SIZE`
     - 设置异步日志队列大小，默认为 1048576。
   * - :ref:`BRTB_SUBLAS_PREFERRED_BACKEND`
     - 设置 BLAS 计算后端，默认为 Sublas。
   * - :ref:`BRTB_ENABLE_NATIVE_OP`
     - 设置是否使用原生算子，默认为 ``false``。
   * - :ref:`BRTB_TRANSFER_SILENCE`
     - 设置是否打印 ``transfer_to_supa`` 日志，默认为 ``false``。

日志配置
-----------

.. _BRTB_TRANSFER_SILENCE:

``BRTB_TRANSFER_SILENCE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

设置是否打印 ``transfer_to_supa`` 日志。当值为 ``ON``、``1``、``YES``、``TRUE`` 或 ``Y`` 时，不打印相关日志信息。

**默认值**

``false``。

**使用示例**

在启动 Python 进程前设置环境变量：

.. code-block:: shell

   # 打开 transfer_to_supa 日志静默模式
   export BRTB_TRANSFER_SILENCE=1
   python your_script.py

也可以在导入 ``torch`` 前在 Python 脚本中设置：

.. code-block:: python

   import os

   os.environ["BRTB_TRANSFER_SILENCE"] = "1"
   import torch

.. _BRTB_LOG_LEVEL:

``BRTB_LOG_LEVEL``
~~~~~~~~~~~~~~~~~~~

**功能说明**

设置 torch_supa 的日志输出级别。日志级别决定了哪些级别的日志信息会被输出。
通常支持的级别包括：

- ``info``: 输出信息级别及以上的日志
- ``warning``: 输出警告级别及以上的日志
- ``error``: 仅输出错误级别的日志

**使用示例**

.. code-block:: shell

   # 设置日志级别为 info
   export BRTB_LOG_LEVEL=info

   # 设置日志级别为 warning
   export BRTB_LOG_LEVEL=warning

在 Python 脚本中设置时，应在导入 ``torch`` 前完成：

.. code-block:: python

   import os

   os.environ["BRTB_LOG_LEVEL"] = "info"
   import torch

.. _BRTB_LOG_BACKEND:

``BRTB_LOG_BACKEND``
~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

设置日志系统的后端实现。torch_supa 支持多种日志后端，默认使用 ``glog``。

**使用示例**

.. code-block:: shell

   # 使用 glog 作为日志后端
   export BRTB_LOG_BACKEND=glog

.. _BRTB_LOG_DIR:

``BRTB_LOG_DIR``
~~~~~~~~~~~~~~~~~~~

**功能说明**

设置日志文件的保存目录。日志系统会在此目录下创建日志文件，记录运行时信息。

**使用示例**

.. code-block:: shell

   # 设置日志目录
   export BRTB_LOG_DIR=/var/log/torch_supa

   # 使用当前目录下的 logs 文件夹
   export BRTB_LOG_DIR=./logs

.. _BRTB_ENABLE_REALTIME_LOG:

``BRTB_ENABLE_REALTIME_LOG``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

控制是否启用实时日志打印功能。启用后，日志会立即输出到控制台，便于调试和问题定位。
在大规模训练场景下，建议关闭以提高性能。

**使用示例**

.. code-block:: shell

   # 启用实时日志
   export BRTB_ENABLE_REALTIME_LOG=true

   # 禁用实时日志（默认）
   export BRTB_ENABLE_REALTIME_LOG=false

.. _BRTB_REALTIME_LOG_SIZE:

``BRTB_REALTIME_LOG_SIZE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

设置实时日志的大小阈值，单位为 KB。当日志文件大小超过此阈值时，会触发相应的处理机制。

**使用示例**

.. code-block:: shell

   # 设置实时日志大小为 25600 KB (25 MB)
   export BRTB_REALTIME_LOG_SIZE=25600

.. _BRTB_ENABLE_SIGNAL_HANDLING:

``BRTB_ENABLE_SIGNAL_HANDLING``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

控制是否启用信号处理功能。启用后，torch_supa 会捕获系统信号（如 ``SIGSEGV``、 ``SIGABRT`` 等），
并在程序崩溃时输出诊断信息，便于问题定位。

**使用示例**

.. code-block:: shell

   # 启用信号处理
   export BRTB_ENABLE_SIGNAL_HANDLING=true

   # 禁用信号处理
   export BRTB_ENABLE_SIGNAL_HANDLING=false

.. _BRTB_ENABLE_FLUSH_LOG_INSTANTLY:

``BRTB_ENABLE_FLUSH_LOG_INSTANTLY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

控制是否立即刷新日志缓冲区。启用后，每条日志都会立即写入文件，确保日志完整性，
但会影响性能。在调试场景下建议启用，生产环境建议禁用。

**使用示例**

.. code-block:: shell

   # 启用立即刷新（调试时推荐）
   export BRTB_ENABLE_FLUSH_LOG_INSTANTLY=true

   # 禁用立即刷新（生产环境推荐）
   export BRTB_ENABLE_FLUSH_LOG_INSTANTLY=false

.. _BRTB_ASYNC_LOGGER_QUEUE_SIZE:

``BRTB_ASYNC_LOGGER_QUEUE_SIZE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

设置异步日志队列的大小。异步日志系统会将日志消息放入队列，由后台线程写入文件。
较大的队列可以处理更高的日志吞吐量，但会占用更多内存。

**使用示例**

.. code-block:: shell

   # 设置异步日志队列大小为 2M
   export BRTB_ASYNC_LOGGER_QUEUE_SIZE=2097152

性能调优
-----------

.. _BRTB_SUBLAS_PREFERRED_BACKEND:

``BRTB_SUBLAS_PREFERRED_BACKEND``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

设置 BLAS（基本线性代数子程序）计算的后端库。torch_supa 提供了多种 BLAS 后端实现，
不同的后端在不同场景下可能有性能差异。

可选值：

- Sublas: 壁仞优化的 BLAS 实现（默认）
- Sublaslt: 壁仞轻量级 BLAS 实现

也可以通过 Python API 动态设置：

.. code-block:: python

   import torch_supa.backends.supa as supa_backends

   # 查看当前 BLAS 后端
   supa_backends.preferred_blas_library()

   # 设置 BLAS 后端为 sublaslt
   supa_backends.preferred_blas_library("sublaslt")

**使用示例**

.. code-block:: shell

   # 设置 BLAS 后端为 Sublas
   export BRTB_SUBLAS_PREFERRED_BACKEND=Sublas

   # 设置 BLAS 后端为 Sublaslt
   export BRTB_SUBLAS_PREFERRED_BACKEND=Sublaslt

.. _BRTB_ENABLE_NATIVE_OP:

``BRTB_ENABLE_NATIVE_OP``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**功能说明**

控制是否优先使用 PyTorch 原生算子实现。默认情况下，torch_supa 会使用针对壁仞硬件优化的算子实现。
启用此选项后，会回退使用 PyTorch 原生实现，主要用于兼容性测试和问题排查。

**注意**: 启用此选项可能影响性能，仅建议在调试或遇到算子兼容性问题时使用。

**使用示例**

.. code-block:: shell

   # 启用原生算子（调试用）
   export BRTB_ENABLE_NATIVE_OP=true

   # 使用优化算子（默认）
   export BRTB_ENABLE_NATIVE_OP=false

使用建议
-----------

**开发调试场景**::

   export BRTB_LOG_LEVEL=debug
   export BRTB_LOG_BACKEND=stdout

**生产训练场景**::

   export BRTB_LOG_LEVEL=empty
   export BRTB_LOG_BACKEND=empty

**问题排查场景**::

   export BRTB_LOG_LEVEL=info
   export BRTB_LOG_BACKEND=stdout
   export BRTB_ENABLE_NATIVE_OP=true  # 如怀疑算子问题
