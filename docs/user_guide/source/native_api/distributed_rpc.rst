.. _native-api-distributed_rpc:

Distributed RPC Framework
===============================

.. list-table::
   :header-rows: 1
   :widths: 60 15 25

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.distributed.rpc.init_rpc``
     - 
     - 
   * - ``torch.distributed.rpc.rpc_sync``
     - 
     - 
   * - ``torch.distributed.rpc.rpc_async``
     - 
     - 
   * - ``torch.distributed.rpc.remote``
     - 
     - 
   * - ``torch.distributed.rpc.get_worker_info``
     - 
     - 
   * - ``torch.distributed.rpc.shutdown``
     - 
     - 
   * - ``torch.distributed.rpc.WorkerInfo``
     - 
     - 
   * - ``torch.distributed.rpc.WorkerInfo.id``
     - 
     - 
   * - ``torch.distributed.rpc.WorkerInfo.name``
     - 
     - 
   * - ``torch.distributed.rpc.functions.async_execution``
     - 
     - 
   * - ``torch.distributed.rpc.BackendType``
     - 
     - 
   * - ``torch.distributed.rpc.RpcBackendOptions``
     - 
     - 
   * - ``torch.distributed.rpc.RpcBackendOptions.init_method``
     - 
     - 
   * - ``torch.distributed.rpc.RpcBackendOptions.rpc_timeout``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.device_maps``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.devices``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.init_method``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.num_worker_threads``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.rpc_timeout``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map``
     - 
     - 
   * - ``torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.backward``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.confirmed_by_owner``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.is_owner``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.local_value``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.owner``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.owner_name``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.remote``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.rpc_async``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.rpc_sync``
     - 
     - 
   * - ``torch.distributed.rpc.PyRRef.to_here``
     - 
     - 
   * - ``torch.distributed.nn.api.remote_module.RemoteModule``
     - 
     - 
   * - ``torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref``
     - 
     - 
   * - ``torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters``
     - 
     - 
   * - ``torch.distributed.autograd.backward``
     - 
     - 
   * - ``torch.distributed.autograd.context``
     - 
     - 
   * - ``torch.distributed.autograd.get_gradients``
     - 
     - 

