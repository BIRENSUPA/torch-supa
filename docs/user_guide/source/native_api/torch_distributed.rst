.. _native-api-torch_distributed:

torch.distributed
===================
.. list-table::
   :header-rows: 1
   :widths: 60 15 25

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.distributed.is_available``
     - 是
     - 
   * - ``torch.distributed.init_process_group``
     - 是
     - 
   * - ``torch.distributed.is_initialized``
     - 是
     - 
   * - ``torch.distributed.is_mpi_available``
     - 是
     - 
   * - ``torch.distributed.is_nccl_available``
     - 是
     - 
   * - ``torch.distributed.is_gloo_available``
     - 是
     - 
   * - ``torch.distributed.is_torchelastic_launched``
     - 是
     - 
   * - ``torch.distributed.Backend``
     - 是
     - 
   * - ``torch.distributed.Backend.register_backend``
     - 
     - 
   * - ``torch.distributed.get_backend``
     - 是
     - 
   * - ``torch.distributed.get_rank``
     - 是
     - 
   * - ``torch.distributed.get_world_size``
     - 是
     - 
   * - ``torch.distributed.Store``
     - 是
     - 
   * - ``torch.distributed.TCPStore``
     - 是
     - 
   * - ``torch.distributed.HashStore``
     - 是
     - 
   * - ``torch.distributed.FileStore``
     - 是
     - 
   * - ``torch.distributed.PrefixStore``
     - 是
     - 
   * - ``torch.distributed.Store.set``
     - 是
     - 
   * - ``torch.distributed.Store.get``
     - 是
     - 
   * - ``torch.distributed.Store.add``
     - 是
     - 
   * - ``torch.distributed.Store.compare_set``
     - 是
     - 
   * - ``torch.distributed.Store.wait``
     - 是
     - 
   * - ``torch.distributed.Store.num_keys``
     - 是
     - 
   * - ``torch.distributed.Store.delete_key``
     - 是
     - 
   * - ``torch.distributed.Store.set_timeout``
     - 是
     - 
   * - ``torch.distributed.new_group``
     - 是
     - 
   * - ``torch.distributed.get_group_rank``
     - 是
     - 
   * - ``torch.distributed.get_global_rank``
     - 是
     - 
   * - ``torch.distributed.get_process_group_ranks``
     - 是
     - 
   * - ``torch.distributed.send``
     - 是
     - 
   * - ``torch.distributed.recv``
     - 是
     - 
   * - ``torch.distributed.isend``
     - 
     - 
   * - ``torch.distributed.irecv``
     - 
     - 
   * - ``torch.distributed.batch_isend_irecv``
     - 
     - 
   * - ``torch.distributed.P2POp``
     - 是
     - 
   * - ``torch.distributed.broadcast``
     - 是
     - 
   * - ``torch.distributed.broadcast_object_list``
     - 是
     - 
   * - ``torch.distributed.all_reduce``
     - 是
     - 
   * - ``torch.distributed.reduce``
     - 
     - 
   * - ``torch.distributed.all_gather``
     - 是
     - 
   * - ``torch.distributed.all_gather_into_tensor``
     - 是
     - 
   * - ``torch.distributed.all_gather_object``
     - 是
     - 
   * - ``torch.distributed.gather``
     - 
     - 
   * - ``torch.distributed.gather_object``
     - 
     - 
   * - ``torch.distributed.scatter``
     - 
     - 
   * - ``torch.distributed.scatter_object_list``
     - 
     - 
   * - ``torch.distributed.reduce_scatter``
     - 是
     - 
   * - ``torch.distributed.reduce_scatter_tensor``
     - 是
     - 
   * - ``torch.distributed.all_to_all_single``
     - 是
     - 
   * - ``torch.distributed.all_to_all``
     - 是
     - 
   * - ``torch.distributed.barrier``
     - 是
     - 
   * - ``torch.distributed.monitored_barrier``
     - 
     - 
   * - ``torch.distributed.ReduceOp``
     - 是
     - 
   * - ``torch.distributed.reduce_op``
     - 是
     - 
   * - ``torch.distributed.broadcast_multigpu``
     - 
     - 
   * - ``torch.distributed.all_reduce_multigpu``
     - 
     - 
   * - ``torch.distributed.reduce_multigpu``
     - 
     - 
   * - ``torch.distributed.all_gather_multigpu``
     - 
     - 
   * - ``torch.distributed.reduce_scatter_multigpu``
     - 
     - 
   * - ``torch.distributed.DistBackendError``
     - 是
     - 

