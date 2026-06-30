.. _native-api-torch_distributed_checkpoint:

torch.distributed.checkpoint
=============================

.. list-table::
   :header-rows: 1
   :widths: 60 15 25

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.distributed.checkpoint.load_state_dict``
     - 
     - 
   * - ``torch.distributed.checkpoint.save_state_dict``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader.prepare_global_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader.prepare_local_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader.read_data``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader.read_metadata``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageReader.set_up_storage_reader``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter.finish``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter.prepare_global_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter.prepare_local_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter.set_up_storage_writer``
     - 
     - 
   * - ``torch.distributed.checkpoint.StorageWriter.write_data``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.commit_tensor``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.create_global_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.create_local_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.finish_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.load_bytes``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.resolve_tensor``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlanner.set_up_planner``
     - 
     - 
   * - ``torch.distributed.checkpoint.LoadPlan``
     - 
     - 
   * - ``torch.distributed.checkpoint.ReadItem``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner.create_global_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner.create_local_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner.finish_plan``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner.resolve_data``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlanner.set_up_planner``
     - 
     - 
   * - ``torch.distributed.checkpoint.SavePlan``
     - 
     - 
   * - ``torch.distributed.checkpoint.WriteItem``
     - 
     - 
   * - ``torch.distributed.checkpoint.FileSystemReader``
     - 
     - 
   * - ``torch.distributed.checkpoint.FileSystemWriter``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultSavePlanner``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultSavePlanner.lookup_object``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultSavePlanner.transform_object``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultLoadPlanner``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor``
     - 
     - 
   * - ``torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor``
     - 
     - 
