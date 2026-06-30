.. _native-api-torch_cuda:

torch.cuda
==========

- 设备自动映射： ``cuda:0`` 自动映射到 ``supa:0``
- API 自动替换： ``torch.cuda.*`` 自动替换为 ``torch.supa.*``
- 张量设备兼容：支持 ``tensor.cuda()`` 和 ``tensor.supa()`` 两种写法

.. list-table::
   :header-rows: 1
   :widths: 40 10 40 10

   * - PyTorch API
     - 是否支持
     - SUPA API
     - 限制
   * - ``torch.cuda.StreamContext``
     - 是
     - ``torch.supa.StreamContext``
     -
   * - ``torch.cuda.can_device_access_peer``
     - 是
     - ``torch.supa.can_device_access_peer``
     -
   * - ``torch.cuda.check_error``
     - 是
     - ``torch.supa.check_error``
     -
   * - ``torch.cuda.current_blas_handle``
     - 是
     - ``torch.supa.current_blas_handle``
     -
   * - ``torch.cuda.current_device``
     - 是
     - ``torch.supa.current_device``
     -
   * - ``torch.cuda.current_stream``
     - 是
     - ``torch.supa.current_stream``
     -
   * - ``torch.cuda.cudart``
     - 是
     - ``torch.supa.cudart``
     -
   * - ``torch.cuda.default_stream``
     - 是
     - ``torch.supa.default_stream``
     -
   * - ``torch.cuda.device``
     - 是
     - ``torch.supa.device``
     -
   * - ``torch.cuda.device_count``
     - 是
     - ``torch.supa.device_count``
     -
   * - ``torch.cuda.device_memory_used``
     - 是
     - ``torch.supa.device_memory_used``
     -
   * - ``torch.cuda.device_of``
     - 是
     - ``torch.supa.device_of``
     -
   * - ``torch.cuda.get_arch_list``
     - 是
     - ``torch.supa.get_arch_list``
     -
   * - ``torch.cuda.get_device_capability``
     - 是
     - ``torch.supa.get_device_capability``
     -
   * - ``torch.cuda.get_device_name``
     - 是
     - ``torch.supa.get_device_name``
     -
   * - ``torch.cuda.get_device_properties``
     - 是
     - ``torch.supa.get_device_properties``
     -
   * - ``torch.cuda.get_gencode_flags``
     - 是
     - ``torch.supa.get_gencode_flags``
     -
   * - ``torch.cuda.get_stream_from_external``
     - 是
     - ``torch.supa.get_stream_from_external``
     -
   * - ``torch.cuda.get_sync_debug_mode``
     - 是
     - ``torch.supa.get_sync_debug_mode``
     -
   * - ``torch.cuda.init``
     - 是
     - ``torch.supa.init``
     -
   * - ``torch.cuda.ipc_collect``
     - 是
     - ``torch.supa.ipc_collect``
     -
   * - ``torch.cuda.is_available``
     - 是
     - ``torch.supa.is_available``
     -
   * - ``torch.cuda.is_bf16_supported``
     - 是
     - ``torch.supa.is_bf16_supported``
     -
   * - ``torch.cuda.is_initialized``
     - 是
     - ``torch.supa.is_initialized``
     -
   * - ``torch.cuda.is_tf32_supported``
     - 是
     - ``torch.supa.is_tf32_supported``
     -
   * - ``torch.cuda.memory_usage``
     - 是
     - ``torch.supa.memory_usage``
     -
   * - ``torch.cuda.set_device``
     - 是
     - ``torch.supa.set_device``
     -
   * - ``torch.cuda.set_stream``
     - 是
     - ``torch.supa.set_stream``
     -
   * - ``torch.cuda.set_sync_debug_mode``
     - 是
     - ``torch.supa.set_sync_debug_mode``
     -
   * - ``torch.cuda.stream``
     - 是
     - ``torch.supa.stream``
     -
   * - ``torch.cuda.synchronize``
     - 是
     - ``torch.supa.synchronize``
     -
   * - ``torch.cuda.utilization``
     - 是
     - ``torch.supa.utilization``
     -
   * - ``torch.cuda.temperature``
     - 是
     - ``torch.supa.temperature``
     -
   * - ``torch.cuda.power_draw``
     - 是
     - ``torch.supa.power_draw``
     -
   * - ``torch.cuda.clock_rate``
     - 是
     - ``torch.supa.clock_rate``
     -
   * - ``torch.cuda.AcceleratorError``
     - 是
     - ``torch.supa.AcceleratorError``
     -
   * - ``torch.cuda.OutOfMemoryError``
     - 是
     - ``torch.supa.OutOfMemoryError``
     -
   * - ``torch.cuda.get_rng_state``
     - 是
     - ``torch.supa.get_rng_state``
     -
   * - ``torch.cuda.get_rng_state_all``
     - 是
     - ``torch.supa.get_rng_state_all``
     -
   * - ``torch.cuda.set_rng_state``
     - 是
     - ``torch.supa.set_rng_state``
     -
   * - ``torch.cuda.set_rng_state_all``
     - 是
     - ``torch.supa.set_rng_state_all``
     -
   * - ``torch.cuda.manual_seed``
     - 是
     - ``torch.supa.manual_seed``
     -
   * - ``torch.cuda.manual_seed_all``
     - 是
     - ``torch.supa.manual_seed_all``
     -
   * - ``torch.cuda.seed``
     - 是
     - ``torch.supa.seed``
     -
   * - ``torch.cuda.seed_all``
     - 是
     - ``torch.supa.seed_all``
     -
   * - ``torch.cuda.initial_seed``
     - 是
     - ``torch.supa.initial_seed``
     -
   * - ``torch.cuda.comm.broadcast``
     - 是
     - ``torch.supa.comm.broadcast``
     -
   * - ``torch.cuda.comm.broadcast_coalesced``
     - 是
     - ``torch.supa.comm.broadcast_coalesced``
     -
   * - ``torch.cuda.comm.reduce_add``
     - 是
     - ``torch.supa.comm.reduce_add``
     -
   * - ``torch.cuda.comm.reduce_add_coalesced``
     - 是
     - ``torch.supa.comm.reduce_add_coalesced``
     -
   * - ``torch.cuda.comm.scatter``
     - 是
     - ``torch.supa.comm.scatter``
     -
   * - ``torch.cuda.comm.gather``
     - 是
     - ``torch.supa.comm.gather``
     -
   * - ``torch.cuda.Stream``
     - 是
     - ``torch.supa.Stream``
     -
   * - ``torch.cuda.Stream.wait_stream``
     - 是
     - ``torch.supa.Stream.wait_stream``
     -
   * - ``torch.cuda.Event``
     - 是
     - ``torch.supa.Event``
     -
   * - ``torch.cuda.Event.elapsed_time``
     - 是
     - ``torch.supa.Event.elapsed_time``
     -
   * - ``torch.cuda.Event.query``
     - 是
     - ``torch.supa.Event.query``
     -
   * - ``torch.cuda.Event.wait``
     - 是
     - ``torch.supa.Event.wait``
     -
   * - ``torch.cuda.ExternalStream``
     - 是
     - ``torch.supa.ExternalStream``
     -
   * - ``torch.cuda.is_current_stream_capturing``
     - 是
     - ``torch.supa.is_current_stream_capturing``
     -
   * - ``torch.cuda.graph_pool_handle``
     - 是
     - ``torch.supa.graph_pool_handle``
     -
   * - ``torch.cuda.CUDAGraph``
     - 是
     - ``torch.supa.SUPAGraph``
     -
   * - ``torch.cuda.graph``
     - 是
     - ``torch.supa.graph``
     -
   * - ``torch.cuda.make_graphed_callables``
     - 是
     - ``torch.supa.make_graphed_callables``
     -
   * - ``torch.cuda.memory.empty_cache``
     - 是
     - ``torch.supa.memory.empty_cache``
     -
   * - ``torch.cuda.memory.get_per_process_memory_fraction``
     - 是
     - ``torch.supa.memory.get_per_process_memory_fraction``
     -
   * - ``torch.cuda.memory.list_gpu_processes``
     - 是
     - ``torch.supa.memory.list_gpu_processes``
     -
   * - ``torch.cuda.memory.mem_get_info``
     - 是
     - ``torch.supa.memory.mem_get_info``
     -
   * - ``torch.cuda.memory.memory_stats``
     - 是
     - ``torch.supa.memory.memory_stats``
     -
   * - ``torch.cuda.memory.memory_stats_as_nested_dict``
     - 是
     - ``torch.supa.memory.memory_stats_as_nested_dict``
     -
   * - ``torch.cuda.memory.reset_accumulated_memory_stats``
     - 是
     - ``torch.supa.memory.reset_accumulated_memory_stats``
     -
   * - ``torch.cuda.memory.host_memory_stats``
     - 是
     - ``torch.supa.memory.host_memory_stats``
     -
   * - ``torch.cuda.memory.host_memory_stats_as_nested_dict``
     - 是
     - ``torch.supa.memory.host_memory_stats_as_nested_dict``
     -
   * - ``torch.cuda.memory.reset_accumulated_host_memory_stats``
     - 是
     - ``torch.supa.memory.reset_accumulated_host_memory_stats``
     -
   * - ``torch.cuda.memory.memory_summary``
     - 是
     - ``torch.supa.memory.memory_summary``
     -
   * - ``torch.cuda.memory.memory_snapshot``
     - 是
     - ``torch.supa.memory.memory_snapshot``
     -
   * - ``torch.cuda.memory.memory_allocated``
     - 是
     - ``torch.supa.memory.memory_allocated``
     -
   * - ``torch.cuda.memory.max_memory_allocated``
     - 是
     - ``torch.supa.memory.max_memory_allocated``
     -
   * - ``torch.cuda.memory.reset_max_memory_allocated``
     - 是
     - ``torch.supa.memory.reset_max_memory_allocated``
     -
   * - ``torch.cuda.memory.memory_reserved``
     - 是
     - ``torch.supa.memory.memory_reserved``
     -
   * - ``torch.cuda.memory.max_memory_reserved``
     - 是
     - ``torch.supa.memory.max_memory_reserved``
     -
   * - ``torch.cuda.memory.set_per_process_memory_fraction``
     - 是
     - ``torch.supa.memory.set_per_process_memory_fraction``
     -
   * - ``torch.cuda.memory.memory_cached``
     - 是
     - ``torch.supa.memory.memory_cached``
     -
   * - ``torch.cuda.memory.max_memory_cached``
     - 是
     - ``torch.supa.memory.max_memory_cached``
     -
   * - ``torch.cuda.memory.reset_max_memory_cached``
     - 是
     - ``torch.supa.memory.reset_max_memory_cached``
     -
   * - ``torch.cuda.memory.reset_peak_memory_stats``
     - 是
     - ``torch.supa.memory.reset_peak_memory_stats``
     -
   * - ``torch.cuda.memory.reset_peak_host_memory_stats``
     - 是
     - ``torch.supa.memory.reset_peak_host_memory_stats``
     -
   * - ``torch.cuda.memory.caching_allocator_alloc``
     - 是
     - ``torch.supa.memory.caching_allocator_alloc``
     -
   * - ``torch.cuda.memory.caching_allocator_delete``
     - 是
     - ``torch.supa.memory.caching_allocator_delete``
     -
   * - ``torch.cuda.memory.get_allocator_backend``
     - 是
     - ``torch.supa.memory.get_allocator_backend``
     -
   * - ``torch.cuda.CUDAPluggableAllocator``
     - 是
     - ``torch.supa.CUDAPluggableAllocator``
     -
   * - ``torch.cuda.memory.change_current_allocator``
     - 是
     - ``torch.supa.memory.change_current_allocator``
     -
   * - ``torch.cuda.MemPool``
     - 是
     - ``torch.supa.MemPool``
     -
   * - ``torch.cuda.memory.caching_allocator_disabled``
     - 是
     - ``torch.supa.memory.caching_allocator_disabled``
     -
   * - ``torch.cuda.memory.caching_allocator_enable``
     - 是
     - ``torch.supa.memory.caching_allocator_enable``
     -
   * - ``torch.cuda.nvtx.mark``
     - 是
     - ``torch.supa.brtx.mark``
     -
   * - ``torch.cuda.nvtx.range_push``
     - 是
     - ``torch.supa.brtx.range_push``
     -
   * - ``torch.cuda.nvtx.range_pop``
     - 是
     - ``torch.supa.brtx.range_pop``
     -
   * - ``torch.cuda.brtx.range``
     - 是
     - ``torch.supa.brtx.range``
     -
   * - ``torch.cuda.jiterator._create_jit_fn``
     - 是
     - ``torch.supa.jiterator._create_jit_fn``
     -
   * - ``torch.cuda.jiterator._create_multi_output_jit_fn``
     - 是
     - ``torch.supa.jiterator._create_multi_output_jit_fn``
     -
   * - ``torch.cuda.TunableOp``
     - 是
     - ``torch.supa.TunableOp``
     -
   * - ``torch.cuda.CUDA_Stream_Sanitizer``
     - 是
     - ``torch.supa.CUDA_Stream_Sanitizer``
     -
   * - ``torch.cuda.gds.gds_register_buffer``
     - 是
     - ``torch.supa.gds.gds_register_buffer``
     -
   * - ``torch.cuda.gds.gds_deregister_buffer``
     - 是
     - ``torch.supa.gds.gds_deregister_buffer``
     -
   * - ``torch.cuda.GdsFile``
     - 是
     - ``torch.supa.GdsFile``
     -
   * - ``torch.cuda.GreenContext``
     - 是
     - ``torch.supa.GreenContext``
     -
