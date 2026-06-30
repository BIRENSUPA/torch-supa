.. _native-api-torch_utils:

torch.utils
=================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.rename_privateuse1_backend``
     - 
     - 
   * - ``torch.utils.generate_methods_for_privateuse1_backend``
     - 
     - 
   * - ``torch.utils.get_cpp_backtrace``
     - 
     - 
   * - ``torch.utils.set_module``
     - 
     - 

.. _native-api-torch_utils_benchmark:

torch.utils.benchmark
=========================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.benchmark.Timer``
     - 
     - 
   * - ``torch.utils.benchmark.Timer.blocked_autorange``
     - 
     - 
   * - ``torch.utils.benchmark.Timer.collect_callgrind``
     - 是
     - 
   * - ``torch.utils.benchmark.Timer.timeit``
     - 
     - 
   * - ``torch.utils.benchmark.Measurement``
     - 是
     - 
   * - ``torch.utils.benchmark.Measurement.merge``
     - 
     - 
   * - ``torch.utils.benchmark.Measurement.significant_figures``
     - 
     - 
   * - ``torch.utils.benchmark.CallgrindStats``
     - 是
     - 
   * - ``torch.utils.benchmark.CallgrindStats.as_standardized``
     - 是
     - 
   * - ``torch.utils.benchmark.CallgrindStats.counts``
     - 是
     - 
   * - ``torch.utils.benchmark.CallgrindStats.delta``
     - 是
     - 
   * - ``torch.utils.benchmark.CallgrindStats.stats``
     - 是
     - 
   * - ``torch.utils.benchmark.FunctionCounts``
     - 是
     - 
   * - ``torch.utils.benchmark.FunctionCounts.denoise``
     - 是
     - 
   * - ``torch.utils.benchmark.FunctionCounts.filter``
     - 是
     - 
   * - ``torch.utils.benchmark.FunctionCounts.transform``
     - 是
     - 
.. _native-api-torch_utils_checkpoint:

torch.utils.checkpoint
========================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.checkpoint.checkpoint``
     - 是
     - 
   * - ``torch.utils.checkpoint.checkpoint_sequential``
     - 是
     - 


.. _native-api-torch_utils_cpp_extension:

torch.utils.cpp_extension
===============================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.cpp_extension.CppExtension``
     - 是
     - 
   * - ``torch.utils.cpp_extension.CUDAExtension``
     - 是
     - 
   * - ``torch.utils.cpp_extension.BuildExtension``
     - 是
     - 
   * - ``torch.utils.cpp_extension.load``
     - 是
     - 
   * - ``torch.utils.cpp_extension.load_inline``
     - 是
     - 
   * - ``torch.utils.cpp_extension.include_paths``
     - 是
     - 
   * - ``torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version``
     - 是
     - 
   * - ``torch.utils.cpp_extension.verify_ninja_availability``
     - 是
     - 
   * - ``torch.utils.cpp_extension.is_ninja_available``
     - 是
     - 
.. _native-api-torch_utils_data:

torch.utils.data
==================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.data.DataLoader``
     - 是
     - 
   * - ``torch.utils.data.Dataset``
     - 是
     - 
   * - ``torch.utils.data.IterableDataset``
     - 是
     - 
   * - ``torch.utils.data.TensorDataset``
     - 是
     - 
   * - ``torch.utils.data.StackDataset``
     - 
     - 
   * - ``torch.utils.data.ConcatDataset``
     - 是
     - 
   * - ``torch.utils.data.ChainDataset``
     - 是
     - 
   * - ``torch.utils.data.Subset``
     - 是
     - 
   * - ``torch.utils.data._utils.collate.collate``
     - 是
     - 
   * - ``torch.utils.data.default_collate``
     - 是
     - 
   * - ``torch.utils.data.default_convert``
     - 是
     - 
   * - ``torch.utils.data.get_worker_info``
     - 是
     - 
   * - ``torch.utils.data.random_split``
     - 是
     - 
   * - ``torch.utils.data.Sampler``
     - 是
     - 
   * - ``torch.utils.data.SequentialSampler``
     - 是
     - 
   * - ``torch.utils.data.RandomSampler``
     - 是
     - 
   * - ``torch.utils.data.SubsetRandomSampler``
     - 是
     - 
   * - ``torch.utils.data.WeightedRandomSampler``
     - 是
     - 
   * - ``torch.utils.data.BatchSampler``
     - 是
     - 
   * - ``torch.utils.data.distributed.DistributedSampler``
     - 是
     - 


.. _native-api-torch_utils_dlpack:

torch.utils.dlpack
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.dlpack.from_dlpack``
     - 
     - 
   * - ``torch.utils.dlpack.to_dlpack``
     - 
     - 


.. _native-api-torch_utils_mobile_optimizer:

torch.utils.mobile_optimizer
=============================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.mobile_optimizer.optimize_for_mobile``
     - 
     - 


.. _native-api-torch_utils_model_zoo:

torch.utils.model_zoo
===========================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.model_zoo.load_url``
     - 是
     - 


.. _native-api-torch_utils_tensorboard:

torch.utils.tensorboard
===========================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.utils.tensorboard.writer.SummaryWriter``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.__init__``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_scalar``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_scalars``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_histogram``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_image``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_images``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_figure``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_video``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_audio``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_text``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_graph``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_embedding``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_mesh``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.add_hparams``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.flush``
     - 是
     - 
   * - ``torch.utils.tensorboard.writer.SummaryWriter.close``
     - 是
     - 
