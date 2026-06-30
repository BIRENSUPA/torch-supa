.. _native-api-torch_backend:

torch.backend
====================

.. list-table::
   :header-rows: 1
   :widths: 60 15 25

   * - PyTorch API
     - 是否兼容
     - 限制
   * - ``torch.backends.cpu.get_cpu_capability``
     - 
     - 
   * - ``torch.backends.cuda.is_built``
     - 是
     - 
   * - ``torch.backends.cuda.matmul.allow_tf32``
     - 是
     - 
   * - ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction``
     - 
     - 
   * - ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``
     - 
     - 
   * - ``torch.backends.cuda.cufft_plan_cache``
     - 
     - 
   * - ``torch.backends.cuda.cufft_plan_cache.size``
     - 
     - 
   * - ``torch.backends.cuda.cufft_plan_cache.max_size``
     - 
     - 
   * - ``torch.backends.cuda.cufft_plan_cache.clear``
     - 
     - 
   * - ``torch.backends.cuda.preferred_linalg_library``
     - 是
     - 
   * - ``torch.backends.cuda.SDPBackend``
     - 是
     - 
   * - ``torch.backends.cuda.flash_sdp_enabled``
     - 是
     - 
   * - ``torch.backends.cuda.enable_mem_efficient_sdp``
     - 是
     - 
   * - ``torch.backends.cuda.mem_efficient_sdp_enabled``
     - 是
     - 
   * - ``torch.backends.cuda.enable_flash_sdp``
     - 是
     - 
   * - ``torch.backends.cuda.math_sdp_enabled``
     - 是
     - 
   * - ``torch.backends.cuda.enable_math_sdp``
     - 是
     - 
   * - ``torch.backends.cuda.sdp_kernel``
     - 是
     - 
   * - ``torch.backends.cudnn.version``
     - 
     - 
   * - ``torch.backends.cudnn.is_available``
     - 是
     - 
   * - ``torch.backends.cudnn.enabled``
     - 是
     - 
   * - ``torch.backends.cudnn.allow_tf32``
     - 是
     - 
   * - ``torch.backends.cudnn.deterministic``
     - 是
     - 
   * - ``torch.backends.cudnn.benchmark``
     - 是
     - 
   * - ``torch.backends.cudnn.benchmark_limit``
     - 
     - 
   * - ``torch.backends.mps.is_available``
     - 是
     - 
   * - ``torch.backends.mps.is_built``
     - 是
     - 
   * - ``torch.backends.mkl.is_available``
     - 是
     - 
   * - ``torch.backends.mkl.verbose``
     - 是
     - 
   * - ``torch.backends.mkldnn.is_available``
     - 是
     - 
   * - ``torch.backends.mkldnn.verbose``
     - 是
     - 
   * - ``torch.backends.openmp.is_available``
     - 是
     - 
   * - ``torch.backends.opt_einsum.is_available``
     - 是
     - 
   * - ``torch.backends.opt_einsum.get_opt_einsum``
     - 是
     - 
   * - ``torch.backends.opt_einsum.enabled``
     - 
     - 
   * - ``torch.backends.opt_einsum.strategy``
     - 
     - 