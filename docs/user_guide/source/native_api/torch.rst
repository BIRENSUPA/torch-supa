.. _native-api-torch:

torch
=====

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.is_tensor``
     - 是
     -
   * - ``torch.is_storage``
     - 是
     -
   * - ``torch.is_complex``
     - 是
     -
   * - ``torch.is_conj``
     - 是
     -
   * - ``torch.is_floating_point``
     - 是
     -
   * - ``torch.is_nonzero``
     - 是
     -
   * - ``torch.set_default_dtype``
     - 是
     -
   * - ``torch.get_default_dtype``
     - 是
     -
   * - ``torch.set_default_device``
     - 是
     -
   * - ``torch.get_default_device``
     - 是
     -
   * - ``torch.set_default_tensor_type``
     - 是
     -
   * - ``torch.numel``
     - 是
     -
   * - ``torch.set_printoptions``
     - 是
     -
   * - ``torch.set_flush_denormal``
     - 是
     -
   * - ``torch.tensor``
     - 是
     -
   * - ``torch.sparse_coo_tensor``
     - 否
     -
   * - ``torch.sparse_csr_tensor``
     - 否
     -
   * - ``torch.sparse_csc_tensor``
     - 否
     -
   * - ``torch.sparse_bsr_tensor``
     - 否
     -
   * - ``torch.sparse_bsc_tensor``
     - 否
     -
   * - ``torch.asarray``
     - 是
     -
   * - ``torch.as_tensor``
     - 是
     -
   * - ``torch.as_strided``
     - 是
     -
   * - ``torch.from_file``
     - 是
     -
   * - ``torch.from_numpy``
     - 是
     -
   * - ``torch.from_dlpack``
     - 是
     -
   * - ``torch.frombuffer``
     - 是
     -
   * - ``torch.zeros``
     - 是
     -
   * - ``torch.zeros_like``
     - 是
     -
   * - ``torch.ones``
     - 是
     -
   * - ``torch.ones_like``
     - 是
     -
   * - ``torch.arange``
     - 是
     -
   * - ``torch.range``
     - 是
     -
   * - ``torch.linspace``
     - 是
     -
   * - ``torch.logspace``
     - 是
     -
   * - ``torch.eye``
     - 是
     -
   * - ``torch.empty``
     - 是
     -
   * - ``torch.empty_like``
     - 是
     -
   * - ``torch.empty_strided``
     - 是
     -
   * - ``torch.full``
     - 是
     -
   * - ``torch.full_like``
     - 是
     -
   * - ``torch.quantize_per_tensor``
     - 
     -
   * - ``torch.quantize_per_channel``
     - 
     -
   * - ``torch.dequantize``
     - 
     -
   * - ``torch.complex``
     - 是
     -
   * - ``torch.polar``
     - 是
     -
   * - ``torch.heaviside``
     - 是
     -
   * - ``torch.adjoint``
     - 是
     -
   * - ``torch.argwhere``
     - 是
     -
   * - ``torch.cat``
     - 是
     -
   * - ``torch.concat``
     - 是
     -
   * - ``torch.concatenate``
     - 是
     -
   * - ``torch.conj``
     - 是
     -
   * - ``torch.chunk``
     - 是
     -
   * - ``torch.dsplit``
     - 是
     -
   * - ``torch.column_stack``
     - 是
     -
   * - ``torch.dstack``
     - 是
     -
   * - ``torch.gather``
     - 是
     -
   * - ``torch.hsplit``
     - 是
     -
   * - ``torch.hstack``
     - 是
     -
   * - ``torch.index_add``
     - 是
     -
   * - ``torch.index_copy``
     - 是
     -
   * - ``torch.index_reduce``
     - 是
     -
   * - ``torch.index_select``
     - 是
     -
   * - ``torch.masked_select``
     - 是
     -
   * - ``torch.movedim``
     - 是
     -
   * - ``torch.moveaxis``
     - 是
     -
   * - ``torch.narrow``
     - 是
     -
   * - ``torch.narrow_copy``
     - 是
     -
   * - ``torch.nonzero``
     - 是
     -
   * - ``torch.permute``
     - 是
     -
   * - ``torch.reshape``
     - 是
     -
   * - ``torch.row_stack``
     - 是
     -
   * - ``torch.select``
     - 是
     -
   * - ``torch.scatter``
     - 是
     -
   * - ``torch.diagonal_scatter``
     - 是
     -
   * - ``torch.select_scatter``
     - 是
     -
   * - ``torch.slice_scatter``
     - 是
     -
   * - ``torch.scatter_add``
     - 是
     -
   * - ``torch.scatter_reduce``
     - 是
     -
   * - ``torch.segment_reduce``
     - 是
     -
   * - ``torch.split``
     - 是
     -
   * - ``torch.squeeze``
     - 是
     -
   * - ``torch.stack``
     - 是
     -
   * - ``torch.swapaxes``
     - 是
     -
   * - ``torch.swapdims``
     - 是
     -
   * - ``torch.t``
     - 是
     -
   * - ``torch.take``
     - 是
     -
   * - ``torch.take_along_dim``
     - 是
     -
   * - ``torch.tensor_split``
     - 是
     -
   * - ``torch.tile``
     - 是
     -
   * - ``torch.transpose``
     - 是
     -
   * - ``torch.unbind``
     - 是
     -
   * - ``torch.unravel_index``
     - 是
     -
   * - ``torch.unsqueeze``
     - 是
     -
   * - ``torch.vsplit``
     - 是
     -
   * - ``torch.vstack``
     - 是
     -
   * - ``torch.where``
     - 是
     -
   * - ``torch.Generator``
     - 是
     -
   * - ``torch.Generator.device``
     - 
     - 
   * - ``torch.Generator.get_state``
     - 是
     - 
   * - ``torch.Generator.initial_seed``
     - 是
     - 
   * - ``torch.Generator.manual_seed``
     - 是
     - 
   * - ``torch.Generator.seed``
     - 是
     - 
   * - ``torch.Generator.set_state``
     - 是
     - 
   * - ``torch.seed``
     - 是
     -
   * - ``torch.manual_seed``
     - 是
     -
   * - ``torch.initial_seed``
     - 是
     -
   * - ``torch.get_rng_state``
     - 是
     -
   * - ``torch.set_rng_state``
     - 是
     -
   * - ``torch.bernoulli``
     - 是
     -
   * - ``torch.multinomial``
     - 是
     -
   * - ``torch.normal``
     - 是
     -
   * - ``torch.poisson``
     - 是
     -
   * - ``torch.rand``
     - 是
     -
   * - ``torch.rand_like``
     - 是
     -
   * - ``torch.randint``
     - 是
     -
   * - ``torch.randint_like``
     - 是
     -
   * - ``torch.randn``
     - 是
     -
   * - ``torch.randn_like``
     - 是
     -
   * - ``torch.randperm``
     - 是
     -
   * - ``torch.quasirandom.SobolEngine``
     - 
     - 
   * - ``torch.quasirandom.SobolEngine.draw``
     - 
     - 
   * - ``torch.quasirandom.SobolEngine.draw_base2``
     - 
     - 
   * - ``torch.quasirandom.SobolEngine.fast_forward``
     - 
     - 
   * - ``torch.quasirandom.SobolEngine.reset``
     - 
     - 
   * - ``torch.save``
     - 是
     -
   * - ``torch.load``
     - 是
     -
   * - ``torch.get_num_threads``
     - 是
     -
   * - ``torch.set_num_threads``
     - 是
     -
   * - ``torch.get_num_interop_threads``
     - 是
     -
   * - ``torch.set_num_interop_threads``
     - 是
     -
   * - ``torch.no_grad``
     - 是
     - 
   * - ``torch.enable_grad``
     - 是
     - 
   * - ``torch.set_grad_enabled``
     - 是
     -
   * - ``torch.is_grad_enabled``
     - 是
     -
   * - ``torch.inference_mode``
     - 是
     -
   * - ``torch.is_inference_mode_enabled``
     - 是
     -
   * - ``torch.abs``
     - 是
     -
   * - ``torch.absolute``
     - 是
     -
   * - ``torch.acos``
     - 是
     -
   * - ``torch.arccos``
     - 是
     -
   * - ``torch.acosh``
     - 是
     -
   * - ``torch.arccosh``
     - 是
     -
   * - ``torch.add``
     - 是
     -
   * - ``torch.addcdiv``
     - 是
     -
   * - ``torch.addcmul``
     - 是
     -
   * - ``torch.angle``
     - 是
     -
   * - ``torch.asin``
     - 是
     -
   * - ``torch.arcsin``
     - 是
     -
   * - ``torch.asinh``
     - 是
     -
   * - ``torch.arcsinh``
     - 是
     -
   * - ``torch.atan``
     - 是
     -
   * - ``torch.arctan``
     - 是
     -
   * - ``torch.atanh``
     - 是
     -
   * - ``torch.arctanh``
     - 是
     -
   * - ``torch.atan2``
     - 是
     -
   * - ``torch.arctan2``
     - 是
     -
   * - ``torch.bitwise_not``
     - 是
     -
   * - ``torch.bitwise_and``
     - 是
     -
   * - ``torch.bitwise_or``
     - 是
     -
   * - ``torch.bitwise_xor``
     - 是
     -
   * - ``torch.bitwise_left_shift``
     - 是
     -
   * - ``torch.bitwise_right_shift``
     - 是
     -
   * - ``torch.ceil``
     - 是
     -
   * - ``torch.clamp``
     - 是
     -
   * - ``torch.clip``
     - 是
     -
   * - ``torch.conj_physical``
     - 是
     -
   * - ``torch.copysign``
     - 是
     -
   * - ``torch.cos``
     - 是
     -
   * - ``torch.cosh``
     - 是
     -
   * - ``torch.deg2rad``
     - 是
     -
   * - ``torch.div``
     - 是
     -
   * - ``torch.divide``
     - 是
     -
   * - ``torch.digamma``
     - 是
     -
   * - ``torch.erf``
     - 是
     -
   * - ``torch.erfc``
     - 是
     -
   * - ``torch.erfinv``
     - 是
     -
   * - ``torch.exp``
     - 是
     -
   * - ``torch.exp2``
     - 是
     -
   * - ``torch.expm1``
     - 是
     -
   * - ``torch.fake_quantize_per_channel_affine``
     - 是
     -
   * - ``torch.fake_quantize_per_tensor_affine``
     - 是
     -
   * - ``torch.fix``
     - 是
     -
   * - ``torch.float_power``
     - 是
     -
   * - ``torch.floor``
     - 是
     -
   * - ``torch.floor_divide``
     - 是
     -
   * - ``torch.fmod``
     - 是
     -
   * - ``torch.frac``
     - 是
     -
   * - ``torch.frexp``
     - 是
     -
   * - ``torch.gradient``
     - 是
     -
   * - ``torch.imag``
     - 是
     -
   * - ``torch.ldexp``
     - 是
     -
   * - ``torch.lerp``
     - 是
     -
   * - ``torch.lgamma``
     - 是
     -
   * - ``torch.log``
     - 是
     -
   * - ``torch.log10``
     - 是
     -
   * - ``torch.log1p``
     - 是
     -
   * - ``torch.log2``
     - 是
     -
   * - ``torch.logaddexp``
     - 是
     -
   * - ``torch.logaddexp2``
     - 是
     -
   * - ``torch.logical_and``
     - 是
     -
   * - ``torch.logical_not``
     - 是
     -
   * - ``torch.logical_or``
     - 是
     -
   * - ``torch.logical_xor``
     - 是
     -
   * - ``torch.logit``
     - 是
     -
   * - ``torch.hypot``
     - 是
     -
   * - ``torch.i0``
     - 是
     -
   * - ``torch.igamma``
     - 是
     -
   * - ``torch.igammac``
     - 是
     -
   * - ``torch.mul``
     - 是
     -
   * - ``torch.multiply``
     - 是
     -
   * - ``torch.mvlgamma``
     - 是
     -
   * - ``torch.nan_to_num``
     - 是
     -
   * - ``torch.neg``
     - 是
     -
   * - ``torch.negative``
     - 是
     -
   * - ``torch.nextafter``
     - 是
     -
   * - ``torch.polygamma``
     - 是
     -
   * - ``torch.positive``
     - 是
     -
   * - ``torch.pow``
     - 是
     -
   * - ``torch.quantized_batch_norm``
     - 是
     -
   * - ``torch.quantized_max_pool1d``
     - 是
     -
   * - ``torch.quantized_max_pool2d``
     - 是
     -
   * - ``torch.rad2deg``
     - 是
     -
   * - ``torch.real``
     - 是
     -
   * - ``torch.reciprocal``
     - 是
     -
   * - ``torch.remainder``
     - 是
     -
   * - ``torch.round``
     - 是
     -
   * - ``torch.rsqrt``
     - 是
     -
   * - ``torch.sigmoid``
     - 是
     -
   * - ``torch.sign``
     - 是
     -
   * - ``torch.sgn``
     - 是
     -
   * - ``torch.signbit``
     - 是
     -
   * - ``torch.sin``
     - 是
     -
   * - ``torch.sinc``
     - 是
     -
   * - ``torch.sinh``
     - 是
     -
   * - ``torch.softmax``
     - 是
     -
   * - ``torch.sqrt``
     - 是
     -
   * - ``torch.square``
     - 是
     -
   * - ``torch.sub``
     - 是
     -
   * - ``torch.subtract``
     - 是
     -
   * - ``torch.tan``
     - 是
     -
   * - ``torch.tanh``
     - 是
     -
   * - ``torch.true_divide``
     - 是
     -
   * - ``torch.trunc``
     - 是
     -
   * - ``torch.xlogy``
     - 是
     -
   * - ``torch.argmax``
     - 是
     -
   * - ``torch.argmin``
     - 是
     -
   * - ``torch.amax``
     - 是
     -
   * - ``torch.amin``
     - 是
     -
   * - ``torch.aminmax``
     - 是
     -
   * - ``torch.all``
     - 是
     -
   * - ``torch.any``
     - 是
     -
   * - ``torch.max``
     - 是
     -
   * - ``torch.min``
     - 是
     -
   * - ``torch.dist``
     - 是
     -
   * - ``torch.logsumexp``
     - 是
     -
   * - ``torch.mean``
     - 是
     -
   * - ``torch.nanmean``
     - 是
     -
   * - ``torch.median``
     - 是
     -
   * - ``torch.nanmedian``
     - 是
     -
   * - ``torch.mode``
     - 是
     -
   * - ``torch.norm``
     - 是
     -
   * - ``torch.nansum``
     - 是
     -
   * - ``torch.prod``
     - 是
     -
   * - ``torch.quantile``
     - 是
     -
   * - ``torch.nanquantile``
     - 是
     -
   * - ``torch.std``
     - 是
     -
   * - ``torch.std_mean``
     - 是
     -
   * - ``torch.sum``
     - 是
     -
   * - ``torch.unique``
     - 是
     -
   * - ``torch.unique_consecutive``
     - 是
     -
   * - ``torch.var``
     - 是
     -
   * - ``torch.var_mean``
     - 是
     -
   * - ``torch.count_nonzero``
     - 是
     -
   * - ``torch.hash_tensor``
     - 是
     -
   * - ``torch.allclose``
     - 是
     -
   * - ``torch.argsort``
     - 是
     -
   * - ``torch.eq``
     - 是
     -
   * - ``torch.equal``
     - 是
     -
   * - ``torch.ge``
     - 是
     -
   * - ``torch.greater_equal``
     - 是
     -
   * - ``torch.gt``
     - 是
     -
   * - ``torch.greater``
     - 是
     -
   * - ``torch.isclose``
     - 是
     -
   * - ``torch.isfinite``
     - 是
     -
   * - ``torch.isin``
     - 是
     -
   * - ``torch.isinf``
     - 是
     -
   * - ``torch.isposinf``
     - 是
     -
   * - ``torch.isneginf``
     - 是
     -
   * - ``torch.isnan``
     - 是
     -
   * - ``torch.isreal``
     - 是
     -
   * - ``torch.kthvalue``
     - 是
     -
   * - ``torch.le``
     - 是
     -
   * - ``torch.less_equal``
     - 是
     -
   * - ``torch.lt``
     - 是
     -
   * - ``torch.less``
     - 是
     -
   * - ``torch.maximum``
     - 是
     -
   * - ``torch.minimum``
     - 是
     -
   * - ``torch.fmax``
     - 是
     -
   * - ``torch.fmin``
     - 是
     -
   * - ``torch.ne``
     - 是
     -
   * - ``torch.not_equal``
     - 是
     -
   * - ``torch.sort``
     - 是
     -
   * - ``torch.topk``
     - 是
     -
   * - ``torch.msort``
     - 是
     -
   * - ``torch.stft``
     - 是
     -
   * - ``torch.istft``
     - 是
     -
   * - ``torch.bartlett_window``
     - 是
     -
   * - ``torch.blackman_window``
     - 是
     -
   * - ``torch.hamming_window``
     - 是
     -
   * - ``torch.hann_window``
     - 是
     -
   * - ``torch.kaiser_window``
     - 是
     -
   * - ``torch.atleast_1d``
     - 是
     -
   * - ``torch.atleast_2d``
     - 是
     -
   * - ``torch.atleast_3d``
     - 是
     -
   * - ``torch.bincount``
     - 是
     -
   * - ``torch.block_diag``
     - 是
     -
   * - ``torch.broadcast_tensors``
     - 是
     -
   * - ``torch.broadcast_to``
     - 是
     -
   * - ``torch.broadcast_shapes``
     - 是
     -
   * - ``torch.bucketize``
     - 是
     -
   * - ``torch.cartesian_prod``
     - 是
     -
   * - ``torch.cdist``
     - 是
     -
   * - ``torch.clone``
     - 是
     -
   * - ``torch.combinations``
     - 是
     -
   * - ``torch.corrcoef``
     - 是
     -
   * - ``torch.cov``
     - 是
     -
   * - ``torch.cross``
     - 是
     -
   * - ``torch.cummax``
     - 是
     -
   * - ``torch.cummin``
     - 是
     -
   * - ``torch.cumprod``
     - 是
     -
   * - ``torch.cumsum``
     - 是
     -
   * - ``torch.diag``
     - 是
     -
   * - ``torch.diag_embed``
     - 是
     -
   * - ``torch.diagflat``
     - 是
     -
   * - ``torch.diagonal``
     - 是
     -
   * - ``torch.diff``
     - 是
     -
   * - ``torch.einsum``
     - 是
     -
   * - ``torch.flatten``
     - 是
     -
   * - ``torch.flip``
     - 是
     -
   * - ``torch.fliplr``
     - 是
     -
   * - ``torch.flipud``
     - 是
     -
   * - ``torch.kron``
     - 是
     -
   * - ``torch.rot90``
     - 是
     -
   * - ``torch.gcd``
     - 是
     -
   * - ``torch.histc``
     - 是
     -
   * - ``torch.histogram``
     - 是
     -
   * - ``torch.histogramdd``
     - 是
     -
   * - ``torch.meshgrid``
     - 是
     -
   * - ``torch.lcm``
     - 是
     -
   * - ``torch.logcumsumexp``
     - 是
     -
   * - ``torch.ravel``
     - 是
     -
   * - ``torch.renorm``
     - 是
     -
   * - ``torch.repeat_interleave``
     - 是
     -
   * - ``torch.roll``
     - 是
     -
   * - ``torch.searchsorted``
     - 是
     -
   * - ``torch.tensordot``
     - 是
     -
   * - ``torch.trace``
     - 是
     -
   * - ``torch.tril``
     - 是
     -
   * - ``torch.tril_indices``
     - 是
     -
   * - ``torch.triu``
     - 是
     -
   * - ``torch.triu_indices``
     - 是
     -
   * - ``torch.unflatten``
     - 是
     -
   * - ``torch.vander``
     - 是
     -
   * - ``torch.view_as_real``
     - 是
     -
   * - ``torch.view_as_complex``
     - 是
     -
   * - ``torch.resolve_conj``
     - 是
     -
   * - ``torch.resolve_neg``
     - 是
     -
   * - ``torch.addbmm``
     - 是
     -
   * - ``torch.addmm``
     - 是
     -
   * - ``torch.addmv``
     - 是
     -
   * - ``torch.addr``
     - 是
     -
   * - ``torch.baddbmm``
     - 是
     -
   * - ``torch.bmm``
     - 是
     -
   * - ``torch.chain_matmul``
     - 是
     -
   * - ``torch.cholesky``
     - 是
     -
   * - ``torch.cholesky_inverse``
     - 是
     -
   * - ``torch.cholesky_solve``
     - 是
     -
   * - ``torch.dot``
     - 是
     -
   * - ``torch.geqrf``
     - 是
     -
   * - ``torch.ger``
     - 是
     -
   * - ``torch.inner``
     - 是
     -
   * - ``torch.inverse``
     - 是
     - 仅支持float32
   * - ``torch.det``
     - 是
     -
   * - ``torch.logdet``
     - 是
     -
   * - ``torch.slogdet``
     - 是
     -
   * - ``torch.lu``
     - 是
     -
   * - ``torch.lu_solve``
     - 是
     -
   * - ``torch.lu_unpack``
     - 是
     -
   * - ``torch.matmul``
     - 是
     -
   * - ``torch.matrix_power``
     - 是
     -
   * - ``torch.matrix_exp``
     - 是
     -
   * - ``torch.mm``
     - 是
     -
   * - ``torch.mv``
     - 是
     -
   * - ``torch.orgqr``
     - 是
     -
   * - ``torch.ormqr``
     - 是
     -
   * - ``torch.outer``
     - 是
     -
   * - ``torch.pinverse``
     - 是
     -
   * - ``torch.qr``
     - 是
     -
   * - ``torch.svd``
     - 是
     -
   * - ``torch.svd_lowrank``
     - 是
     -
   * - ``torch.pca_lowrank``
     - 是
     -
   * - ``torch.lobpcg``
     - 是
     -
   * - ``torch.trapz``
     - 是
     -
   * - ``torch.trapezoid``
     - 是
     -
   * - ``torch.cumulative_trapezoid``
     - 是
     -
   * - ``torch.triangular_solve``
     - 是
     -
   * - ``torch.vdot``
     - 是
     -
   * - ``torch._foreach_abs``
     - 是
     -
   * - ``torch._foreach_abs_``
     - 是
     -
   * - ``torch._foreach_acos``
     - 是
     -
   * - ``torch._foreach_acos_``
     - 是
     -
   * - ``torch._foreach_asin``
     - 是
     -
   * - ``torch._foreach_asin_``
     - 是
     -
   * - ``torch._foreach_atan``
     - 是
     -
   * - ``torch._foreach_atan_``
     - 是
     -
   * - ``torch._foreach_ceil``
     - 是
     -
   * - ``torch._foreach_ceil_``
     - 是
     -
   * - ``torch._foreach_clone``
     - 是
     -
   * - ``torch._foreach_cos``
     - 是
     -
   * - ``torch._foreach_cos_``
     - 是
     -
   * - ``torch._foreach_cosh``
     - 是
     -
   * - ``torch._foreach_cosh_``
     - 是
     -
   * - ``torch._foreach_erf``
     - 是
     -
   * - ``torch._foreach_erf_``
     - 是
     -
   * - ``torch._foreach_erfc``
     - 是
     -
   * - ``torch._foreach_erfc_``
     - 是
     -
   * - ``torch._foreach_exp``
     - 是
     -
   * - ``torch._foreach_exp_``
     - 是
     -
   * - ``torch._foreach_expm1``
     - 是
     -
   * - ``torch._foreach_expm1_``
     - 是
     -
   * - ``torch._foreach_floor``
     - 是
     -
   * - ``torch._foreach_floor_``
     - 是
     -
   * - ``torch._foreach_log``
     - 是
     -
   * - ``torch._foreach_log_``
     - 是
     -
   * - ``torch._foreach_log10``
     - 是
     -
   * - ``torch._foreach_log10_``
     - 是
     -
   * - ``torch._foreach_log1p``
     - 是
     -
   * - ``torch._foreach_log1p_``
     - 是
     -
   * - ``torch._foreach_log2``
     - 是
     -
   * - ``torch._foreach_log2_``
     - 是
     -
   * - ``torch._foreach_neg``
     - 是
     -
   * - ``torch._foreach_neg_``
     - 是
     -
   * - ``torch._foreach_tan``
     - 是
     -
   * - ``torch._foreach_tan_``
     - 是
     -
   * - ``torch._foreach_sin``
     - 是
     -
   * - ``torch._foreach_sin_``
     - 是
     -
   * - ``torch._foreach_sinh``
     - 是
     -
   * - ``torch._foreach_sinh_``
     - 是
     -
   * - ``torch._foreach_round``
     - 是
     -
   * - ``torch._foreach_round_``
     - 是
     -
   * - ``torch._foreach_sqrt``
     - 是
     -
   * - ``torch._foreach_sqrt_``
     - 是
     -
   * - ``torch._foreach_lgamma``
     - 是
     -
   * - ``torch._foreach_lgamma_``
     - 是
     -
   * - ``torch._foreach_frac``
     - 是
     -
   * - ``torch._foreach_frac_``
     - 是
     -
   * - ``torch._foreach_reciprocal``
     - 是
     -
   * - ``torch._foreach_reciprocal_``
     - 是
     -
   * - ``torch._foreach_sigmoid``
     - 是
     -
   * - ``torch._foreach_sigmoid_``
     - 是
     -
   * - ``torch._foreach_trunc``
     - 是
     -
   * - ``torch._foreach_trunc_``
     - 是
     -
   * - ``torch._foreach_zero_``
     - 是
     -
   * - ``torch.compiled_with_cxx11_abi``
     - 是
     -
   * - ``torch.result_type``
     - 是
     -
   * - ``torch.can_cast``
     - 是
     -
   * - ``torch.promote_types``
     - 是
     -
   * - ``torch.use_deterministic_algorithms``
     - 是
     -
   * - ``torch.are_deterministic_algorithms_enabled``
     - 是
     -
   * - ``torch.is_deterministic_algorithms_warn_only_enabled``
     - 是
     -
   * - ``torch.set_deterministic_debug_mode``
     - 是
     -
   * - ``torch.get_deterministic_debug_mode``
     - 是
     -
   * - ``torch.set_float32_matmul_precision``
     - 是
     -
   * - ``torch.get_float32_matmul_precision``
     - 是
     -
   * - ``torch.set_warn_always``
     - 是
     -
   * - ``torch.get_device_module``
     - 是
     -
   * - ``torch.is_warn_always_enabled``
     - 是
     -
   * - ``torch.vmap``
     - 是
     -
   * - ``torch._assert``
     - 是
     -
   * - ``torch.typename``
     - 是
     -
   * - ``torch.sym_float``
     - 是
     -
   * - ``torch.sym_fresh_size``
     - 是
     -
   * - ``torch.sym_int``
     - 是
     -
   * - ``torch.sym_max``
     - 是
     -
   * - ``torch.sym_min``
     - 是
     -
   * - ``torch.sym_not``
     - 是
     -
   * - ``torch.sym_ite``
     - 是
     -
   * - ``torch.sym_sqrt``
     - 是
     -
   * - ``torch.sym_sum``
     - 是
     -
   * - ``torch.cond``
     - 是
     -
   * - ``torch.compile``
     - 是
     -
