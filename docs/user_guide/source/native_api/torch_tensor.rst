.. _native-api-torch_tensor:

torch.Tensor
============

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.Tensor.new_tensor``
     - 是
     -
   * - ``torch.Tensor.new_full``
     - 是
     -
   * - ``torch.Tensor.new_empty``
     - 是
     -
   * - ``torch.Tensor.new_ones``
     - 是
     -
   * - ``torch.Tensor.new_zeros``
     - 是
     -
   * - ``torch.Tensor.is_cuda``
     - 是
     -
   * - ``torch.Tensor.is_quantized``
     - 是
     -
   * - ``torch.Tensor.is_meta``
     - 是
     -
   * - ``torch.Tensor.device``
     - 是
     -
   * - ``torch.Tensor.grad``
     - 是
     -
   * - ``torch.Tensor.ndim``
     - 是
     -
   * - ``torch.Tensor.real``
     - 是
     -
   * - ``torch.Tensor.imag``
     - 是
     -
   * - ``torch.Tensor.nbytes``
     - 是
     -
   * - ``torch.Tensor.itemsize``
     - 是
     -
   * - ``torch.Tensor.abs``
     - 是
     -
   * - ``torch.Tensor.abs_``
     - 是
     -
   * - ``torch.Tensor.absolute``
     - 是
     -
   * - ``torch.Tensor.absolute_``
     - 是
     -
   * - ``torch.Tensor.acos``
     - 是
     -
   * - ``torch.Tensor.acos_``
     - 是
     -
   * - ``torch.Tensor.arccos``
     - 是
     -
   * - ``torch.Tensor.arccos_``
     - 是
     -
   * - ``torch.Tensor.add``
     - 是
     -
   * - ``torch.Tensor.add_``
     - 是
     -
   * - ``torch.Tensor.addbmm``
     - 是
     -
   * - ``torch.Tensor.addbmm_``
     - 是
     -
   * - ``torch.Tensor.addcdiv``
     - 是
     -
   * - ``torch.Tensor.addcdiv_``
     - 是
     -
   * - ``torch.Tensor.addcmul``
     - 是
     -
   * - ``torch.Tensor.addcmul_``
     - 是
     -
   * - ``torch.Tensor.addmm``
     - 是
     -
   * - ``torch.Tensor.addmm_``
     - 是
     -
   * - ``torch.Tensor.sspaddmm``
     - 是
     -
   * - ``torch.Tensor.addmv``
     - 是
     -
   * - ``torch.Tensor.addmv_``
     - 是
     -
   * - ``torch.Tensor.addr``
     - 是
     -
   * - ``torch.Tensor.addr_``
     - 是
     -
   * - ``torch.Tensor.adjoint``
     - 是
     -
   * - ``torch.Tensor.allclose``
     - 是
     -
   * - ``torch.Tensor.amax``
     - 是
     -
   * - ``torch.Tensor.amin``
     - 是
     -
   * - ``torch.Tensor.aminmax``
     - 是
     -
   * - ``torch.Tensor.angle``
     - 是
     -
   * - ``torch.Tensor.apply_``
     - 是
     -
   * - ``torch.Tensor.argmax``
     - 是
     -
   * - ``torch.Tensor.argmin``
     - 是
     -
   * - ``torch.Tensor.argsort``
     - 是
     -
   * - ``torch.Tensor.argwhere``
     - 是
     -
   * - ``torch.Tensor.asin``
     - 是
     -
   * - ``torch.Tensor.asin_``
     - 是
     -
   * - ``torch.Tensor.arcsin``
     - 是
     -
   * - ``torch.Tensor.arcsin_``
     - 是
     -
   * - ``torch.Tensor.as_strided``
     - 是
     -
   * - ``torch.Tensor.atan``
     - 是
     -
   * - ``torch.Tensor.atan_``
     - 是
     -
   * - ``torch.Tensor.arctan``
     - 是
     -
   * - ``torch.Tensor.arctan_``
     - 是
     -
   * - ``torch.Tensor.atan2``
     - 是
     -
   * - ``torch.Tensor.atan2_``
     - 是
     -
   * - ``torch.Tensor.arctan2``
     - 是
     -
   * - ``torch.Tensor.arctan2_``
     - 是
     -
   * - ``torch.Tensor.all``
     - 是
     -
   * - ``torch.Tensor.any``
     - 是
     -
   * - ``torch.Tensor.backward``
     - 是
     -
   * - ``torch.Tensor.baddbmm``
     - 是
     -
   * - ``torch.Tensor.baddbmm_``
     - 是
     -
   * - ``torch.Tensor.bernoulli``
     - 是
     -
   * - ``torch.Tensor.bernoulli_``
     - 是
     -
   * - ``torch.Tensor.bfloat16``
     - 是
     -
   * - ``torch.Tensor.bincount``
     - 是
     -
   * - ``torch.Tensor.bitwise_not``
     - 是
     -
   * - ``torch.Tensor.bitwise_not_``
     - 是
     -
   * - ``torch.Tensor.bitwise_and``
     - 是
     -
   * - ``torch.Tensor.bitwise_and_``
     - 是
     -
   * - ``torch.Tensor.bitwise_or``
     - 是
     -
   * - ``torch.Tensor.bitwise_or_``
     - 是
     -
   * - ``torch.Tensor.bitwise_xor``
     - 是
     -
   * - ``torch.Tensor.bitwise_xor_``
     - 是
     -
   * - ``torch.Tensor.bitwise_left_shift``
     - 是
     -
   * - ``torch.Tensor.bitwise_left_shift_``
     - 是
     -
   * - ``torch.Tensor.bitwise_right_shift``
     - 是
     -
   * - ``torch.Tensor.bitwise_right_shift_``
     - 是
     -
   * - ``torch.Tensor.bmm``
     - 是
     -
   * - ``torch.Tensor.bool``
     - 是
     -
   * - ``torch.Tensor.byte``
     - 是
     -
   * - ``torch.Tensor.broadcast_to``
     - 是
     -
   * - ``torch.Tensor.cauchy_``
     - 是
     -
   * - ``torch.Tensor.ceil``
     - 是
     -
   * - ``torch.Tensor.ceil_``
     - 是
     -
   * - ``torch.Tensor.char``
     - 是
     -
   * - ``torch.Tensor.cholesky``
     - 是
     -
   * - ``torch.Tensor.cholesky_inverse``
     - 是
     -
   * - ``torch.Tensor.cholesky_solve``
     - 是
     -
   * - ``torch.Tensor.chunk``
     - 是
     -
   * - ``torch.Tensor.clamp``
     - 是
     -
   * - ``torch.Tensor.clamp_``
     - 是
     -
   * - ``torch.Tensor.clip``
     - 是
     -
   * - ``torch.Tensor.clip_``
     - 是
     -
   * - ``torch.Tensor.clone``
     - 是
     -
   * - ``torch.Tensor.contiguous``
     - 是
     -
   * - ``torch.Tensor.copy_``
     - 是
     -
   * - ``torch.Tensor.conj``
     - 是
     -
   * - ``torch.Tensor.conj_physical``
     - 是
     -
   * - ``torch.Tensor.conj_physical_``
     - 是
     -
   * - ``torch.Tensor.resolve_conj``
     - 是
     -
   * - ``torch.Tensor.resolve_neg``
     - 是
     -
   * - ``torch.Tensor.copysign``
     - 是
     -
   * - ``torch.Tensor.copysign_``
     - 是
     -
   * - ``torch.Tensor.cos``
     - 是
     -
   * - ``torch.Tensor.cos_``
     - 是
     -
   * - ``torch.Tensor.cosh``
     - 是
     -
   * - ``torch.Tensor.cosh_``
     - 是
     -
   * - ``torch.Tensor.corrcoef``
     - 是
     -
   * - ``torch.Tensor.count_nonzero``
     - 是
     -
   * - ``torch.Tensor.cov``
     - 是
     -
   * - ``torch.Tensor.acosh``
     - 是
     -
   * - ``torch.Tensor.acosh_``
     - 是
     -
   * - ``torch.Tensor.arccosh``
     - 是
     -
   * - ``torch.Tensor.arccosh_``
     - 是
     -
   * - ``torch.Tensor.cpu``
     - 是
     -
   * - ``torch.Tensor.cross``
     - 是
     -
   * - ``torch.Tensor.cuda``
     - 是
     -
   * - ``torch.Tensor.logcumsumexp``
     - 是
     -
   * - ``torch.Tensor.cummax``
     - 是
     -
   * - ``torch.Tensor.cummin``
     - 是
     -
   * - ``torch.Tensor.cumprod``
     - 是
     -
   * - ``torch.Tensor.cumprod_``
     - 是
     -
   * - ``torch.Tensor.cumsum``
     - 是
     -
   * - ``torch.Tensor.cumsum_``
     - 是
     -
   * - ``torch.Tensor.chalf``
     - 是
     -
   * - ``torch.Tensor.cfloat``
     - 是
     -
   * - ``torch.Tensor.cdouble``
     - 是
     -
   * - ``torch.Tensor.data_ptr``
     - 是
     -
   * - ``torch.Tensor.deg2rad``
     - 是
     -
   * - ``torch.Tensor.dequantize``
     - 是
     -
   * - ``torch.Tensor.det``
     - 是
     -
   * - ``torch.Tensor.dense_dim``
     - 是
     -
   * - ``torch.Tensor.detach``
     - 是
     -
   * - ``torch.Tensor.detach_``
     - 是
     -
   * - ``torch.Tensor.diag``
     - 是
     -
   * - ``torch.Tensor.diag_embed``
     - 是
     -
   * - ``torch.Tensor.diagflat``
     - 是
     -
   * - ``torch.Tensor.diagonal``
     - 是
     -
   * - ``torch.Tensor.diagonal_scatter``
     - 是
     -
   * - ``torch.Tensor.fill_diagonal_``
     - 是
     -
   * - ``torch.Tensor.fmax``
     - 是
     -
   * - ``torch.Tensor.fmin``
     - 是
     -
   * - ``torch.Tensor.diff``
     - 是
     -
   * - ``torch.Tensor.digamma``
     - 是
     -
   * - ``torch.Tensor.digamma_``
     - 是
     -
   * - ``torch.Tensor.dim``
     - 是
     -
   * - ``torch.Tensor.dim_order``
     - 是
     -
   * - ``torch.Tensor.dist``
     - 是
     -
   * - ``torch.Tensor.div``
     - 是
     -
   * - ``torch.Tensor.div_``
     - 是
     -
   * - ``torch.Tensor.divide``
     - 是
     -
   * - ``torch.Tensor.divide_``
     - 是
     -
   * - ``torch.Tensor.dot``
     - 是
     -
   * - ``torch.Tensor.double``
     - 是
     -
   * - ``torch.Tensor.dsplit``
     - 是
     -
   * - ``torch.Tensor.element_size``
     - 是
     -
   * - ``torch.Tensor.eq``
     - 是
     -
   * - ``torch.Tensor.eq_``
     - 是
     -
   * - ``torch.Tensor.equal``
     - 是
     -
   * - ``torch.Tensor.erf``
     - 是
     -
   * - ``torch.Tensor.erf_``
     - 是
     -
   * - ``torch.Tensor.erfc``
     - 是
     -
   * - ``torch.Tensor.erfc_``
     - 是
     -
   * - ``torch.Tensor.erfinv``
     - 是
     -
   * - ``torch.Tensor.erfinv_``
     - 是
     -
   * - ``torch.Tensor.exp``
     - 是
     -
   * - ``torch.Tensor.exp_``
     - 是
     -
   * - ``torch.Tensor.expm1``
     - 是
     -
   * - ``torch.Tensor.expm1_``
     - 是
     -
   * - ``torch.Tensor.expand``
     - 是
     -
   * - ``torch.Tensor.expand_as``
     - 是
     -
   * - ``torch.Tensor.exponential_``
     - 是
     -
   * - ``torch.Tensor.fix``
     - 是
     -
   * - ``torch.Tensor.fix_``
     - 是
     -
   * - ``torch.Tensor.fill_``
     - 是
     -
   * - ``torch.Tensor.flatten``
     - 是
     -
   * - ``torch.Tensor.flip``
     - 是
     -
   * - ``torch.Tensor.fliplr``
     - 是
     -
   * - ``torch.Tensor.flipud``
     - 是
     -
   * - ``torch.Tensor.float``
     - 是
     -
   * - ``torch.Tensor.float_power``
     - 是
     -
   * - ``torch.Tensor.float_power_``
     - 是
     -
   * - ``torch.Tensor.floor``
     - 是
     -
   * - ``torch.Tensor.floor_``
     - 是
     -
   * - ``torch.Tensor.floor_divide``
     - 是
     -
   * - ``torch.Tensor.floor_divide_``
     - 是
     -
   * - ``torch.Tensor.fmod``
     - 是
     -
   * - ``torch.Tensor.fmod_``
     - 是
     -
   * - ``torch.Tensor.frac``
     - 是
     -
   * - ``torch.Tensor.frac_``
     - 是
     -
   * - ``torch.Tensor.frexp``
     - 是
     -
   * - ``torch.Tensor.gather``
     - 是
     -
   * - ``torch.Tensor.gcd``
     - 是
     -
   * - ``torch.Tensor.gcd_``
     - 是
     -
   * - ``torch.Tensor.ge``
     - 是
     -
   * - ``torch.Tensor.ge_``
     - 是
     -
   * - ``torch.Tensor.greater_equal``
     - 是
     -
   * - ``torch.Tensor.greater_equal_``
     - 是
     -
   * - ``torch.Tensor.geometric_``
     - 是
     -
   * - ``torch.Tensor.geqrf``
     - 是
     -
   * - ``torch.Tensor.ger``
     - 是
     -
   * - ``torch.Tensor.get_device``
     - 是
     -
   * - ``torch.Tensor.gt``
     - 是
     -
   * - ``torch.Tensor.gt_``
     - 是
     -
   * - ``torch.Tensor.greater``
     - 是
     -
   * - ``torch.Tensor.greater_``
     - 是
     -
   * - ``torch.Tensor.half``
     - 是
     -
   * - ``torch.Tensor.hardshrink``
     - 是
     -
   * - ``torch.Tensor.heaviside``
     - 是
     -
   * - ``torch.Tensor.histc``
     - 是
     -
   * - ``torch.Tensor.histogram``
     - 是
     -
   * - ``torch.Tensor.hsplit``
     - 是
     -
   * - ``torch.Tensor.hypot``
     - 是
     -
   * - ``torch.Tensor.hypot_``
     - 是
     -
   * - ``torch.Tensor.i0``
     - 是
     -
   * - ``torch.Tensor.i0_``
     - 是
     -
   * - ``torch.Tensor.igamma``
     - 是
     -
   * - ``torch.Tensor.igamma_``
     - 是
     -
   * - ``torch.Tensor.igammac``
     - 是
     -
   * - ``torch.Tensor.igammac_``
     - 是
     -
   * - ``torch.Tensor.index_add_``
     - 是
     -
   * - ``torch.Tensor.index_add``
     - 是
     -
   * - ``torch.Tensor.index_copy_``
     - 是
     -
   * - ``torch.Tensor.index_copy``
     - 是
     -
   * - ``torch.Tensor.index_fill_``
     - 是
     -
   * - ``torch.Tensor.index_fill``
     - 是
     -
   * - ``torch.Tensor.index_put_``
     - 是
     -
   * - ``torch.Tensor.index_put``
     - 是
     -
   * - ``torch.Tensor.index_reduce_``
     - 是
     -
   * - ``torch.Tensor.index_reduce``
     - 是
     -
   * - ``torch.Tensor.index_select``
     - 是
     -
   * - ``torch.Tensor.indices``
     - 是
     -
   * - ``torch.Tensor.inner``
     - 是
     -
   * - ``torch.Tensor.int``
     - 是
     -
   * - ``torch.Tensor.int_repr``
     - 是
     -
   * - ``torch.Tensor.inverse``
     - 是
     -
   * - ``torch.Tensor.isclose``
     - 是
     -
   * - ``torch.Tensor.isfinite``
     - 是
     -
   * - ``torch.Tensor.isinf``
     - 是
     -
   * - ``torch.Tensor.isposinf``
     - 是
     -
   * - ``torch.Tensor.isneginf``
     - 是
     -
   * - ``torch.Tensor.isnan``
     - 是
     -
   * - ``torch.Tensor.is_contiguous``
     - 是
     -
   * - ``torch.Tensor.is_complex``
     - 是
     -
   * - ``torch.Tensor.is_conj``
     - 是
     -
   * - ``torch.Tensor.is_floating_point``
     - 是
     -
   * - ``torch.Tensor.is_inference``
     - 是
     -
   * - ``torch.Tensor.is_leaf``
     - 是
     -
   * - ``torch.Tensor.is_pinned``
     - 是
     -
   * - ``torch.Tensor.is_set_to``
     - 是
     -
   * - ``torch.Tensor.is_shared``
     - 是
     -
   * - ``torch.Tensor.is_signed``
     - 是
     -
   * - ``torch.Tensor.is_sparse``
     - 是
     -
   * - ``torch.Tensor.istft``
     - 是
     -
   * - ``torch.Tensor.isreal``
     - 是
     -
   * - ``torch.Tensor.item``
     - 是
     -
   * - ``torch.Tensor.kthvalue``
     - 是
     -
   * - ``torch.Tensor.lcm``
     - 是
     -
   * - ``torch.Tensor.lcm_``
     - 是
     -
   * - ``torch.Tensor.ldexp``
     - 是
     -
   * - ``torch.Tensor.ldexp_``
     - 是
     -
   * - ``torch.Tensor.le``
     - 是
     -
   * - ``torch.Tensor.le_``
     - 是
     -
   * - ``torch.Tensor.less_equal``
     - 是
     -
   * - ``torch.Tensor.less_equal_``
     - 是
     -
   * - ``torch.Tensor.lerp``
     - 是
     -
   * - ``torch.Tensor.lerp_``
     - 是
     -
   * - ``torch.Tensor.lgamma``
     - 是
     -
   * - ``torch.Tensor.lgamma_``
     - 是
     -
   * - ``torch.Tensor.log``
     - 是
     -
   * - ``torch.Tensor.log_``
     - 是
     -
   * - ``torch.Tensor.logdet``
     - 是
     -
   * - ``torch.Tensor.log10``
     - 是
     -
   * - ``torch.Tensor.log10_``
     - 是
     -
   * - ``torch.Tensor.log1p``
     - 是
     -
   * - ``torch.Tensor.log1p_``
     - 是
     -
   * - ``torch.Tensor.log2``
     - 是
     -
   * - ``torch.Tensor.log2_``
     - 是
     -
   * - ``torch.Tensor.log_normal_``
     - 是
     -
   * - ``torch.Tensor.logaddexp``
     - 是
     -
   * - ``torch.Tensor.logaddexp2``
     - 是
     -
   * - ``torch.Tensor.logsumexp``
     - 是
     -
   * - ``torch.Tensor.logical_and``
     - 是
     -
   * - ``torch.Tensor.logical_and_``
     - 是
     -
   * - ``torch.Tensor.logical_not``
     - 是
     -
   * - ``torch.Tensor.logical_not_``
     - 是
     -
   * - ``torch.Tensor.logical_or``
     - 是
     -
   * - ``torch.Tensor.logical_or_``
     - 是
     -
   * - ``torch.Tensor.logical_xor``
     - 是
     -
   * - ``torch.Tensor.logical_xor_``
     - 是
     -
   * - ``torch.Tensor.logit``
     - 是
     -
   * - ``torch.Tensor.logit_``
     - 是
     -
   * - ``torch.Tensor.long``
     - 是
     -
   * - ``torch.Tensor.lt``
     - 是
     -
   * - ``torch.Tensor.lt_``
     - 是
     -
   * - ``torch.Tensor.less``
     - 是
     -
   * - ``torch.Tensor.less_``
     - 是
     -
   * - ``torch.Tensor.lu``
     - 是
     -
   * - ``torch.Tensor.lu_solve``
     - 是
     -
   * - ``torch.Tensor.as_subclass``
     - 是
     -
   * - ``torch.Tensor.map_``
     - 是
     -
   * - ``torch.Tensor.masked_scatter_``
     - 是
     -
   * - ``torch.Tensor.masked_scatter``
     - 是
     -
   * - ``torch.Tensor.masked_fill_``
     - 是
     -
   * - ``torch.Tensor.masked_fill``
     - 是
     -
   * - ``torch.Tensor.masked_select``
     - 是
     -
   * - ``torch.Tensor.matmul``
     - 是
     -
   * - ``torch.Tensor.matrix_power``
     - 是
     -
   * - ``torch.Tensor.matrix_exp``
     - 是
     -
   * - ``torch.Tensor.max``
     - 是
     -
   * - ``torch.Tensor.maximum``
     - 是
     -
   * - ``torch.Tensor.mean``
     - 是
     -
   * - ``torch.Tensor.module_load``
     - 是
     -
   * - ``torch.Tensor.nanmean``
     - 是
     -
   * - ``torch.Tensor.median``
     - 是
     -
   * - ``torch.Tensor.nanmedian``
     - 是
     -
   * - ``torch.Tensor.min``
     - 是
     -
   * - ``torch.Tensor.minimum``
     - 是
     -
   * - ``torch.Tensor.mm``
     - 是
     -
   * - ``torch.Tensor.smm``
     - 是
     -
   * - ``torch.Tensor.mode``
     - 是
     -
   * - ``torch.Tensor.movedim``
     - 是
     -
   * - ``torch.Tensor.moveaxis``
     - 是
     -
   * - ``torch.Tensor.msort``
     - 是
     -
   * - ``torch.Tensor.mul``
     - 是
     -
   * - ``torch.Tensor.mul_``
     - 是
     -
   * - ``torch.Tensor.multiply``
     - 是
     -
   * - ``torch.Tensor.multiply_``
     - 是
     -
   * - ``torch.Tensor.multinomial``
     - 是
     -
   * - ``torch.Tensor.mv``
     - 是
     -
   * - ``torch.Tensor.mvlgamma``
     - 是
     -
   * - ``torch.Tensor.mvlgamma_``
     - 是
     -
   * - ``torch.Tensor.nansum``
     - 是
     -
   * - ``torch.Tensor.narrow``
     - 是
     -
   * - ``torch.Tensor.narrow_copy``
     - 是
     -
   * - ``torch.Tensor.ndimension``
     - 是
     -
   * - ``torch.Tensor.nan_to_num``
     - 是
     -
   * - ``torch.Tensor.nan_to_num_``
     - 是
     -
   * - ``torch.Tensor.ne``
     - 是
     -
   * - ``torch.Tensor.ne_``
     - 是
     -
   * - ``torch.Tensor.not_equal``
     - 是
     -
   * - ``torch.Tensor.not_equal_``
     - 是
     -
   * - ``torch.Tensor.neg``
     - 是
     -
   * - ``torch.Tensor.neg_``
     - 是
     -
   * - ``torch.Tensor.negative``
     - 是
     -
   * - ``torch.Tensor.negative_``
     - 是
     -
   * - ``torch.Tensor.nelement``
     - 是
     -
   * - ``torch.Tensor.nextafter``
     - 是
     -
   * - ``torch.Tensor.nextafter_``
     - 是
     -
   * - ``torch.Tensor.nonzero``
     - 是
     -
   * - ``torch.Tensor.norm``
     - 是
     -
   * - ``torch.Tensor.normal_``
     - 是
     -
   * - ``torch.Tensor.numel``
     - 是
     -
   * - ``torch.Tensor.numpy``
     - 是
     -
   * - ``torch.Tensor.orgqr``
     - 是
     -
   * - ``torch.Tensor.ormqr``
     - 是
     -
   * - ``torch.Tensor.outer``
     - 是
     -
   * - ``torch.Tensor.permute``
     - 是
     -
   * - ``torch.Tensor.pin_memory``
     - 是
     -
   * - ``torch.Tensor.pinverse``
     - 是
     -
   * - ``torch.Tensor.polygamma``
     - 是
     -
   * - ``torch.Tensor.polygamma_``
     - 是
     -
   * - ``torch.Tensor.positive``
     - 是
     -
   * - ``torch.Tensor.pow``
     - 是
     -
   * - ``torch.Tensor.pow_``
     - 是
     -
   * - ``torch.Tensor.prod``
     - 是
     -
   * - ``torch.Tensor.put_``
     - 是
     -
   * - ``torch.Tensor.qr``
     - 是
     -
   * - ``torch.Tensor.qscheme``
     - 是
     -
   * - ``torch.Tensor.quantile``
     - 是
     -
   * - ``torch.Tensor.nanquantile``
     - 是
     -
   * - ``torch.Tensor.q_scale``
     - 是
     -
   * - ``torch.Tensor.q_zero_point``
     - 是
     -
   * - ``torch.Tensor.q_per_channel_scales``
     - 是
     -
   * - ``torch.Tensor.q_per_channel_zero_points``
     - 是
     -
   * - ``torch.Tensor.q_per_channel_axis``
     - 是
     -
   * - ``torch.Tensor.rad2deg``
     - 是
     -
   * - ``torch.Tensor.random_``
     - 是
     -
   * - ``torch.Tensor.ravel``
     - 是
     -
   * - ``torch.Tensor.reciprocal``
     - 是
     -
   * - ``torch.Tensor.reciprocal_``
     - 是
     -
   * - ``torch.Tensor.record_stream``
     - 是
     -
   * - ``torch.Tensor.register_hook``
     - 是
     -
   * - ``torch.Tensor.register_post_accumulate_grad_hook``
     - 是
     -
   * - ``torch.Tensor.remainder``
     - 是
     -
   * - ``torch.Tensor.remainder_``
     - 是
     -
   * - ``torch.Tensor.renorm``
     - 是
     -
   * - ``torch.Tensor.renorm_``
     - 是
     -
   * - ``torch.Tensor.repeat``
     - 是
     -
   * - ``torch.Tensor.repeat_interleave``
     - 是
     -
   * - ``torch.Tensor.requires_grad``
     - 是
     -
   * - ``torch.Tensor.requires_grad_``
     - 是
     -
   * - ``torch.Tensor.reshape``
     - 是
     -
   * - ``torch.Tensor.reshape_as``
     - 是
     -
   * - ``torch.Tensor.resize_``
     - 是
     -
   * - ``torch.Tensor.resize_as_``
     - 是
     -
   * - ``torch.Tensor.retain_grad``
     - 是
     -
   * - ``torch.Tensor.retains_grad``
     - 是
     -
   * - ``torch.Tensor.roll``
     - 是
     -
   * - ``torch.Tensor.rot90``
     - 是
     -
   * - ``torch.Tensor.round``
     - 是
     -
   * - ``torch.Tensor.round_``
     - 是
     -
   * - ``torch.Tensor.rsqrt``
     - 是
     -
   * - ``torch.Tensor.rsqrt_``
     - 是
     -
   * - ``torch.Tensor.scatter``
     - 是
     -
   * - ``torch.Tensor.scatter_``
     - 是
     -
   * - ``torch.Tensor.scatter_add_``
     - 是
     -
   * - ``torch.Tensor.scatter_add``
     - 是
     -
   * - ``torch.Tensor.scatter_reduce_``
     - 是
     -
   * - ``torch.Tensor.scatter_reduce``
     - 是
     -
   * - ``torch.Tensor.select``
     - 是
     -
   * - ``torch.Tensor.select_scatter``
     - 是
     -
   * - ``torch.Tensor.set_``
     - 是
     -
   * - ``torch.Tensor.share_memory_``
     - 是
     -
   * - ``torch.Tensor.short``
     - 是
     -
   * - ``torch.Tensor.sigmoid``
     - 是
     -
   * - ``torch.Tensor.sigmoid_``
     - 是
     -
   * - ``torch.Tensor.sign``
     - 是
     -
   * - ``torch.Tensor.sign_``
     - 是
     -
   * - ``torch.Tensor.signbit``
     - 是
     -
   * - ``torch.Tensor.sgn``
     - 是
     -
   * - ``torch.Tensor.sgn_``
     - 是
     -
   * - ``torch.Tensor.sin``
     - 是
     -
   * - ``torch.Tensor.sin_``
     - 是
     -
   * - ``torch.Tensor.sinc``
     - 是
     -
   * - ``torch.Tensor.sinc_``
     - 是
     -
   * - ``torch.Tensor.sinh``
     - 是
     -
   * - ``torch.Tensor.sinh_``
     - 是
     -
   * - ``torch.Tensor.asinh``
     - 是
     -
   * - ``torch.Tensor.asinh_``
     - 是
     -
   * - ``torch.Tensor.arcsinh``
     - 是
     -
   * - ``torch.Tensor.arcsinh_``
     - 是
     -
   * - ``torch.Tensor.shape``
     - 是
     -
   * - ``torch.Tensor.size``
     - 是
     -
   * - ``torch.Tensor.slogdet``
     - 是
     -
   * - ``torch.Tensor.slice_scatter``
     - 是
     -
   * - ``torch.Tensor.softmax``
     - 是
     -
   * - ``torch.Tensor.sort``
     - 是
     -
   * - ``torch.Tensor.split``
     - 是
     -
   * - ``torch.Tensor.sparse_mask``
     - 是
     -
   * - ``torch.Tensor.sparse_dim``
     - 是
     -
   * - ``torch.Tensor.sqrt``
     - 是
     -
   * - ``torch.Tensor.sqrt_``
     - 是
     -
   * - ``torch.Tensor.square``
     - 是
     -
   * - ``torch.Tensor.square_``
     - 是
     -
   * - ``torch.Tensor.squeeze``
     - 是
     -
   * - ``torch.Tensor.squeeze_``
     - 是
     -
   * - ``torch.Tensor.std``
     - 是
     -
   * - ``torch.Tensor.stft``
     - 是
     -
   * - ``torch.Tensor.storage``
     - 是
     -
   * - ``torch.Tensor.untyped_storage``
     - 是
     -
   * - ``torch.Tensor.storage_offset``
     - 是
     -
   * - ``torch.Tensor.storage_type``
     - 是
     -
   * - ``torch.Tensor.stride``
     - 是
     -
   * - ``torch.Tensor.sub``
     - 是
     -
   * - ``torch.Tensor.sub_``
     - 是
     -
   * - ``torch.Tensor.subtract``
     - 是
     -
   * - ``torch.Tensor.subtract_``
     - 是
     -
   * - ``torch.Tensor.sum``
     - 是
     -
   * - ``torch.Tensor.sum_to_size``
     - 是
     -
   * - ``torch.Tensor.svd``
     - 是
     -
   * - ``torch.Tensor.swapaxes``
     - 是
     -
   * - ``torch.Tensor.swapdims``
     - 是
     -
   * - ``torch.Tensor.t``
     - 是
     -
   * - ``torch.Tensor.t_``
     - 是
     -
   * - ``torch.Tensor.tensor_split``
     - 是
     -
   * - ``torch.Tensor.tile``
     - 是
     -
   * - ``torch.Tensor.to``
     - 是
     -
   * - ``torch.Tensor.to_mkldnn``
     - 是
     -
   * - ``torch.Tensor.take``
     - 是
     -
   * - ``torch.Tensor.take_along_dim``
     - 是
     -
   * - ``torch.Tensor.tan``
     - 是
     -
   * - ``torch.Tensor.tan_``
     - 是
     -
   * - ``torch.Tensor.tanh``
     - 是
     -
   * - ``torch.Tensor.tanh_``
     - 是
     -
   * - ``torch.Tensor.atanh``
     - 是
     -
   * - ``torch.Tensor.atanh_``
     - 是
     -
   * - ``torch.Tensor.arctanh``
     - 是
     -
   * - ``torch.Tensor.arctanh_``
     - 是
     -
   * - ``torch.Tensor.tolist``
     - 是
     -
   * - ``torch.Tensor.topk``
     - 是
     -
   * - ``torch.Tensor.to_dense``
     - 是
     -
   * - ``torch.Tensor.to_sparse``
     - 是
     -
   * - ``torch.Tensor.to_sparse_csr``
     - 是
     -
   * - ``torch.Tensor.to_sparse_csc``
     - 是
     -
   * - ``torch.Tensor.to_sparse_bsr``
     - 是
     -
   * - ``torch.Tensor.to_sparse_bsc``
     - 是
     -
   * - ``torch.Tensor.trace``
     - 是
     -
   * - ``torch.Tensor.transpose``
     - 是
     -
   * - ``torch.Tensor.transpose_``
     - 是
     -
   * - ``torch.Tensor.triangular_solve``
     - 是
     -
   * - ``torch.Tensor.tril``
     - 是
     -
   * - ``torch.Tensor.tril_``
     - 是
     -
   * - ``torch.Tensor.triu``
     - 是
     -
   * - ``torch.Tensor.triu_``
     - 是
     -
   * - ``torch.Tensor.true_divide``
     - 是
     -
   * - ``torch.Tensor.true_divide_``
     - 是
     -
   * - ``torch.Tensor.trunc``
     - 是
     -
   * - ``torch.Tensor.trunc_``
     - 是
     -
   * - ``torch.Tensor.type``
     - 是
     -
   * - ``torch.Tensor.type_as``
     - 是
     -
   * - ``torch.Tensor.unbind``
     - 是
     -
   * - ``torch.Tensor.unflatten``
     - 是
     -
   * - ``torch.Tensor.unfold``
     - 是
     -
   * - ``torch.Tensor.uniform_``
     - 是
     -
   * - ``torch.Tensor.unique``
     - 是
     -
   * - ``torch.Tensor.unique_consecutive``
     - 是
     -
   * - ``torch.Tensor.unsqueeze``
     - 是
     -
   * - ``torch.Tensor.unsqueeze_``
     - 是
     -
   * - ``torch.Tensor.values``
     - 是
     -
   * - ``torch.Tensor.var``
     - 是
     -
   * - ``torch.Tensor.vdot``
     - 是
     -
   * - ``torch.Tensor.view``
     - 是
     -
   * - ``torch.Tensor.view_as``
     - 是
     -
   * - ``torch.Tensor.vsplit``
     - 是
     -
   * - ``torch.Tensor.where``
     - 是
     -
   * - ``torch.Tensor.xlogy``
     - 是
     -
   * - ``torch.Tensor.xlogy_``
     - 是
     -
   * - ``torch.Tensor.xpu``
     - 是
     -
   * - ``torch.Tensor.zero_``
     - 是
     -
