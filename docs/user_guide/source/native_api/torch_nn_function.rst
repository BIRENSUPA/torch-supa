.. _native-api-torch_nn_function:

torch.nn.functional
===================

.. list-table::
   :header-rows: 1
   :widths: 50 10 40

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.nn.functional.conv1d``
     - 是
     -
   * - ``torch.nn.functional.conv2d``
     - 是
     -
   * - ``torch.nn.functional.conv3d``
     - 是
     -
   * - ``torch.nn.functional.conv_transpose1d``
     - 是
     -
   * - ``torch.nn.functional.conv_transpose2d``
     - 是
     -
   * - ``torch.nn.functional.conv_transpose3d``
     - 是
     -
   * - ``torch.nn.functional.unfold``
     - 是
     -
   * - ``torch.nn.functional.fold``
     - 是
     -
   * - ``torch.nn.functional.avg_pool1d``
     - 是
     -
   * - ``torch.nn.functional.avg_pool2d``
     - 是
     -
   * - ``torch.nn.functional.avg_pool3d``
     - 是
     -
   * - ``torch.nn.functional.max_pool1d``
     - 是
     -
   * - ``torch.nn.functional.max_pool2d``
     - 是
     -
   * - ``torch.nn.functional.max_pool3d``
     - 是
     -
   * - ``torch.nn.functional.max_unpool1d``
     - 是
     -
   * - ``torch.nn.functional.max_unpool2d``
     - 是
     -
   * - ``torch.nn.functional.max_unpool3d``
     - 是
     -
   * - ``torch.nn.functional.lp_pool1d``
     - 是
     -
   * - ``torch.nn.functional.lp_pool2d``
     - 是
     -
   * - ``torch.nn.functional.lp_pool3d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_max_pool1d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_max_pool2d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_max_pool3d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_avg_pool1d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_avg_pool2d``
     - 是
     -
   * - ``torch.nn.functional.adaptive_avg_pool3d``
     - 是
     -
   * - ``torch.nn.functional.fractional_max_pool2d``
     - 是
     -
   * - ``torch.nn.functional.fractional_max_pool3d``
     - 是
     -
   * - ``torch.nn.functional.scaled_dot_product_attention``
     - 是
     - - 不支持 ``efficient_attention`` 后端, - ``flash_attention`` 后端，仅支持 ``BFloat16`` 和 ``FLOAT32``, 不支持 ``attn_mask``，head_size小于等于512, 
   * - ``torch.nn.functional.threshold``
     - 是
     -
   * - ``torch.nn.functional.threshold_``
     - 是
     -
   * - ``torch.nn.functional.relu``
     - 是
     -
   * - ``torch.nn.functional.relu_``
     - 是
     -
   * - ``torch.nn.functional.hardtanh``
     - 是
     -
   * - ``torch.nn.functional.hardtanh_``
     - 是
     -
   * - ``torch.nn.functional.hardswish``
     - 是
     -
   * - ``torch.nn.functional.relu6``
     - 是
     -
   * - ``torch.nn.functional.elu``
     - 是
     -
   * - ``torch.nn.functional.elu_``
     - 是
     -
   * - ``torch.nn.functional.selu``
     - 是
     -
   * - ``torch.nn.functional.celu``
     - 是
     -
   * - ``torch.nn.functional.leaky_relu``
     - 是
     -
   * - ``torch.nn.functional.leaky_relu_``
     - 是
     -
   * - ``torch.nn.functional.prelu``
     - 是
     -
   * - ``torch.nn.functional.rrelu``
     - 是
     -
   * - ``torch.nn.functional.rrelu_``
     - 是
     -
   * - ``torch.nn.functional.glu``
     - 是
     -
   * - ``torch.nn.functional.gelu``
     - 是
     -
   * - ``torch.nn.functional.logsigmoid``
     - 是
     -
   * - ``torch.nn.functional.hardshrink``
     - 是
     -
   * - ``torch.nn.functional.tanhshrink``
     - 是
     -
   * - ``torch.nn.functional.softsign``
     - 是
     -
   * - ``torch.nn.functional.softplus``
     - 是
     -
   * - ``torch.nn.functional.softmin``
     - 是
     -
   * - ``torch.nn.functional.softmax``
     - 是
     -
   * - ``torch.nn.functional.softshrink``
     - 是
     -
   * - ``torch.nn.functional.gumbel_softmax``
     - 是
     -
   * - ``torch.nn.functional.log_softmax``
     - 是
     -
   * - ``torch.nn.functional.tanh``
     - 是
     -
   * - ``torch.nn.functional.sigmoid``
     - 是
     -
   * - ``torch.nn.functional.hardsigmoid``
     - 是
     -
   * - ``torch.nn.functional.silu``
     - 是
     -
   * - ``torch.nn.functional.mish``
     - 是
     -
   * - ``torch.nn.functional.batch_norm``
     - 是
     -
   * - ``torch.nn.functional.group_norm``
     - 是
     -
   * - ``torch.nn.functional.instance_norm``
     - 是
     -
   * - ``torch.nn.functional.layer_norm``
     - 是
     -
   * - ``torch.nn.functional.local_response_norm``
     - 是
     -
   * - ``torch.nn.functional.rms_norm``
     - 是
     -
   * - ``torch.nn.functional.normalize``
     - 是
     -
   * - ``torch.nn.functional.linear``
     - 是
     -
   * - ``torch.nn.functional.bilinear``
     - 是
     -
   * - ``torch.nn.functional.dropout``
     - 是
     -
   * - ``torch.nn.functional.alpha_dropout``
     - 是
     -
   * - ``torch.nn.functional.feature_alpha_dropout``
     - 是
     -
   * - ``torch.nn.functional.dropout1d``
     - 是
     -
   * - ``torch.nn.functional.dropout2d``
     - 是
     -
   * - ``torch.nn.functional.dropout3d``
     - 是
     -
   * - ``torch.nn.functional.embedding``
     - 是
     -
   * - ``torch.nn.functional.embedding_bag``
     - 是
     -
   * - ``torch.nn.functional.one_hot``
     - 是
     -
   * - ``torch.nn.functional.pairwise_distance``
     - 是
     -
   * - ``torch.nn.functional.cosine_similarity``
     - 是
     -
   * - ``torch.nn.functional.pdist``
     - 是
     -
   * - ``torch.nn.functional.binary_cross_entropy``
     - 是
     -
   * - ``torch.nn.functional.binary_cross_entropy_with_logits``
     - 是
     -
   * - ``torch.nn.functional.poisson_nll_loss``
     - 是
     -
   * - ``torch.nn.functional.cosine_embedding_loss``
     - 是
     -
   * - ``torch.nn.functional.cross_entropy``
     - 是
     -
   * - ``torch.nn.functional.ctc_loss``
     - 是
     -
   * - ``torch.nn.functional.gaussian_nll_loss``
     - 是
     -
   * - ``torch.nn.functional.hinge_embedding_loss``
     - 是
     -
   * - ``torch.nn.functional.kl_div``
     - 是
     -
   * - ``torch.nn.functional.l1_loss``
     - 是
     -
   * - ``torch.nn.functional.mse_loss``
     - 是
     -
   * - ``torch.nn.functional.margin_ranking_loss``
     - 是
     -
   * - ``torch.nn.functional.multilabel_margin_loss``
     - 是
     -
   * - ``torch.nn.functional.multilabel_soft_margin_loss``
     - 是
     -
   * - ``torch.nn.functional.multi_margin_loss``
     - 是
     -
   * - ``torch.nn.functional.nll_loss``
     - 是
     -
   * - ``torch.nn.functional.huber_loss``
     - 是
     -
   * - ``torch.nn.functional.smooth_l1_loss``
     - 是
     -
   * - ``torch.nn.functional.soft_margin_loss``
     - 是
     -
   * - ``torch.nn.functional.triplet_margin_loss``
     - 是
     -
   * - ``torch.nn.functional.triplet_margin_with_distance_loss``
     - 是
     -
   * - ``torch.nn.functional.pixel_shuffle``
     - 是
     -
   * - ``torch.nn.functional.pixel_unshuffle``
     - 是
     -
   * - ``torch.nn.functional.pad``
     - 是
     -
   * - ``torch.nn.functional.interpolate``
     - 是
     -
   * - ``torch.nn.functional.upsample``
     - 是
     -
   * - ``torch.nn.functional.upsample_nearest``
     - 是
     -
   * - ``torch.nn.functional.upsample_bilinear``
     - 是
     -
   * - ``torch.nn.functional.grid_sample``
     - 是
     -
   * - ``torch.nn.functional.affine_grid``
     - 是
     -
   * - ``torch.nn.functional.torch.nn.parallel.data_parallel``
     - 是
     -
   * - ``ScalingType``
     - 是
     -
   * - ``SwizzleType``
     - 是
     -
   * - ``torch.nn.functional.grouped_mm``
     - 是
     -
   * - ``torch.nn.functional.scaled_mm``
     - 是
     -
   * - ``torch.nn.functional.scaled_grouped_mm``
     - 是
     -
