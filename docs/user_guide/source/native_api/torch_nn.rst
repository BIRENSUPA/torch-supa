.. _native-api-torch_nn:

torch.nn
========

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.nn.parameter.Parameter``
     - 是
     - 
   * - ``torch.nn.parameter.UninitializedParameter``
     - 是
     - 
   * - ``torch.nn.parameter.UninitializedParameter.cls_to_become``
     - 是
     - 
   * - ``torch.nn.parameter.UninitializedBuffer``
     - 是
     - 
   * - ``torch.nn.Module``
     - 是
     - 
   * - ``torch.nn.Module.add_module``
     - 是
     - 
   * - ``torch.nn.Module.apply``
     - 是
     - 
   * - ``torch.nn.Module.bfloat16``
     - 是
     - 
   * - ``torch.nn.Module.buffers``
     - 是
     - 
   * - ``torch.nn.Module.children``
     - 是
     - 
   * - ``torch.nn.Module.compile``
     - 是
     - 
   * - ``torch.nn.Module.cpu``
     - 是
     - 
   * - ``torch.nn.Module.cuda``
     - 是
     - 
   * - ``torch.nn.Module.double``
     - 是
     - 
   * - ``torch.nn.Module.eval``
     - 是
     - 
   * - ``torch.nn.Module.extra_repr``
     - 是
     - 
   * - ``torch.nn.Module.float``
     - 是
     - 
   * - ``torch.nn.Module.forward``
     - 是
     - 
   * - ``torch.nn.Module.get_buffer``
     - 是
     - 
   * - ``torch.nn.Module.get_extra_state``
     - 是
     - 
   * - ``torch.nn.Module.get_parameter``
     - 是
     - 
   * - ``torch.nn.Module.get_submodule``
     - 是
     - 
   * - ``torch.nn.Module.half``
     - 是
     - 
   * - ``torch.nn.Module.ipu``
     - 是
     - 
   * - ``torch.nn.Module.load_state_dict``
     - 是
     - 
   * - ``torch.nn.Module.modules``
     - 是
     - 
   * - ``torch.nn.Module.named_buffers``
     - 是
     - 
   * - ``torch.nn.Module.named_children``
     - 是
     - 
   * - ``torch.nn.Module.named_modules``
     - 是
     - 
   * - ``torch.nn.Module.named_parameters``
     - 是
     - 
   * - ``torch.nn.Module.parameters``
     - 是
     - 
   * - ``torch.nn.Module.register_backward_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_buffer``
     - 是
     - 
   * - ``torch.nn.Module.register_forward_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_forward_pre_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_full_backward_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_full_backward_pre_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_load_state_dict_post_hook``
     - 是
     - 
   * - ``torch.nn.Module.register_module``
     - 是
     - 
   * - ``torch.nn.Module.register_parameter``
     - 是
     - 
   * - ``torch.nn.Module.register_state_dict_pre_hook``
     - 是
     - 
   * - ``torch.nn.Module.requires_grad_``
     - 是
     - 
   * - ``torch.nn.Module.set_extra_state``
     - 是
     - 
   * - ``torch.nn.Module.share_memory``
     - 是
     - 
   * - ``torch.nn.Module.state_dict``
     - 是
     - 
   * - ``torch.nn.Module.to``
     - 是
     - 
   * - ``torch.nn.Module.to_empty``
     - 是
     - 
   * - ``torch.nn.Module.train``
     - 是
     - 
   * - ``torch.nn.Module.type``
     - 是
     - 
   * - ``torch.nn.Module.xpu``
     - 是
     - 
   * - ``torch.nn.Module.zero_grad``
     - 是
     - 
   * - ``torch.nn.Sequential``
     - 是
     - 
   * - ``torch.nn.Sequential.append``
     - 是
     - 
   * - ``torch.nn.ModuleList``
     - 是
     - 
   * - ``torch.nn.ModuleList.append``
     - 是
     - 
   * - ``torch.nn.ModuleList.extend``
     - 是
     - 
   * - ``torch.nn.ModuleList.insert``
     - 是
     - 
   * - ``torch.nn.ModuleDict``
     - 是
     - 
   * - ``torch.nn.ModuleDict.clear``
     - 是
     - 
   * - ``torch.nn.ModuleDict.items``
     - 是
     - 
   * - ``torch.nn.ModuleDict.keys``
     - 是
     - 
   * - ``torch.nn.ModuleDict.pop``
     - 是
     - 
   * - ``torch.nn.ModuleDict.update``
     - 是
     - 
   * - ``torch.nn.ModuleDict.values``
     - 是
     - 
   * - ``torch.nn.ParameterList``
     - 是
     - 
   * - ``torch.nn.ParameterList.append``
     - 是
     - 
   * - ``torch.nn.ParameterList.extend``
     - 是
     - 
   * - ``torch.nn.ParameterDict``
     - 是
     - 
   * - ``torch.nn.ParameterDict.clear``
     - 是
     - 
   * - ``torch.nn.ParameterDict.copy``
     - 是
     - 
   * - ``torch.nn.ParameterDict.fromkeys``
     - 是
     - 
   * - ``torch.nn.ParameterDict.get``
     - 是
     - 
   * - ``torch.nn.ParameterDict.items``
     - 是
     - 
   * - ``torch.nn.ParameterDict.keys``
     - 是
     - 
   * - ``torch.nn.ParameterDict.pop``
     - 是
     - 
   * - ``torch.nn.ParameterDict.popitem``
     - 是
     - 
   * - ``torch.nn.ParameterDict.setdefault``
     - 是
     - 
   * - ``torch.nn.ParameterDict.update``
     - 是
     - 
   * - ``torch.nn.ParameterDict.values``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_forward_pre_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_forward_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_backward_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_full_backward_pre_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_full_backward_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_buffer_registration_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_module_registration_hook``
     - 是
     - 
   * - ``torch.nn.modules.module.register_module_parameter_registration_hook``
     - 是
     - 
   * - ``torch.nn.Conv1d``
     - 是
     - 
   * - ``torch.nn.Conv2d``
     - 是
     - 
   * - ``torch.nn.Conv3d``
     - 是
     - 
   * - ``torch.nn.ConvTranspose1d``
     - 是
     - 
   * - ``torch.nn.ConvTranspose2d``
     - 是
     - 
   * - ``torch.nn.ConvTranspose3d``
     - 是
     - 
   * - ``torch.nn.LazyConv1d``
     - 是
     - 
   * - ``torch.nn.LazyConv1d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyConv2d``
     - 是
     - 
   * - ``torch.nn.LazyConv2d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyConv3d``
     - 是
     - 
   * - ``torch.nn.LazyConv3d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose1d``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose1d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose2d``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose2d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose3d``
     - 是
     - 
   * - ``torch.nn.LazyConvTranspose3d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.Unfold``
     - 是
     - 
   * - ``torch.nn.Fold``
     - 是
     - 
   * - ``torch.nn.MaxPool1d``
     - 是
     - 
   * - ``torch.nn.MaxPool2d``
     - 是
     - 
   * - ``torch.nn.MaxPool3d``
     - 是
     - 
   * - ``torch.nn.MaxUnpool1d``
     - 是
     - 
   * - ``torch.nn.MaxUnpool2d``
     - 是
     - 
   * - ``torch.nn.MaxUnpool3d``
     - 是
     - 
   * - ``torch.nn.AvgPool1d``
     - 是
     - 
   * - ``torch.nn.AvgPool2d``
     - 是
     - 
   * - ``torch.nn.AvgPool3d``
     - 是
     - 
   * - ``torch.nn.FractionalMaxPool2d``
     - 是
     - 
   * - ``torch.nn.FractionalMaxPool3d``
     - 是
     - 
   * - ``torch.nn.LPPool1d``
     - 是
     - 
   * - ``torch.nn.LPPool2d``
     - 是
     - 
   * - ``torch.nn.AdaptiveMaxPool1d``
     - 是
     - 
   * - ``torch.nn.AdaptiveMaxPool2d``
     - 是
     - 
   * - ``torch.nn.AdaptiveMaxPool3d``
     - 是
     - 
   * - ``torch.nn.AdaptiveAvgPool1d``
     - 是
     - 
   * - ``torch.nn.AdaptiveAvgPool2d``
     - 是
     - 
   * - ``torch.nn.AdaptiveAvgPool3d``
     - 是
     - 
   * - ``torch.nn.ReflectionPad1d``
     - 是
     - 
   * - ``torch.nn.ReflectionPad2d``
     - 是
     - 
   * - ``torch.nn.ReflectionPad3d``
     - 是
     - 
   * - ``torch.nn.ReplicationPad1d``
     - 是
     - 
   * - ``torch.nn.ReplicationPad2d``
     - 是
     - 
   * - ``torch.nn.ReplicationPad3d``
     - 是
     - 
   * - ``torch.nn.ZeroPad1d``
     - 是
     - 
   * - ``torch.nn.ZeroPad2d``
     - 是
     - 
   * - ``torch.nn.ZeroPad3d``
     - 是
     - 
   * - ``torch.nn.ConstantPad1d``
     - 是
     - 
   * - ``torch.nn.ConstantPad2d``
     - 是
     - 
   * - ``torch.nn.ConstantPad3d``
     - 是
     - 
   * - ``torch.nn.ELU``
     - 是
     - 
   * - ``torch.nn.Hardshrink``
     - 是
     - 
   * - ``torch.nn.Hardsigmoid``
     - 是
     - 
   * - ``torch.nn.Hardtanh``
     - 是
     - 
   * - ``torch.nn.Hardswish``
     - 是
     - 
   * - ``torch.nn.LeakyReLU``
     - 是
     - 
   * - ``torch.nn.LogSigmoid``
     - 是
     - 
   * - ``torch.nn.MultiheadAttention``
     - 是
     - 
   * - ``torch.nn.MultiheadAttention.forward``
     - 是
     - 
   * - ``torch.nn.MultiheadAttention.merge_masks``
     - 是
     - 
   * - ``torch.nn.PReLU``
     - 是
     - 
   * - ``torch.nn.ReLU``
     - 是
     - 
   * - ``torch.nn.ReLU6``
     - 是
     - 
   * - ``torch.nn.RReLU``
     - 是
     - 
   * - ``torch.nn.SELU``
     - 是
     - 
   * - ``torch.nn.CELU``
     - 是
     - 
   * - ``torch.nn.GELU``
     - 是
     - 
   * - ``torch.nn.Sigmoid``
     - 是
     - 
   * - ``torch.nn.SiLU``
     - 是
     - 
   * - ``torch.nn.Mish``
     - 是
     - 
   * - ``torch.nn.Softplus``
     - 是
     - 
   * - ``torch.nn.Softshrink``
     - 是
     - 
   * - ``torch.nn.Softsign``
     - 是
     - 
   * - ``torch.nn.Tanh``
     - 是
     - 
   * - ``torch.nn.Tanhshrink``
     - 是
     - 
   * - ``torch.nn.Threshold``
     - 是
     - 
   * - ``torch.nn.GLU``
     - 是
     - 
   * - ``torch.nn.Softmin``
     - 是
     - 
   * - ``torch.nn.Softmax``
     - 是
     - 
   * - ``torch.nn.Softmax2d``
     - 是
     - 
   * - ``torch.nn.LogSoftmax``
     - 是
     - 
   * - ``torch.nn.AdaptiveLogSoftmaxWithLoss``
     - 是
     - 
   * - ``torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob``
     - 是
     - 
   * - ``torch.nn.AdaptiveLogSoftmaxWithLoss.predict``
     - 是
     - 
   * - ``torch.nn.BatchNorm1d``
     - 是
     - 
   * - ``torch.nn.BatchNorm2d``
     - 是
     - 
   * - ``torch.nn.BatchNorm3d``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm1d``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm1d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm2d``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm2d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm3d``
     - 是
     - 
   * - ``torch.nn.LazyBatchNorm3d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.GroupNorm``
     - 是
     - 
   * - ``torch.nn.SyncBatchNorm``
     - 是
     - 
   * - ``torch.nn.SyncBatchNorm.convert_sync_batchnorm``
     - 是
     - 
   * - ``torch.nn.InstanceNorm1d``
     - 是
     - 
   * - ``torch.nn.InstanceNorm2d``
     - 是
     - 
   * - ``torch.nn.InstanceNorm3d``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm1d``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm1d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm2d``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm2d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm3d``
     - 是
     - 
   * - ``torch.nn.LazyInstanceNorm3d.cls_to_become``
     - 是
     - 
   * - ``torch.nn.LayerNorm``
     - 是
     - 
   * - ``torch.nn.LocalResponseNorm``
     - 是
     - 
   * - ``torch.nn.RNNBase``
     - 是
     - 
   * - ``torch.nn.RNNBase.flatten_parameters``
     - 是
     - 
   * - ``torch.nn.RNN``
     - 是
     - 
   * - ``torch.nn.LSTM``
     - 是
     - 
   * - ``torch.nn.GRU``
     - 是
     - 
   * - ``torch.nn.RNNCell``
     - 是
     - 
   * - ``torch.nn.LSTMCell``
     - 是
     - 
   * - ``torch.nn.GRUCell``
     - 是
     - 
   * - ``torch.nn.Transformer``
     - 是
     - 
   * - ``torch.nn.Transformer.forward``
     - 是
     - 
   * - ``torch.nn.Transformer.generate_square_subsequent_mask``
     - 是
     - 
   * - ``torch.nn.TransformerEncoder``
     - 是
     - 
   * - ``torch.nn.TransformerEncoder.forward``
     - 是
     - 
   * - ``torch.nn.TransformerDecoder``
     - 是
     - 
   * - ``torch.nn.TransformerDecoder.forward``
     - 是
     - 
   * - ``torch.nn.TransformerEncoderLayer``
     - 是
     - 
   * - ``torch.nn.TransformerEncoderLayer.forward``
     - 是
     - 
   * - ``torch.nn.TransformerDecoderLayer``
     - 是
     - 
   * - ``torch.nn.TransformerDecoderLayer.forward``
     - 是
     - 
   * - ``torch.nn.Identity``
     - 是
     - 
   * - ``torch.nn.Linear``
     - 是
     - 
   * - ``torch.nn.Bilinear``
     - 是
     - 
   * - ``torch.nn.LazyLinear``
     - 是
     - 
   * - ``torch.nn.LazyLinear.cls_to_become``
     - 是
     - 
   * - ``torch.nn.Dropout``
     - 是
     - 
   * - ``torch.nn.Dropout1d``
     - 是
     - 
   * - ``torch.nn.Dropout2d``
     - 是
     - 
   * - ``torch.nn.Dropout3d``
     - 是
     - 
   * - ``torch.nn.AlphaDropout``
     - 是
     - 
   * - ``torch.nn.FeatureAlphaDropout``
     - 是
     - 
   * - ``torch.nn.Embedding``
     - 是
     - 
   * - ``torch.nn.Embedding.from_pretrained``
     - 是
     - 
   * - ``torch.nn.EmbeddingBag``
     - 是
     - 
   * - ``torch.nn.EmbeddingBag.forward``
     - 是
     - 
   * - ``torch.nn.EmbeddingBag.from_pretrained``
     - 是
     - 
   * - ``torch.nn.CosineSimilarity``
     - 是
     - 
   * - ``torch.nn.PairwiseDistance``
     - 是
     - 
   * - ``torch.nn.L1Loss``
     - 是
     - 
   * - ``torch.nn.MSELoss``
     - 是
     - 
   * - ``torch.nn.CrossEntropyLoss``
     - 是
     - 
   * - ``torch.nn.CTCLoss``
     - 是
     - 
   * - ``torch.nn.NLLLoss``
     - 是
     - 
   * - ``torch.nn.PoissonNLLLoss``
     - 是
     - 
   * - ``torch.nn.GaussianNLLLoss``
     - 是
     - 
   * - ``torch.nn.KLDivLoss``
     - 是
     - 
   * - ``torch.nn.BCELoss``
     - 是
     - 
   * - ``torch.nn.BCEWithLogitsLoss``
     - 是
     - 
   * - ``torch.nn.MarginRankingLoss``
     - 是
     - 
   * - ``torch.nn.HingeEmbeddingLoss``
     - 是
     - 
   * - ``torch.nn.MultiLabelMarginLoss``
     - 是
     - 
   * - ``torch.nn.HuberLoss``
     - 是
     - 
   * - ``torch.nn.SmoothL1Loss``
     - 是
     - 
   * - ``torch.nn.SoftMarginLoss``
     - 是
     - 
   * - ``torch.nn.MultiLabelSoftMarginLoss``
     - 是
     - 
   * - ``torch.nn.CosineEmbeddingLoss``
     - 是
     - 
   * - ``torch.nn.MultiMarginLoss``
     - 是
     - 
   * - ``torch.nn.TripletMarginLoss``
     - 是
     - 
   * - ``torch.nn.TripletMarginWithDistanceLoss``
     - 是
     - 
   * - ``torch.nn.PixelShuffle``
     - 是
     - 
   * - ``torch.nn.PixelUnshuffle``
     - 是
     - 
   * - ``torch.nn.Upsample``
     - 是
     - 
   * - ``torch.nn.UpsamplingNearest2d``
     - 是
     - 
   * - ``torch.nn.UpsamplingBilinear2d``
     - 是
     - 
   * - ``torch.nn.ChannelShuffle``
     - 是
     - 
   * - ``torch.nn.DataParallel``
     - 是
     - 
   * - ``torch.nn.parallel.DistributedDataParallel``
     - 是
     - 
   * - ``torch.nn.parallel.DistributedDataParallel.join``
     - 是
     - 
   * - ``torch.nn.parallel.DistributedDataParallel.join_hook``
     - 是
     - 
   * - ``torch.nn.parallel.DistributedDataParallel.no_sync``
     - 是
     - 
   * - ``torch.nn.parallel.DistributedDataParallel.register_comm_hook``
     - 是
     - 
   * - ``torch.nn.utils.clip_grad_norm_``
     - 是
     - 
   * - ``torch.nn.utils.clip_grad_value_``
     - 是
     - 
   * - ``torch.nn.utils.parameters_to_vector``
     - 是
     - 
   * - ``torch.nn.utils.vector_to_parameters``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod.compute_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.BasePruningMethod.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.add_pruning_method``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.compute_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.PruningContainer.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.Identity``
     - 是
     - 
   * - ``torch.nn.utils.prune.Identity.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.Identity.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.Identity.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.Identity.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomUnstructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomUnstructured.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomUnstructured.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomUnstructured.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomUnstructured.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.L1Unstructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.L1Unstructured.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.L1Unstructured.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.L1Unstructured.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.L1Unstructured.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured.compute_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.RandomStructured.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured.compute_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.LnStructured.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.CustomFromMask``
     - 是
     - 
   * - ``torch.nn.utils.prune.CustomFromMask.apply``
     - 是
     - 
   * - ``torch.nn.utils.prune.CustomFromMask.apply_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.CustomFromMask.prune``
     - 是
     - 
   * - ``torch.nn.utils.prune.CustomFromMask.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.identity``
     - 是
     - 
   * - ``torch.nn.utils.prune.random_unstructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.l1_unstructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.random_structured``
     - 是
     - 
   * - ``torch.nn.utils.prune.ln_structured``
     - 是
     - 
   * - ``torch.nn.utils.prune.global_unstructured``
     - 是
     - 
   * - ``torch.nn.utils.prune.custom_from_mask``
     - 是
     - 
   * - ``torch.nn.utils.prune.remove``
     - 是
     - 
   * - ``torch.nn.utils.prune.is_pruned``
     - 是
     - 
   * - ``torch.nn.utils.weight_norm``
     - 是
     - 
   * - ``torch.nn.utils.remove_weight_norm``
     - 是
     - 
   * - ``torch.nn.utils.spectral_norm``
     - 是
     - 
   * - ``torch.nn.utils.remove_spectral_norm``
     - 是
     - 
   * - ``torch.nn.utils.skip_init``
     - 是
     - 
   * - ``torch.nn.utils.parametrizations.orthogonal``
     - 是
     - 
   * - ``torch.nn.utils.parametrizations.spectral_norm``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.register_parametrization``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.remove_parametrizations``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.cached``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.is_parametrized``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.ParametrizationList``
     - 是
     - 
   * - ``torch.nn.utils.parametrize.ParametrizationList.right_inverse``
     - 是
     - 
   * - ``torch.nn.utils.stateless.functional_call``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.batch_sizes``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.count``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.data``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.index``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.is_cuda``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.is_pinned``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.sorted_indices``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.to``
     - 是
     - 
   * - ``torch.nn.utils.rnn.PackedSequence.unsorted_indices``
     - 是
     - 
   * - ``torch.nn.utils.rnn.pack_padded_sequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.pad_packed_sequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.pad_sequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.pack_sequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.unpack_sequence``
     - 是
     - 
   * - ``torch.nn.utils.rnn.unpad_sequence``
     - 是
     - 
   * - ``torch.nn.Flatten``
     - 是
     - 
   * - ``torch.nn.Unflatten``
     - 是
     - 
   * - ``torch.nn.modules.lazy.LazyModuleMixin``
     - 是
     - 
   * - ``torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params``
     - 是
     - 
   * - ``torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters``
     - 是
     - 
