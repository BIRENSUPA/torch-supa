.. _native-api-quantization:

Quantization
================

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.ao.quantization.quantize``
     - 
     - 
   * - ``torch.ao.quantization.quantize_dynamic``
     - 
     - 
   * - ``torch.ao.quantization.quantize_qat``
     - 
     - 
   * - ``torch.ao.quantization.prepare``
     - 
     - 
   * - ``torch.ao.quantization.prepare_qat``
     - 
     - 
   * - ``torch.ao.quantization.convert``
     - 
     - 
   * - ``torch.ao.quantization.fuse_modules``
     - 
     - 
   * - ``torch.ao.quantization.QuantStub``
     - 
     - 
   * - ``torch.ao.quantization.DeQuantStub``
     - 
     - 
   * - ``torch.ao.quantization.QuantWrapper``
     - 
     - 
   * - ``torch.ao.quantization.add_quant_dequant``
     - 
     - 
   * - ``torch.ao.quantization.swap_module``
     - 
     - 
   * - ``torch.ao.quantization.propagate_qconfig_``
     - 
     - 
   * - ``torch.ao.quantization.default_eval_fn``
     - 
     - 
   * - ``torch.ao.quantization.quantize_fx.prepare_fx``
     - 
     - 
   * - ``torch.ao.quantization.quantize_fx.prepare_qat_fx``
     - 
     - 
   * - ``torch.ao.quantization.quantize_fx.convert_fx``
     - 
     - 
   * - ``torch.ao.quantization.quantize_fx.fuse_fx``
     - 
     - 
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendConfig.configs``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.BackendConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendConfig.set_name``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.BackendPatternConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.DTypeConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.DTypeConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.DTypeConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.DTypeWithConstraints``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.ObservationType``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT``
     - 
     - 
   * - ``torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT``
     - 
     - 
   * - ``torch.ao.quantization.fx.custom_config.FuseCustomConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.ConvertCustomConfig``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict``
     - 
     - 
   * - ``torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.ObserverBase``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.ObserverBase.with_args``
     - 
     - 
   * - ``torch.ao.quantization.observer.ObserverBase.with_callable_args``
     - 
     - 
   * - ``torch.ao.quantization.observer.MinMaxObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.MinMaxObserver.calculate_qparams``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.MinMaxObserver.forward``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.MovingAverageMinMaxObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.PerChannelMinMaxObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.HistogramObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.PlaceholderObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.RecordingObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.NoopObserver``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.quantization.observer.get_observer_state_dict``
     - 
     - 
   * - ``torch.ao.quantization.observer.load_observer_state_dict``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_placeholder_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_debug_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_weight_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_histogram_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_per_channel_weight_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_dynamic_quant_observer``
     - 
     - 
   * - ``torch.ao.quantization.observer.default_float_qparams_observer``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.FakeQuantizeBase``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.FakeQuantize``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_weight_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_histogram_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_fused_act_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.disable_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.enable_fake_quant``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.disable_observer``
     - 
     - 
   * - ``torch.ao.quantization.fake_quantize.enable_observer``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.QConfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_debug_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_per_channel_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_dynamic_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.float16_dynamic_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.float16_static_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.per_channel_dynamic_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_qat_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_weight_only_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_activation_only_qconfig``
     - 
     - 
   * - ``torch.ao.quantization.qconfig.default_qat_qconfig_v2``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvReLU1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.LinearReLU``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBn1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBn2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBn3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBnReLU1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBnReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.ConvBnReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.BNReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.BNReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.LinearReLU``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBn1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBnReLU1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBn2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBnReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBn3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvBnReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.ConvReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.update_bn_stats``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.qat.freeze_bn_stats``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.BNReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.BNReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.ConvReLU1d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.ConvReLU2d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.ConvReLU3d``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.LinearReLU``
     - 
     - 
   * - ``torch.ao.nn.intrinsic.quantized.dynamic.LinearReLU``
     - 
     - 
   * - ``torch.ao.nn.qat.Conv2d``
     - 
     - 
   * - ``torch.ao.nn.qat.Conv3d``
     - 
     - 
   * - ``torch.ao.nn.qat.Linear``
     - 
     - 
   * - ``torch.ao.nn.qat.Linear.from_float``
     - 
     - 
   * - ``torch.ao.nn.qat.dynamic.Linear``
     - 
     - 
   * - ``torch.ao.nn.quantized.ReLU6``
     - 
     - 
   * - ``torch.ao.nn.quantized.Hardswish``
     - 
     - 
   * - ``torch.ao.nn.quantized.ELU``
     - 
     - 
   * - ``torch.ao.nn.quantized.LeakyReLU``
     - 
     - 
   * - ``torch.ao.nn.quantized.Sigmoid``
     - 
     - 
   * - ``torch.ao.nn.quantized.BatchNorm2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.BatchNorm3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv1d``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv1d.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv2d.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.Conv3d.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.ConvTranspose1d``
     - 
     - 
   * - ``torch.ao.nn.quantized.ConvTranspose2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.ConvTranspose3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.Embedding``
     - 
     - 
   * - ``torch.ao.nn.quantized.Embedding.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.EmbeddingBag``
     - 
     - 
   * - ``torch.ao.nn.quantized.EmbeddingBag.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.FloatFunctional``
     - 
     - 
   * - ``torch.ao.nn.quantized.FXFloatFunctional``
     - 
     - 
   * - ``torch.ao.nn.quantized.QFunctional``
     - 
     - 
   * - ``torch.ao.nn.quantized.Linear``
     - 
     - 
   * - ``torch.ao.nn.quantized.Linear.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.Linear.from_reference``
     - 
     - 
   * - ``torch.ao.nn.quantized.LayerNorm``
     - 
     - 
   * - ``torch.ao.nn.quantized.GroupNorm``
     - 
     - 
   * - ``torch.ao.nn.quantized.InstanceNorm1d``
     - 
     - 
   * - ``torch.ao.nn.quantized.InstanceNorm2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.InstanceNorm3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.avg_pool2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.avg_pool3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.adaptive_avg_pool2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.adaptive_avg_pool3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.conv1d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.conv2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.conv3d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.interpolate``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.linear``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.max_pool1d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.max_pool2d``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.celu``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.leaky_relu``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.hardtanh``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.hardswish``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.threshold``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.elu``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.hardsigmoid``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.clamp``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.upsample``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.upsample_bilinear``
     - 
     - 
   * - ``torch.ao.nn.quantized.functional.upsample_nearest``
     - 
     - 
   * - ``torch.ao.nn.quantizable.LSTM``
     - 
     - 
   * - ``torch.ao.nn.quantizable.MultiheadAttention``
     - 
     - 
   * - ``torch.ao.nn.quantizable.MultiheadAttention.dequantize``
     - 
     - 
   * - ``torch.ao.nn.quantizable.MultiheadAttention.forward``
     - 
     - 
   * - ``torch.ao.nn.quantized.dynamic.Linear``
     - 
     - 
   * - ``torch.ao.nn.quantized.dynamic.Linear.from_float``
     - 
     - 
   * - ``torch.ao.nn.quantized.dynamic.Linear.from_reference``
     - 
     - 
   * - ``torch.ao.nn.quantized.dynamic.LSTM``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.nn.quantized.dynamic.GRU``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.nn.quantized.dynamic.RNNCell``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.nn.quantized.dynamic.LSTMCell``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.nn.quantized.dynamic.GRUCell``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.compare_weights``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.get_logger_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Logger``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Logger.forward``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.ShadowLogger``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.ShadowLogger.forward``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.OutputLogger``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.OutputLogger.forward``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.forward``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.add``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.add_scalar``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.mul``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.mul_scalar``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.cat``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.Shadow.add_relu``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.prepare_model_with_stubs``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite.compare_model_stub``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite.get_matching_activations``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.ao.ns._numeric_suite.prepare_model_outputs``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite.compare_model_outputs``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.OutputLogger``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.OutputLogger.forward``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.OutputComparisonLogger``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.NSTracer``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.extract_weights``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.add_loggers``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.extract_logger_info``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.add_shadow_loggers``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.loggers_set_enabled``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.loggers_set_save_activations``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.convert_n_shadows_model``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model``
     - 
     - 
   * - ``torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model``
     - 
     - 
   * - ``torch.ao.ns.fx.utils.compute_sqnr``
     - 
     - 
   * - ``torch.ao.ns.fx.utils.compute_normalized_l2_error``
     - 
     - 
   * - ``torch.ao.ns.fx.utils.compute_cosine_similarity``
     - 
     - 
