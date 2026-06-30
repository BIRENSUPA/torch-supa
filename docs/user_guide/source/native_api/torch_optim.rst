.. _native-api-torch_optim:

torch.optim
=================

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.optim.Optimizer``
     - 
     - 
   * - ``Optimizer.add_param_group``
     - 是
     - 
   * - ``Optimizer.load_state_dict``
     - 是
     - 
   * - ``Optimizer.state_dict``
     - 是
     - 
   * - ``Optimizer.step``
     - 是
     - 
   * - ``Optimizer.zero_grad``
     - 是
     - 
   * - ``torch.optim.Adadelta``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.Adadelta.add_param_group``
     - 是
     - 
   * - ``torch.optim.Adadelta.load_state_dict``
     - 是
     - 
   * - ``torch.optim.Adadelta.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.Adadelta.state_dict``
     - 是
     - 
   * - ``torch.optim.Adadelta.step``
     - 
     - 
   * - ``torch.optim.Adadelta.zero_grad``
     - 是
     - 
   * - ``torch.optim.Adagrad``
     - 
     - 
   * - ``torch.optim.Adagrad.add_param_group``
     - 是
     - 
   * - ``torch.optim.Adagrad.load_state_dict``
     - 是
     - 
   * - ``torch.optim.Adagrad.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.Adagrad.state_dict``
     - 是
     - 
   * - ``torch.optim.Adagrad.step``
     - 
     - 
   * - ``torch.optim.Adagrad.zero_grad``
     - 是
     - 
   * - ``torch.optim.Adam``
     - 
     - 
   * - ``torch.optim.Adam.add_param_group``
     - 是
     - 
   * - ``torch.optim.Adam.load_state_dict``
     - 是
     - 
   * - ``torch.optim.Adam.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adam.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adam.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adam.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adam.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.Adam.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.Adam.state_dict``
     - 是
     - 
   * - ``torch.optim.Adam.step``
     - 
     - 
   * - ``torch.optim.Adam.zero_grad``
     - 是
     - 
   * - ``torch.optim.AdamW``
     - 
     - 
   * - ``torch.optim.AdamW.add_param_group``
     - 
     - 
   * - ``torch.optim.AdamW.load_state_dict``
     - 
     - 
   * - ``torch.optim.AdamW.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.AdamW.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.AdamW.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.AdamW.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.AdamW.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.AdamW.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.AdamW.state_dict``
     - 
     - 
   * - ``torch.optim.AdamW.step``
     - 
     - 
   * - ``torch.optim.AdamW.zero_grad``
     - 
     - 
   * - ``torch.optim.SparseAdam``
     - 
     - 
   * - ``torch.optim.SparseAdam.add_param_group``
     - 
     - 
   * - ``torch.optim.SparseAdam.load_state_dict``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.SparseAdam.state_dict``
     - 
     - 
   * - ``torch.optim.SparseAdam.step``
     - 
     - 
   * - ``torch.optim.SparseAdam.zero_grad``
     - 
     - 
   * - ``torch.optim.Adamax``
     - 
     - 
   * - ``torch.optim.Adamax.add_param_group``
     - 
     - 
   * - ``torch.optim.Adamax.load_state_dict``
     - 
     - 
   * - ``torch.optim.Adamax.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adamax.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adamax.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Adamax.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Adamax.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.Adamax.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.Adamax.state_dict``
     - 
     - 
   * - ``torch.optim.Adamax.step``
     - 
     - 
   * - ``torch.optim.Adamax.zero_grad``
     - 
     - 
   * - ``torch.optim.ASGD``
     - 
     - 
   * - ``torch.optim.ASGD.add_param_group``
     - 
     - 
   * - ``torch.optim.ASGD.load_state_dict``
     - 
     - 
   * - ``torch.optim.ASGD.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.ASGD.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.ASGD.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.ASGD.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.ASGD.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.ASGD.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.ASGD.state_dict``
     - 
     - 
   * - ``torch.optim.ASGD.step``
     - 
     - 
   * - ``torch.optim.ASGD.zero_grad``
     - 
     - 
   * - ``torch.optim.LBFGS``
     - 
     - 
   * - ``torch.optim.LBFGS.add_param_group``
     - 
     - 
   * - ``torch.optim.LBFGS.load_state_dict``
     - 
     - 
   * - ``torch.optim.LBFGS.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.LBFGS.state_dict``
     - 
     - 
   * - ``torch.optim.LBFGS.step``
     - 
     - 
   * - ``torch.optim.LBFGS.zero_grad``
     - 
     - 
   * - ``torch.optim.NAdam``
     - 
     - 
   * - ``torch.optim.NAdam.add_param_group``
     - 
     - 
   * - ``torch.optim.NAdam.load_state_dict``
     - 
     - 
   * - ``torch.optim.NAdam.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.NAdam.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.NAdam.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.NAdam.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.NAdam.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.NAdam.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.NAdam.state_dict``
     - 
     - 
   * - ``torch.optim.NAdam.step``
     - 
     - 
   * - ``torch.optim.NAdam.zero_grad``
     - 
     - 
   * - ``torch.optim.RAdam``
     - 
     - 
   * - ``torch.optim.RAdam.add_param_group``
     - 
     - 
   * - ``torch.optim.RAdam.load_state_dict``
     - 
     - 
   * - ``torch.optim.RAdam.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.RAdam.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.RAdam.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.RAdam.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.RAdam.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.RAdam.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.RAdam.state_dict``
     - 
     - 
   * - ``torch.optim.RAdam.step``
     - 
     - 
   * - ``torch.optim.RAdam.zero_grad``
     - 
     - 
   * - ``torch.optim.RMSprop``
     - 
     - 
   * - ``torch.optim.RMSprop.add_param_group``
     - 
     - 
   * - ``torch.optim.RMSprop.load_state_dict``
     - 
     - 
   * - ``torch.optim.RMSprop.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.RMSprop.state_dict``
     - 
     - 
   * - ``torch.optim.RMSprop.step``
     - 
     - 
   * - ``torch.optim.RMSprop.zero_grad``
     - 
     - 
   * - ``torch.optim.Rprop``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.Rprop.add_param_group``
     - 
     - 
   * - ``torch.optim.Rprop.load_state_dict``
     - 
     - 
   * - ``torch.optim.Rprop.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Rprop.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Rprop.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.Rprop.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.Rprop.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.Rprop.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.Rprop.state_dict``
     - 
     - 
   * - ``torch.optim.Rprop.step``
     - 
     - 
   * - ``torch.optim.Rprop.zero_grad``
     - 
     - 
   * - ``torch.optim.SGD``
     - 
     - 
   * - ``torch.optim.SGD.add_param_group``
     - 
     - 
   * - ``torch.optim.SGD.load_state_dict``
     - 
     - 
   * - ``torch.optim.SGD.register_load_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.SGD.register_load_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.SGD.register_state_dict_post_hook``
     - 
     - 
   * - ``torch.optim.SGD.register_state_dict_pre_hook``
     - 
     - 
   * - ``torch.optim.SGD.register_step_post_hook``
     - 
     - 
   * - ``torch.optim.SGD.register_step_pre_hook``
     - 
     - 
   * - ``torch.optim.SGD.state_dict``
     - 
     - 
   * - ``torch.optim.SGD.step``
     - 
     - 
   * - ``torch.optim.SGD.zero_grad``
     - 
     - 
   * - ``torch.optim.lr_scheduler.LambdaLR``
     - 是
     - 
   * - ``torch.optim.lr_scheduler.LambdaLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.LambdaLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.LambdaLR.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.LambdaLR.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.MultiplicativeLR``
     - 
     - 
   * - ``torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.MultiplicativeLR.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.MultiplicativeLR.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.StepLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.StepLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.StepLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.StepLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.StepLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.MultiStepLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.MultiStepLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.MultiStepLR.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.MultiStepLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.MultiStepLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ConstantLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ConstantLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ConstantLR.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ConstantLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ConstantLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.LinearLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.LinearLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.LinearLR.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.LinearLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.LinearLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ExponentialLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ExponentialLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ExponentialLR.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ExponentialLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ExponentialLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.PolynomialLR``
     - 
     - 
   * - ``torch.optim.lr_scheduler.PolynomialLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.PolynomialLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.PolynomialLR.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.PolynomialLR.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.CosineAnnealingLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingLR.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.ChainedScheduler``
     - 
     - 
   * - ``torch.optim.lr_scheduler.ChainedScheduler.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.ChainedScheduler.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.ChainedScheduler.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.ChainedScheduler.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.SequentialLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.SequentialLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.SequentialLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.SequentialLR.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.SequentialLR.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.ReduceLROnPlateau``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CyclicLR``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CyclicLR.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CyclicLR.get_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CyclicLR.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.OneCycleLR``
     - 是
     - 
   * - ``torch.optim.lr_scheduler.OneCycleLR.get_last_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.OneCycleLR.load_state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.OneCycleLR.print_lr``
     - 
     - 
   * - ``torch.optim.lr_scheduler.OneCycleLR.state_dict``
     - 
     - 
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.print_lr``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict``
     - 是
     - 仅支持 fp16、fp32、int64、bool
   * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step``
     - 是
     - 仅支持 fp16、fp32、int64、bool
