.. _native-api-torch_amp:

torch.amp
================

上面的 ``torch.cuda`` 类似，只需将 ``torch.cuda.amp.xxx`` 替换为 ``torch.supa.amp.xxx``。

.. list-table::
   :header-rows: 1
   :widths: 40 15 35 10

   * - PyTorch API
     - 是否支持
     - SUPA API
     - 限制
   * - ``torch.autocast``
     - 是
     - 
     - 
   * - ``torch.cuda.amp.autocast``
     - 是
     - ``torch.supa.amp.autocast``
     - 
   * - ``torch.cuda.amp.custom_fwd``
     - 是
     - ``torch.supa.amp.custom_fwd``
     - 
   * - ``torch.cuda.amp.custom_bwd``
     - 是
     - ``torch.supa.amp.custom_bwd``
     - 
   * - ``torch.cpu.amp.autocast``
     - 是
     - ``torch.cpu.amp.autocast``
     - 
   * - ``torch.cuda.amp.GradScaler``
     - 是
     - ``torch.supa.amp.GradScaler``
     - 
   * - ``torch.cuda.amp.GradScaler.get_backoff_factor``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.get_growth_factor``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.get_growth_interval``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.get_scale``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.is_enabled``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.load_state_dict``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.scale``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.set_backoff_factor``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.set_growth_factor``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.set_growth_interval``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.state_dict``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.step``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.unscale_``
     - 否
     - 
     - 
   * - ``torch.cuda.amp.GradScaler.update``
     - 否
     - 
     - 
