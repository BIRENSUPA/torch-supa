.. _native-api-torch_autograd:

torch.autograd
==============

.. list-table::
   :header-rows: 1
   :widths: 60 20 20

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.autograd.backward``
     - 是
     -
   * - ``torch.autograd.grad``
     - 是
     -
   * - ``dual_level``
     - 是
     -
   * - ``torch.autograd.forward_ad.make_dual``
     - 是
     -
   * - ``torch.autograd.forward_ad.unpack_dual``
     - 是
     -
   * - ``torch.autograd.forward_ad.enter_dual_level``
     - 是
     -
   * - ``torch.autograd.forward_ad.exit_dual_level``
     - 是
     -
   * - ``UnpackedDualTensor``
     - 是
     -
   * - ``torch.autograd.functional.jacobian``
     - 是
     -
   * - ``torch.autograd.functional.hessian``
     - 是
     -
   * - ``torch.autograd.functional.vjp``
     - 是
     -
   * - ``torch.autograd.functional.jvp``
     - 是
     -
   * - ``torch.autograd.functional.vhp``
     - 是
     -
   * - ``torch.autograd.functional.hvp``
     - 是
     -
   * - ``torch.autograd.Function.forward``
     - 是
     -
   * - ``torch.autograd.Function.backward``
     - 是
     -
   * - ``torch.autograd.Function.jvp``
     - 是
     -
   * - ``torch.autograd.Function.vmap``
     - 是
     -
   * - ``torch.autograd.function.FunctionCtx.mark_dirty``
     - 是
     -
   * - ``torch.autograd.function.FunctionCtx.mark_non_differentiable``
     - 是
     -
   * - ``torch.autograd.function.FunctionCtx.save_for_backward``
     - 是
     -
   * - ``torch.autograd.function.FunctionCtx.set_materialize_grads``
     - 是
     -
   * - ``torch.autograd.function.once_differentiable``
     - 是
     -
   * - ``BackwardCFunction``
     - 是
     -
   * - ``InplaceFunction``
     - 是
     -
   * - ``NestedIOFunction``
     - 是
     -
   * - ``torch.autograd.gradcheck.gradcheck``
     - 是
     -
   * - ``torch.autograd.gradcheck.gradgradcheck``
     - 是
     -
   * - ``torch.autograd.gradcheck.GradcheckError``
     - 是
     -
   * - ``torch.autograd.profiler.profile.export_chrome_trace``
     - 是
     -
   * - ``torch.autograd.profiler.profile.key_averages``
     - 是
     -
   * - ``torch.autograd.profiler.profile.self_cpu_time_total``
     - 是
     -
   * - ``torch.autograd.profiler.profile.total_average``
     - 是
     -
   * - ``torch.autograd.profiler.parse_nvprof_trace``
     - 是
     -
   * - ``EnforceUnique``
     - 是
     -
   * - ``KinetoStepTracker``
     - 是
     -
   * - ``record_function``
     - 是
     -
   * - ``Interval``
     - 是
     -
   * - ``Kernel``
     - 是
     -
   * - ``MemRecordsAcc``
     - 是
     -
   * - ``StringTable``
     - 是
     -
   * - ``torch.autograd.profiler.load_nvprof``
     - 是
     -
   * - ``set_multithreading_enabled``
     - 是
     -
   * - ``torch.autograd.graph.Node.name``
     - 是
     -
   * - ``torch.autograd.graph.Node.metadata``
     - 是
     -
   * - ``torch.autograd.graph.Node.next_functions``
     - 是
     -
   * - ``torch.autograd.graph.Node.register_hook``
     - 是
     -
   * - ``torch.autograd.graph.Node.register_prehook``
     - 是
     -
   * - ``torch.autograd.graph.increment_version``
     - 是
     -
