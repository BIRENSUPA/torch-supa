.. _native-api-torch_onnx:

torch.onnx
============

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.onnx.dynamo_export``
     - 是
     - 
   * - ``torch.onnx.ExportOptions``
     - 是
     - 
   * - ``torch.onnx.enable_fake_mode``
     - 
     - 
   * - ``torch.onnx.ExportOutput``
     - 是
     - 
   * - ``torch.onnx.ExportOutput.adapt_torch_inputs_to_onnx``
     - 是
     - 
   * - ``torch.onnx.ExportOutput.adapt_torch_outputs_to_onnx``
     - 是
     - 
   * - ``torch.onnx.ExportOutput.diagnostic_context``
     - 
     - 
   * - ``torch.onnx.ExportOutput.fake_context``
     - 
     - 
   * - ``torch.onnx.ExportOutput.model_proto``
     - 
     - 
   * - ``torch.onnx.ExportOutput.save``
     - 是
     - 
   * - ``torch.onnx.ExportOutput.save_diagnostics``
     - 
     - 
   * - ``torch.onnx.ExportOutputSerializer``
     - 是
     - 
   * - ``torch.onnx.ExportOutputSerializer.serialize``
     - 是
     - 
   * - ``torch.onnx.OnnxExporterError``
     - 
     - 
   * - ``torch.onnx.OnnxRegistry``
     - 
     - 
   * - ``torch.onnx.OnnxRegistry.get_op_functions``
     - 
     - 
   * - ``torch.onnx.OnnxRegistry.is_registered_op``
     - 
     - 
   * - ``torch.onnx.OnnxRegistry.opset_version``
     - 
     - 
   * - ``torch.onnx.OnnxRegistry.register_op``
     - 
     - 
   * - ``torch.onnx.DiagnosticOptions``
     - 
     - 
   * - ``torch.onnx.is_onnxrt_backend_supported``
     - 
     - 
   * - ``torch.onnx.export``
     - 是
     - 
   * - ``torch.onnx.export_to_pretty_string``
     - 是
     - 
   * - ``torch.onnx.register_custom_op_symbolic``
     - 是
     - 
   * - ``torch.onnx.unregister_custom_op_symbolic``
     - 是
     - 
   * - ``torch.onnx.select_model_mode_for_export``
     - 是
     - 
   * - ``torch.onnx.is_in_onnx_export``
     - 是
     - 
   * - ``torch.onnx.enable_log``
     - 是
     - 
   * - ``torch.onnx.disable_log``
     - 是
     - 
   * - ``torch.onnx.verification.find_mismatch``
     - 是
     - 
   * - ``torch.onnx.JitScalarType``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.dtype``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.from_dtype``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.from_value``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.onnx_compatible``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.onnx_type``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.scalar_name``
     - 是
     - 
   * - ``torch.onnx.JitScalarType.torch_name``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.all_mismatch_leaf_graph_info``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.clear``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.essential_node_count``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.essential_node_kinds``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.export_repro``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.find_mismatch``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.find_partition``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.has_mismatch``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.pretty_print_mismatch``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.pretty_print_tree``
     - 是
     - 
   * - ``torch.onnx.verification.GraphInfo.verify_export``
     - 是
     - 
   * - ``torch.onnx.verification.VerificationOptions``
     - 是
     - 
