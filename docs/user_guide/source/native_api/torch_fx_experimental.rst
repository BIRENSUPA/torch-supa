.. _native-api-torch_fx_experimental:

torch.fx.experimental
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.fx.experimental.sym_node.is_channels_last_contiguous_2d``
     -
     -
   * - ``torch.fx.experimental.sym_node.is_channels_last_contiguous_3d``
     -
     -
   * - ``torch.fx.experimental.sym_node.is_channels_last_strides_2d``
     -
     -
   * - ``torch.fx.experimental.sym_node.is_channels_last_strides_3d``
     -
     -
   * - ``torch.fx.experimental.sym_node.is_contiguous``
     -
     -
   * - ``torch.fx.experimental.sym_node.is_non_overlapping_and_dense_indicator``
     -
     -
   * - ``torch.fx.experimental.sym_node.method_to_operator``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_channels_last_contiguous_2d``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_channels_last_contiguous_3d``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_channels_last_strides_2d``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_channels_last_strides_3d``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_channels_last_strides_generic``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_contiguous``
     -
     -
   * - ``torch.fx.experimental.sym_node.sympy_is_contiguous_generic``
     -
     -
   * - ``torch.fx.experimental.sym_node.to_node``
     -
     -
   * - ``ShapeEnv``
     -
     -
   * - ``DimDynamic``
     -
     -
   * - ``StrictMinMaxConstraint``
     -
     -
   * - ``RelaxedUnspecConstraint``
     -
     -
   * - ``EqualityConstraint``
     -
     -
   * - ``SymbolicContext``
     -
     -
   * - ``StatelessSymbolicContext``
     -
     -
   * - ``StatefulSymbolicContext``
     -
     -
   * - ``SubclassSymbolicContext``
     -
     -
   * - ``DimConstraints``
     -
     -
   * - ``ShapeEnvSettings``
     -
     -
   * - ``ConvertIntKey``
     -
     -
   * - ``CallMethodKey``
     -
     -
   * - ``PropagateUnbackedSymInts``
     -
     -
   * - ``DivideByKey``
     -
     -
   * - ``InnerTensorKey``
     -
     -
   * - ``Specialization``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_concrete_int``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_concrete_bool``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_concrete_float``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.has_free_symbols``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_or_true``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_or_false``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_size_oblivious``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.sym_and``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.sym_eq``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.sym_or``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.constrain_range``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.constrain_unify``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.statically_known_true``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.statically_known_false``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.has_static_value``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.lru_cache``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.check_consistent``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.rebind_unbacked``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.resolve_unbacked_bindings``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_accessor_node``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.create_contiguous``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.error``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.eval_guards``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.eval_is_non_overlapping_and_dense``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.find_symbol_binding_fx_nodes``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.free_symbols``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.free_unbacked_symbols``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.fx_placeholder_targets``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.fx_placeholder_vals``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_bool``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_float``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_int``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guard_scalar``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.has_guarding_hint``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.has_symbolic_sizes_strides``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_nested_int``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_symbol_binding_fx_node``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.is_symbolic``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.optimization_hint``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.expect_true``
     -
     -
   * - ``torch.fx.experimental.symbolic_shapes.log_lru_cache_stats``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.make_fx``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.handle_sym_dispatch``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.get_proxy_mode``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.maybe_enable_thunkify``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.maybe_disable_thunkify``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.thunkify``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.track_tensor``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.track_tensor_tree``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.decompose``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.disable_autocast_cache``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.disable_proxy_modes_tracing``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.dispatch_trace``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.extract_val``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.fake_signature``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.fetch_object_proxy``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.fetch_sym_proxy``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.has_proxy_slot``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.is_sym_node``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.maybe_handle_decomp``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.proxy_call``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.set_meta``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.set_original_aten_op``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.set_proxy_slot``
     -
     -
   * - ``torch.fx.experimental.proxy_tensor.snapshot_fake``
     -
     -
   * - ``torch.fx.experimental.optimization.extract_subgraph``
     -
     -
   * - ``torch.fx.experimental.optimization.matches_module_pattern``
     -
     -
   * - ``torch.fx.experimental.optimization.modules_to_mkldnn``
     -
     -
   * - ``torch.fx.experimental.optimization.optimize_for_inference``
     -
     -
   * - ``torch.fx.experimental.optimization.remove_dropout``
     -
     -
   * - ``torch.fx.experimental.optimization.replace_node_module``
     -
     -
   * - ``torch.fx.experimental.optimization.reset_modules``
     -
     -
   * - ``torch.fx.experimental.optimization.use_mkl_length``
     -
     -
   * - ``torch.fx.experimental.recording.record_shapeenv_event``
     -
     -
   * - ``torch.fx.experimental.recording.replay_shape_env_events``
     -
     -
   * - ``torch.fx.experimental.recording.shape_env_check_state_equal``
     -
     -
   * - ``torch.fx.experimental.unification.core.reify``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.utils.typename``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.utils.expand_tuples``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.utils.groupby``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.utils.raises``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.utils.reverse_dict``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.assoc``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.assoc_in``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.dissoc``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.first``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.groupby``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.keyfilter``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.keymap``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.merge``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.merge_with``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.update_in``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.valfilter``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.valmap``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.itemfilter``
     -
     -
   * - ``torch.fx.experimental.unification.unification_tools.itemmap``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_algebraic_expression``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints_trace_time``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_dimension``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_to_z3``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_var``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.transform_to_z3.evaluate_conditional_with_constraints``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint.is_algebraic_expression``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint.is_bool_expr``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint.is_dim``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.adaptive_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.assert_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.batchnorm_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.bmm_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.embedding_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.embedding_inference_rule_functional``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.eq_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.equality_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.expand_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.full_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.gt_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.lt_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.masked_fill_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.neq_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.tensor_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.torch_dim_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.torch_linear_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.type_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.view_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.register_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.transpose_inference_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_generator.range_check``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.apply_padding``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.calc_last_two_dims``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.create_equality_constraints_for_broadcasting``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.is_target_div_by_dim``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.no_broadcast_dim_with_index``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.register_transformation_rule``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.transform_constraint``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.transform_get_item``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.transform_get_item_tensor``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.transform_index_select``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.transform_transpose``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.valid_index``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.valid_index_tensor``
     -
     -
   * - ``torch.fx.experimental.migrate_gradual_types.constraint_transformation.is_dim_div_by_target``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.adaptiveavgpool2d_check``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.adaptiveavgpool2d_inference_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.all_eq``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.bn2d_inference_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.calculate_out_dimension``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.conv_refinement_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.conv_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.element_wise_eq``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.expand_to_tensor_dim``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.first_two_eq``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.register_algebraic_expressions_inference_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.register_inference_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.register_refinement_rule``
     -
     -
   * - ``torch.fx.experimental.graph_gradual_typechecker.transpose_inference_rule``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.embedding_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.functional_relu_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.nn_layernorm_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.proxys_to_metas``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.symbolic_trace``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.torch_abs_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.torch_nn_relu_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.torch_relu_override``
     -
     -
   * - ``torch.fx.experimental.meta_tracer.torch_where_override``
     -
     -
   * - ``torch.fx.experimental.accelerator_partitioner.check_dependency``
     -
     -
   * - ``torch.fx.experimental.accelerator_partitioner.combine_two_partitions``
     -
     -
   * - ``torch.fx.experimental.accelerator_partitioner.reorganize_partitions``
     -
     -
   * - ``torch.fx.experimental.accelerator_partitioner.reset_partition_device``
     -
     -
   * - ``torch.fx.experimental.accelerator_partitioner.set_parents_and_children``
     -
     -
   * - ``torch.fx.experimental.debug.set_trace``
     -
     -
   * - ``torch.fx.experimental.merge_matmul.are_nodes_independent``
     -
     -
   * - ``torch.fx.experimental.merge_matmul.may_depend_on``
     -
     -
   * - ``torch.fx.experimental.merge_matmul.merge_matmul``
     -
     -
   * - ``torch.fx.experimental.unification.match.edge``
     -
     -
   * - ``torch.fx.experimental.unification.match.match``
     -
     -
   * - ``torch.fx.experimental.unification.match.ordering``
     -
     -
   * - ``torch.fx.experimental.unification.match.supercedes``
     -
     -
   * - ``torch.fx.experimental.unification.more.reify_object``
     -
     -
   * - ``torch.fx.experimental.unification.more.unifiable``
     -
     -
   * - ``torch.fx.experimental.unification.more.unify_object``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.ambiguities``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.ambiguous``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.consistent``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.edge``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.ordering``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.super_signature``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.conflict.supercedes``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.core.dispatch``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.core.ismethod``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.ambiguity_warn``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.halt_ordering``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.restart_ordering``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.source``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.str_signature``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.variadic_signature_matches``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.variadic_signature_matches_iter``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.dispatcher.warning_text``
     -
     -
   * - ``torch.fx.experimental.unification.multipledispatch.variadic.isvariadic``
     -
     -
   * - ``torch.fx.experimental.unification.utils.freeze``
     -
     -
   * - ``torch.fx.experimental.unification.utils.hashable``
     -
     -
   * - ``torch.fx.experimental.unification.utils.raises``
     -
     -
   * - ``torch.fx.experimental.unification.utils.reverse_dict``
     -
     -
   * - ``torch.fx.experimental.unification.utils.transitive_get``
     -
     -
   * - ``torch.fx.experimental.unification.utils.xfail``
     -
     -
   * - ``torch.fx.experimental.unification.variable.var``
     -
     -
   * - ``torch.fx.experimental.unification.variable.variables``
     -
     -
   * - ``torch.fx.experimental.unification.variable.vars``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.check_for_type_equality``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.infer_symbolic_types``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.infer_symbolic_types_single_pass``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.substitute_all_types``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.substitute_solution_one_type``
     -
     -
   * - ``torch.fx.experimental.unify_refinements.unify_eq``
     -
     -
   * - ``torch.fx.experimental.validator.bisect``
     -
     -
   * - ``torch.fx.experimental.validator.translation_validation_enabled``
     -
     -
   * - ``torch.fx.experimental.validator.translation_validation_timeout``
     -
     -
   * - ``torch.fx.experimental.validator.z3op``
     -
     -
   * - ``torch.fx.experimental.validator.z3str``
     -
     -
