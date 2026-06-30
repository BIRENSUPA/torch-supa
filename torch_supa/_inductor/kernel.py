# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
from torch_supa.utils import transfer_device_type, torch_version_ge


def patch_tuned_kernel():
    # torch.compile uses tuned_mm for gemm backend, which will check device type.
    # Patch the device for torch.mm and torch.bmm with out_dtype args after v2.10
    if torch_version_ge(2, 10, 0):
        from torch._inductor.kernel.mm import tuned_mm as origin_tuned_mm
        from torch._inductor.kernel.bmm import tuned_bmm as origin_tuned_bmm
        from torch._inductor.lowering import register_lowering

        aten = torch.ops.aten

        def _restore_supa_device(ir_node):
            data = getattr(ir_node, "data", None)
            if data is not None:
                _restore_supa_device(data)

            layout = getattr(ir_node, "layout", None)
            if layout is not None and layout.device.type == "cuda":
                layout.device = torch.device("supa", layout.device.index)

            inputs = getattr(ir_node, "inputs", None)
            if inputs is not None:
                for inp in inputs:
                    _restore_supa_device(inp)

            return ir_node

        def _restore_supa_graph_device_info(device):
            # Avoid mixed supa+cuda devices after transfer_device_type during mm/bmm lowering.
            if device.type != "supa":
                return

            from torch._inductor.virtualized import V

            V.graph.device_types.discard("cuda")
            for graph_device in list(V.graph.device_node_mapping.keys()):
                if graph_device.type == "cuda":
                    V.graph.device_node_mapping.pop(graph_device, None)

        @register_lowering(aten.mm.dtype, type_promotion_kind=None)
        def tuned_mm(*args, **dwargs):
            @transfer_device_type
            def call_orign_mm():
                return origin_tuned_mm(*args, **dwargs)

            result = call_orign_mm()
            result = _restore_supa_device(result)
            _restore_supa_graph_device_info(result.get_device())
            return result

        @register_lowering(aten.bmm.dtype, type_promotion_kind=None)
        def tuned_bmm(*args, **dwargs):
            @transfer_device_type
            def call_orign_bmm():
                return origin_tuned_bmm(*args, **dwargs)

            result = call_orign_bmm()
            result = _restore_supa_device(result)
            _restore_supa_graph_device_info(result.get_device())
            return result


def patch_template_heuristic_device_type():
    if torch_version_ge(2, 9, 0):
        import torch._inductor.template_heuristics.registry as registry

        origin_get_template_heuristic = registry.get_template_heuristic

        def get_template_heuristic(template_name, device_type, op_name):
            if device_type == "supa":
                device_type = "cuda"
            return origin_get_template_heuristic(
                template_name, device_type, op_name
            )

        registry.get_template_heuristic = get_template_heuristic
