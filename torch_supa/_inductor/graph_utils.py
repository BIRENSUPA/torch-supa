# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Dict, Optional
import torch
from torch_supa.utils import torch_version_ge

def patch_graph_device_check():
    from torch._inductor.cudagraph_utils import _get_use_stack_trace, format_default_skip_message

    def check_multiple_devices_or_any_cpu_nodes(
        device_node_mapping: Dict[torch.device, torch.fx.Node]
    ) -> Optional[str]:
        device_node_mapping.pop(torch.device("meta"), None)

        # graph partition supports splitting on cpu op. So we can ignore cpu nodes.
        if torch_version_ge(2, 9, 0):
            from torch._inductor.utils import is_using_cudagraph_partition
            if is_using_cudagraph_partition():
                device_node_mapping.pop(torch.device("cpu"), None)
        elif torch_version_ge(2, 8, 0):
            if torch._inductor.config.graph_partition:
                device_node_mapping.pop(torch.device("cpu"), None)

        if cpu_node := device_node_mapping.get(torch.device("cpu")):
            msg = f"cpu device ({cpu_node.name})"
            if stack_trace := _get_use_stack_trace(cpu_node):
                return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

            return format_default_skip_message(msg)

        # patch device type
        if (
            len(device_node_mapping) == 1
            and next(iter(device_node_mapping.keys())).type in ("supa", "cuda")
        ):
            return None

        keys_repr = (repr(key) for key in device_node_mapping.keys())
        return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")

    torch._inductor.cudagraph_utils.check_multiple_devices_or_any_cpu_nodes = check_multiple_devices_or_any_cpu_nodes
