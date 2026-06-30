# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch._ops
from torch._C import DispatchKey
from torch._inductor.decomposition import remove_decompositions, decompositions, register_decomposition as inductor_register_decomposition
from torch._decomp import register_decomposition, decomposition_table
from torch.utils import _pytree as pytree
from torch_supa.utils import torch_version_lt

aten = torch.ops.aten

DECOMPOSITION_OVERLOAD_OP = [
    aten.scaled_dot_product_attention.default
]


def _should_skip_cia_decompose(func, args, kwargs):
    flat_args, _ = pytree.tree_flatten((args, kwargs))
    has_backend_registration = False
    for a in flat_args:
        if isinstance(a, torch.Tensor):
            backend_key = torch._C._parse_dispatch_key(
                torch._C._dispatch_key_for_device(a.device.type)
            )
            assert backend_key is not None

            has_backend_registration = torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), backend_key
            )
            # in theory we should take all backend keys and take the highest priority one
            # to properly mimic the dispatcher,
            # this just grabs the first tensor and takes its device key
            break
    return has_backend_registration


def patch_opoverload_decompose_for_privateuse1():
    # if we both register privateuse1 and compositeimplicitautograd key,
    # we need to ensure torch compile would dispatch the func to out 
    # decomposition rules, otherwise it will use compositeimplicitautograd.
    # the patch is needed for torch version < 2.10. Official torch fixed this
    # issue using autograd_would_have_decomposed after 2.10
    original_decompose = torch._ops.OpOverload.decompose

    def _patched_decompose(self, *args, **kwargs):
        if (
            self.has_kernel_for_dispatch_key(DispatchKey.PrivateUse1)
            and _should_skip_cia_decompose(self, args, kwargs)
            and self in DECOMPOSITION_OVERLOAD_OP 
            and self in decomposition_table
            and torch_version_lt(2, 10, 0)
        ):
            return decomposition_table[self](*args, **kwargs)
        return original_decompose(self, *args, **kwargs)

    torch._ops.OpOverload.decompose = _patched_decompose


def _register_supa_inductor_decompositions():
    remove_decompositions(decompositions, DECOMPOSITION_OVERLOAD_OP)

    @register_decomposition(aten.scaled_dot_product_attention)
    @inductor_register_decomposition(aten.scaled_dot_product_attention)
    def scaled_dot_product_attention_decomp(*args, **kwargs):
        """
        Customize decomposition using privateuse1 key. We cannot use cia since
        default cia will use sdpa in at::native namespace
        """
        dk = DispatchKey.PrivateUse1
        return aten.scaled_dot_product_attention.default._op_dk(
            dk, *args, **kwargs
        )

    # Also register in torch._decomp.decompositions so torch_decomp_decompositions(func)
    # recognizes this override; otherwise FakeTensor mode will still reapply the default decomposition(CIA).
    for fn in DECOMPOSITION_OVERLOAD_OP:
        decomposition_table[fn].__module__ = "torch._decomp.decompositions"
        setattr(torch._decomp.decompositions, decomposition_table[fn].__name__, decomposition_table[fn])
