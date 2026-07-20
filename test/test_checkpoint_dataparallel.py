# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import warnings
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torch_supa  # noqa: F401  enables DataParallel scatter/gather patching
import torch_supa.contrib.transfer_to_supa  # noqa: F401


@pytest.mark.sanity
@pytest.mark.regression
class TestCheckpointDataParallel:
    def test_checkpointing_without_reentrant_dataparallel(self):
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.

        Migrated from test/native_ci/test_autograd.py::TestCheckpoint::
        test_checkpointing_without_reentrant_dataparallel.
        Exercises the SUPA _supa_scatter/_supa_gather bindings that
        DataParallel relies on.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                return self.linear(inp)

        a = torch.randn(2, 2, device="supa", requires_grad=True)

        model = LinearModule()
        model = model.to("supa")

        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None

        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        torch.testing.assert_close(b_grad, c_grad, rtol=0, atol=0)

    def test_dataparallel_saved_tensors_hooks(self):
        def pack(tensor):
            warnings.warn("pack")
            return tensor

        class Model(torch.nn.Module):
            def forward(self, x):
                with warnings.catch_warnings(record=True) as caught:
                    x * x
                    if torch.supa.device_count() >= 2:
                        assert len(caught) == 0
                    else:
                        assert len(caught) > 0

        x = torch.ones(5, 5, requires_grad=True)
        model = torch.nn.DataParallel(Model())

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model(x)
            with warnings.catch_warnings(record=True) as caught:
                x * x
                assert len(caught) > 0
