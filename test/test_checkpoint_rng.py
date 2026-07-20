# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import torch_supa  # noqa: F401  enables transfer_to_supa patching
import torch_supa.contrib.transfer_to_supa  # noqa: F401


@pytest.mark.regression
def test_checkpoint_rng():
    """Regression test for checkpoint RNG state preservation on SUPA.

    Migrated from pytorch test_utils.py::TestCheckpoint::test_checkpoint_rng_cuda.

    Fix: exposed SUPA initialization through the CUDA-like device module RNG
    state API (torch.supa.get_rng_state/set_rng_state, mapped from
    torch.cuda.* via transfer_to_supa) so checkpoint with use_reentrant=True
    preserves and restores the RNG state, matching CUDA behavior.
    """
    for _ in range(5):
        inp = torch.randn(20000, device="supa").requires_grad_()
        phase1 = torch.nn.Dropout()
        phase2 = torch.nn.Dropout()

        def run_fn(input):
            return phase2(input)

        state = torch.cuda.get_rng_state()

        out = phase1(inp)
        out = checkpoint(run_fn, out, use_reentrant=True)
        out.sum().backward()
        grad_with_checkpointing = inp.grad

        torch.cuda.set_rng_state(state)

        inp.grad = None

        out = phase1(inp)
        out = run_fn(out)
        out.sum().backward()
        grad_no_checkpointing = inp.grad

        assert torch.equal(grad_with_checkpointing, grad_no_checkpointing)
