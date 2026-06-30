# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch.testing._internal.common_utils import TestCase
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import skip_but_pass_in_sandcastle_if
import torch_supa  # noqa


def sum_tensors(inq, outq):
    with torch.cuda.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put(
                (
                    tensor.sum().item(),
                    tensor.get_device(),
                    tensor.numel(),
                    tensor.storage().size(),
                )
            )

def requires_cards(count):
    return skip_but_pass_in_sandcastle_if(
        torch_supa.supa.device_count() < count,
        f"needs {count} cards at least, but found {torch_supa.supa.device_count()} cards. check env configuration."
    )

@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@requires_cards(2)
class TestSupaIPC(TestCase):

    def test_supa_small_tensors(self):
        ctx = mp.get_context("spawn")
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).cuda(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for _ in range(5):
            results.append(outq.get())
        p.join()

        for i, _tensor in enumerate(tensors):
            v, device, tensor_size, _storage_size = results[i]
            self.assertEqual(v, torch.arange(i * 5.0, (i + 1) * 5).sum())
            self.assertEqual(device, i % 2)
            self.assertEqual(tensor_size, 5)

        del _tensor
        del tensors
        torch.cuda.ipc_collect()
