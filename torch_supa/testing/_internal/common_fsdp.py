# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.testing._internal.common_distributed as common_distributed
from torch.testing._internal.common_distributed import MultiThreadedTestCase, MultiProcessTestCase
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, FSDPTest, FSDPTestModel, DEVICEInitMode, TransformerWithSharedParams

DEVICE_TYPE="supa"  # noqa

def patch_common_fsdp():
    patch_FSDPTest_globals()
    patch_FSDPTestMultiThread_setUp()
    patch_FSDPTest_setUp()
    patch_TransformerWithSharedParams___init__()
    patch_TransformerWithSharedParams_get_input()
    patch_sm_is_or_higher_than()


def patch_FSDPTest_globals():
    torch.testing._internal.common_fsdp.FSDPTest._run.__globals__['TEST_CUDA'] = True
    torch.testing._internal.common_fsdp.FSDPTest._run.__globals__['DISTRIBUTED_BACKEND'] = "nccl"
    torch.testing._internal.common_fsdp.FSDPTest._run.__globals__['DEVICE_COUNT'] = torch.supa.device_count()
    torch.testing._internal.common_fsdp.FSDPTest._test_fsdp_parity.__globals__['DEVICE_TYPE'] = DEVICE_TYPE
    torch.testing._internal.common_fsdp.FSDPTest._train_for_several_steps.__globals__['DEVICE_TYPE'] = DEVICE_TYPE


def patch_sm_is_or_higher_than():
    orig_sm_is_or_higher_than = common_distributed.sm_is_or_higher_than

    def sm_is_or_higher_than(device: torch.device, major: int, minor: int) -> bool:
        if device.type == DEVICE_TYPE:
            return torch.supa.is_bf16_supported()
        return orig_sm_is_or_higher_than(device, major, minor)

    common_distributed.sm_is_or_higher_than = sm_is_or_higher_than


def patch_FSDPTestMultiThread_setUp():
    def setUp(self):
        MultiThreadedTestCase.setUp(self)

        os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "3600000"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "36666"
        os.environ["BCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["BCCL_TRACE_BUFFER_SIZE"] = "1024"
        os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"

        self._spawn_threads()

    FSDPTestMultiThread.setUp = setUp


def patch_FSDPTest_setUp():
    def setUp(self):
        MultiProcessTestCase.setUp(self)
        # Set NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["NCCL_DESYNC_DEBUG"] = "0"

        os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "3600000"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "36666"
        os.environ["BCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["BCCL_TRACE_BUFFER_SIZE"] = "1024"

        os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"

        self._spawn_processes()

    FSDPTest.setUp = setUp


def patch_TransformerWithSharedParams___init__():
    def TransformerWithSharedParams___init__(
        self,
        group: dist.ProcessGroup,
        cuda_init_mode: DEVICEInitMode,
        add_bn: bool,
        deterministic: bool,
    ):
        FSDPTestModel.__init__(self)
        self.rank = group.rank()
        self.world_size = group.size()
        if deterministic:
            torch.manual_seed(0)
        d_vocab = 4
        d_model = 4

        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=4,
            dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)

        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer(
            "vocab_bias", self.embed_tokens.weight.new_ones((d_model,))
        )
        self.register_buffer(
            "long_buffer",
            torch.zeros_like(self.vocab_bias, dtype=torch.long),
        )  # type: ignore[arg-type]

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        if cuda_init_mode == DEVICEInitMode.DEVICE_BEFORE:
            self = self.to(DEVICE_TYPE)
        if deterministic:
            self.eval()

    TransformerWithSharedParams.__init__ = TransformerWithSharedParams___init__

def patch_TransformerWithSharedParams_get_input():
    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(4, device=device).view(2, self.bs)  # T x B
        tgt = torch.arange(self.bs * 2, device=device).view(2, self.bs)  # T x B
        return (src, tgt)

    TransformerWithSharedParams.get_input = get_input
