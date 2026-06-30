# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import torch
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor


def patch_api():
    patch_ShardedTensor_supa()


def patch_ShardedTensor_supa():
    def supa(
        self,
        device=None,
        non_blocking=False,
        memory_format=torch.preserve_format,
        process_group=None,
    ) -> ShardedTensor:
        if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
        ):
            raise RuntimeError(
                "Only `torch.contiguous_format` or "
                "`torch.preserve_format` is supported!"
            )

        if device is not None:
            device = torch.device(device) if isinstance(device, str) else device
            assert (
                isinstance(device, torch.device)
                and device.index == torch.supa.current_device()
            ), (
                """Only device without device id (e.g. "cpu" or "supa") is expected for ShardedTensor!"""
            )

        current_device = torch.device("supa", torch.supa.current_device())
        list_shards: list[Shard] = []
        for shard in self._local_shards:
            supa_tensor = shard.tensor.supa(
                device=current_device,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )
            metadata = copy.deepcopy(shard.metadata)
            metadata.placement._device = current_device

            list_shards.append(Shard(supa_tensor, metadata))

        st_meta = copy.deepcopy(self.metadata())
        for meta in st_meta.shards_metadata:
            if meta.placement.device().type != "supa":
                meta.placement._device = current_device

        pg = self._process_group if process_group is None else process_group
        st_supa = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )
        return st_supa

    ShardedTensor.supa = supa
