# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import no_type_check
import torch
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._flat_param import FlatParamHandle


def patch_fsdp__runtime_utils():
    patch__unshard()


def patch__unshard():
    @no_type_check
    def _unshard(
        state: _FSDPState,
        handle: FlatParamHandle,
        unshard_stream: torch.Stream,
        pre_unshard_stream: torch.Stream,
    ) -> None:
        """
        Unshards the handles in ``handles``. If the handles are in
        :meth:`summon_full_params` and are using mixed precision, then they are
        forced to full precision.

        Postcondition: handle's ``FlatParameter`` 's data is the padded
        unsharded flat parameter on the compute device.
        """
        if not handle:
            return
        # NOTE(DIS-1523): Set the compute stream to the current stream
        handle._compute_stream = state._device_handle.current_stream()
        with state._device_handle.stream(pre_unshard_stream):
            ran_pre_unshard = handle.pre_unshard()
        if ran_pre_unshard:
            unshard_stream.wait_stream(pre_unshard_stream)
        if state.limit_all_gathers:
            event = state._free_event_queue.dequeue_if_needed()
            if event:
                with torch.profiler.record_function(
                    "FullyShardedDataParallel.rate_limiter"
                ):
                    event.synchronize()
        with state._device_handle.stream(unshard_stream):
            handle.unshard()
            handle.post_unshard()

    torch.distributed.fsdp._runtime_utils._unshard = _unshard
