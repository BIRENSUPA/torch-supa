# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from functools import wraps
import torch
from torch.distributed.fsdp._flat_param import FlatParamHandle, HandleTrainingState


def patch_fsdp__flat_param():
    patch_FlatParamHandle___init__()
    patch_FlatParamHandle_pre_unshard()


def patch_FlatParamHandle___init__():
    def wrapper_FlatParamHandle___init__(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            self = args[0]
            # NOTE(DIS-1523): The compute stream at the time _unshard() is called, used by
            # pre_unshard() to synchronize before _writeback_orig_params()
            self._compute_stream: torch.Stream | None = None
            return fn(*args, **kwargs)

        return wrapper

    FlatParamHandle.__init__ = wrapper_FlatParamHandle___init__(FlatParamHandle.__init__)


def patch_FlatParamHandle_pre_unshard():
    ###################
    # UNSHARD/RESHARD #
    ###################
    def pre_unshard(self) -> bool:
        """
        Return ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
        if (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._skipped_use_sharded_views
        ):
            # Since this path imposes special semantics for the unsharded flat
            # parameter (e.g. forcing full precision), use sharded views to
            # reuse the existing logic for that special handling
            self._use_sharded_views()
        ret = False
        if self._use_orig_params and not self._skip_writeback_check:
            # NOTE(DIS-1523): Wait for the compute stream since _writeback_orig_params reads
            # original parameters that may still be in use during prefetch.
            if self._compute_stream is not None:
                self._device_handle.current_stream().wait_stream(
                    self._compute_stream
                )
            ret = self._writeback_orig_params()
        if (
            self.uses_sharded_strategy
            and not self._offload_params
            and not self.needs_unshard()
        ):
            pass  # no-op
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            self._use_low_precision_shard()
            ret = True
        elif self._offload_params and self.flat_param.device != self.device:
            # NOTE: This creates a new tensor distinct from any attributes.
            self.flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        return ret

    FlatParamHandle.pre_unshard = pre_unshard
