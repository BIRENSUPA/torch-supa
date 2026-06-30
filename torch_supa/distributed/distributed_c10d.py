# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.

import warnings
import io
from typing import Any, Optional
import timedelta

import torch
from torch.torch_version import TorchVersion
try:
    from torch.monitor import _WaitCounter
except (ImportError, AttributeError):
    _WaitCounter = None

from torch.distributed import GroupMember, Backend
from torch.distributed.distributed_c10d import (
    _update_default_pg, logger, _world, _pickler, _unpickler, _get_default_group,
    _get_split_source, is_initialized, _process_group_color, _rank_not_in_group, 
    is_gloo_available, _time_logger, BackendConfig, _process_group_name, _check_valid_timeout)
from torch._C._distributed_c10d import (
    _DistributedBackendOptions,
    _register_process_group,
    _unregister_all_process_groups,
    _unregister_process_group,
    BarrierOptions,
    DebugLevel,
    get_debug_level,
    PrefixStore,
    ProcessGroup,
    ProcessGroupGloo,
)
from torch.distributed.distributed_c10d import default_pg_nccl_timeout, default_pg_timeout
from torch.distributed.distributed_c10d import (
    _new_process_group_helper as origin_new_process_group_helper,
    batch_isend_irecv as origin_batch_isend_irecv
)
import torch_supa
from torch_supa.utils import torch_version_le

from .constants import *  # noqa

original_barrier = ProcessGroup.barrier

try:
    from torch_supa._C._distributed_c10d import ProcessGroupBCCL, _hash_tensors, _dump_bccl_trace_json, _dump_bccl_trace
    _BCCL_AVAILABLE = True
except ImportError as e:
    print(e)
    _BCCL_AVAILABLE = False


def is_bccl_available():
    return _BCCL_AVAILABLE

def dump_bccl_trace_json(includeCollectives: Optional[bool] = None, onlyActive: Optional[bool] = None) -> bytes:
    return _dump_bccl_trace_json(includeCollectives, onlyActive)


def dump_bccl_trace(includeCollectives: Optional[bool] = None, includeStackTraces: Optional[bool] = None, onlyActive: Optional[bool] = None) -> bytes:
    return _dump_bccl_trace(includeCollectives, includeStackTraces, onlyActive)


def patch_distributed():
    if not _BCCL_AVAILABLE:
        return

    def create_fn(dist_backend_opts, pg_options=None):
        if pg_options is not None:
            assert isinstance(
                pg_options, ProcessGroupBCCL.Options
            ), "Expected pg_options argument to be of type ProcessGroupBCCL.Options"
            if pg_options._timeout != dist_backend_opts.timeout:
                warnings.warn(
                    "backend_options._timeout was specified, "
                    "but timeout kwarg has a default value that will always override it. "
                )
        else:
            pg_options = ProcessGroupBCCL.Options()
            pg_options.is_high_priority_stream = False
            # only set timeout when pg_options is None
            # Warning: split_group will always pass a pg_options, avoid setting timeout
            pg_options._timeout = dist_backend_opts.timeout

        if is_initialized() and _get_default_group().bound_device_id:
            split_from = _get_split_source(_get_default_group())
        else:
            split_from = None

        if split_from:
            pg_options.split_from = split_from
            pg_options.split_color = _process_group_color(
                pg_options.global_ranks_in_group
            )
        pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
        pg_options.group_name = dist_backend_opts.group_id
        return ProcessGroupBCCL(
            dist_backend_opts.store, dist_backend_opts.group_rank, dist_backend_opts.group_size, pg_options
        )

    torch.distributed.Backend.register_backend("bccl", create_fn, extended_api=True, devices=["supa"])

    def _get_bccl_backend_by_pg(pg: ProcessGroup) -> Backend:
        backend = None
        try:
            backend = pg._get_backend(torch.device("supa"))
        except RuntimeError:
            pass

        return backend if isinstance(backend, ProcessGroupBCCL) else None

    def _shutdown_backend(pg: ProcessGroup):
        """
        Try to shut down the backend of a process group.
        Currently, only ProcessGroupBCCL backend is supported.
        No op for other backends.
        """
        backend = _get_bccl_backend_by_pg(pg)
        if backend:
            # explictly call shutdown to ensure that BCCL resources are released
            if pg._has_hooks():
                pg._wait_for_pending_works()
            backend._shutdown()

    torch.distributed.distributed_c10d._shutdown_backend = _shutdown_backend

    def _abort_backend(pg: ProcessGroup):
        """
        Abort the backend of a process group.
        Currently, only ProcessGroupBCCL backend is supported.
        No op for other backends.
        """
        backend = _get_bccl_backend_by_pg(pg)
        if backend:
            backend.abort()

    def _abort_process_group(group: Optional[ProcessGroup] = None):
        """
        Abort a given process group. If group.WORLD (i.e. `None`) is given, all
        process groups including the default one will be aborted.

        Args:
            group (ProcessGroup, optional): The process group to be aborted.

        .. note:: this API is experimental and currently only works with the BCCL
            backend.

        .. note:: this API should be used with `BCCL_ASYNC_ERROR_HANDLING` `TORCH_NCCL_ASYNC_ERROR_HANDLING`
            turned off (i.e. set to 0). Otherwise, ProcessGroupBCCL's watchdog may
            automatically handle errors or timeouts for you including aborting the
            ProcessGroup.
        """

        if group == GroupMember.NON_GROUP_MEMBER:
            return

        pg = group or GroupMember.WORLD

        assert pg is not None
        if _world.pg_map.get(pg, None) is None:
            raise ValueError("Invalid process group specified or has been destroyed.")

        backend = _get_bccl_backend_by_pg(pg)
        if not backend:
            logger.warning(
                "`abort_process_group` currently only has implementation for ProcessGroupBCCL; "
                "however, no BCCL backend is found. This call will be a no-op."
            )
            return

        if group == GroupMember.WORLD:
            # Abort all backends within a bcclGroupStart|End semantic.
            # This ensures that different BCCL communicators' abort calls won't
            # deadlock each other.
            # For details, please see: https://github.com/pytorch/pytorch/issues/119797
            backend._group_start()
            for pg_to_abort in sorted(
                _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
            ):
                _abort_backend(pg_to_abort)

            backend._group_end()

            _update_default_pg(None)
            _world.pg_map.clear()
            _world.pg_names.clear()
            _world.pg_group_ranks.clear()
            _world.pg_backend_config.clear()
            _world.pg_to_tag.clear()
            _world.tags_to_pg.clear()
            _world.pg_coalesce_state.clear()
            _unregister_all_process_groups()

            # when process group doesn't have an explicit name (only WORLD (default)
            # process group can have an explicit name), we use global _world.group_count
            # to generate the name. We need to reset the counter on destruction to
            # allow consistent value to be generated when we re-create process
            # groups after some trainers recover from failure
            #
            # We only reset this when WORLD is being destroyed because if this
            # process group is in good state, we aren't dealing with failures.
            _world.group_count = 0
        else:
            _abort_backend(pg)
            del _world.pg_map[pg]
            del _world.pg_names[pg]
            del _world.pg_group_ranks[pg]
            del _world.pg_backend_config[pg]
            if pg in _world.pg_coalesce_state.keys():
                warnings.warn(
                    "Some coalesced collectives haven't been launched when "
                    "ProcessGroup is aborted. They will be cleaned."
                )
                del _world.pg_coalesce_state[pg]

            tag = _world.pg_to_tag.get(pg)
            del _world.pg_to_tag[pg]
            if tag is not None:
                try:
                    _world.tags_to_pg[tag].remove(pg)
                    if tag.startswith("ptd:"):
                        _world.tags_to_pg[""].remove(pg)
                except Exception:
                    pass
            _unregister_process_group(pg.group_name)

    torch.distributed.distributed_c10d._abort_process_group = _abort_process_group

    def _object_to_tensor(obj, device, group: Optional[ProcessGroup]):
        if _WaitCounter is not None and TorchVersion(torch.__version__) >= (2, 5, 0):
            counter = _WaitCounter("pytorch.wait_counter.c10d._tensor_to_object").guard()
        else:
            from contextlib import nullcontext
            counter = nullcontext()

        with counter:
            f = io.BytesIO()
            _pickler(f).dump(obj)
            byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
            # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
            # Otherwise, it will casue 100X slowdown.
            # See: https://github.com/pytorch/pytorch/issues/65696
            byte_tensor = torch.ByteTensor(byte_storage).to(device)
            if get_debug_level() == DebugLevel.DETAIL and _BCCL_AVAILABLE:
                backend = _get_bccl_backend_by_pg(group)
                if backend:
                    hash = _hash_tensors([byte_tensor])
                    logger.warning(
                        "_object_to_tensor size: %s hash value: %s",
                        byte_tensor.numel(),
                        hash,
                    )
            local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
            return byte_tensor, local_size

    torch.distributed.distributed_c10d._object_to_tensor = _object_to_tensor

    def _tensor_to_object(tensor, tensor_size, group):
        if _WaitCounter is not None and TorchVersion(torch.__version__) >= (2, 5, 0):
            counter = _WaitCounter("pytorch.wait_counter.c10d._tensor_to_object").guard()
        else:
            from contextlib import nullcontext
            counter = nullcontext()

        with counter:
            if get_debug_level() == DebugLevel.DETAIL and is_bccl_available():
                backend = _get_bccl_backend_by_pg(group)
                if backend:
                    hash = _hash_tensors([tensor])
                    logger.warning(
                        "_tensor_to_object size: %s hash value: %s", tensor.numel(), hash
                    )
            tensor = tensor.cpu()
            buf = tensor.numpy().tobytes()[:tensor_size]
            return _unpickler(io.BytesIO(buf)).load()

    torch.distributed.distributed_c10d._tensor_to_object = _tensor_to_object

    def _set_pg_timeout(timeout: timedelta, group: Optional[ProcessGroup] = None) -> None:
        if group is None:
            group = _get_default_group()
        if _rank_not_in_group(group):
            raise ValueError("Invalid process group specified")
        assert isinstance(group, ProcessGroup)
        devices = group._device_types
        backends = set()
        if torch.device("cpu") in devices and is_gloo_available():
            backend = group._get_backend(torch.device("cpu"))
            if isinstance(backend, ProcessGroupGloo):
                backends.add(backend)
        if torch.device("supa") in devices:
            backend = group._get_backend(torch.device("supa"))
            if is_bccl_available() and isinstance(backend, ProcessGroupBCCL):
                backends.add(backend)  # type: ignore[arg-type]
            elif is_gloo_available() and isinstance(backend, ProcessGroupGloo):
                backends.add(backend)  # type: ignore[arg-type]
        if len(backends) == 0:
            warnings.warn("Set timeout is now only supported for either bccl or gloo.")
        for backend in backends:
            backend._set_default_timeout(timeout)

    torch.distributed.distributed_c10d._set_pg_timeout = _set_pg_timeout

    def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
        for pg in _world.pg_map.keys():
            devices = pg._device_types
            if torch.device("supa") in devices:
                backend = pg._get_backend(torch.device("supa"))
                if is_bccl_available() and isinstance(backend, ProcessGroupBCCL):
                    backend._add_ephemeral_timeout(timeout)

    torch.distributed.distributed_c10d._add_ephemeral_timeout_for_all_pgs = _add_ephemeral_timeout_for_all_pgs

    def _get_process_group_uid(pg: ProcessGroup) -> int:
        backend = None
        try:
            backend = pg._get_backend(torch.device("cuda"))
        except RuntimeError:
            pass
        if is_bccl_available() and isinstance(backend, ProcessGroupBCCL):
            return backend.uid
        return -1

    torch.distributed.distributed_c10d._get_process_group_uid = _get_process_group_uid

    def _get_default_timeout(backend: Backend) -> timedelta:
        # see note on nccl vs other backend timeout (constants.py)
        if backend == Backend.NCCL:
            if not isinstance(default_pg_nccl_timeout, timedelta):
                # TODO moco benchmark on CPU initializes pgnccl backend today, triggered this assert in CI before it was
                # changed to be a warning.  We should fix the moco model.
                warnings.warn(
                    "Attempted to get default timeout for nccl backend, but NCCL support is not compiled"
                )
                return default_pg_timeout
            return default_pg_nccl_timeout
        else:
            return default_pg_bccl_timeout

    torch.distributed.distributed_c10d._get_default_timeout = _get_default_timeout

    def _bccl_get_sequence_number_for_group(self):
        origin_get_sequence_number_for_group = ProcessGroup._get_sequence_number_for_group
        backend = torch.distributed.get_backend_config(self)
        if backend == "bccl" or backend == "supa:bccl":
            return self._get_backend(torch.device("supa"))._get_sequence_number_for_group()
        else:
            return origin_get_sequence_number_for_group(self)

    torch._C._distributed_c10d.ProcessGroup._get_sequence_number_for_group = _bccl_get_sequence_number_for_group

    # in torch <= 2.6, _new_process_group_helper will check if the device.type equals to cuda
    def _new_process_group_helper(*args, **kwargs):
        status = torch_supa._C._transfer.device_type_status()
        torch_supa._C._transfer.device_type(True)
        args_list = list(args)
        backend = args_list[3]
        if str(backend) == Backend.UNDEFINED:
            # patch the backend
            args_list[3] = Backend("bccl")
            args = tuple(args_list)
        else:
            backend_config = BackendConfig(backend)
            device_backend_map = backend_config.get_device_backend_map()
            if device_backend_map.get("cuda") == Backend.NCCL:
                device_backend_map = {
                    ("supa" if device == "cuda" else device):
                    (Backend("bccl") if device == "cuda" else device_backend)
                    for device, device_backend in device_backend_map.items()
                }
                args_list[3] = Backend(",".join(
                    f"{device}:{device_backend}"
                    for device, device_backend in device_backend_map.items()
                ))
                args = tuple(args_list)

        if isinstance(kwargs.get("backend_options", None), ProcessGroupBCCL.Options):
            # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
            # we need to set the timeout here to avoid conflicts with split_group
            kwargs["backend_options"]._timeout = kwargs["timeout"]
        result = origin_new_process_group_helper(*args, **kwargs)
        torch_supa._C._transfer.device_type(status)
        return result

    torch.distributed.distributed_c10d._new_process_group_helper = _new_process_group_helper

    if torch_version_le(2, 6, 1):
        # in torch <= 2.6, native c10d use device type to decide coalescing 
        def batch_isend_irecv(*args, **kwargs):
            status = torch_supa._C._transfer.device_type_status()
            torch_supa._C._transfer.device_type(True)
            result = origin_batch_isend_irecv(*args, **kwargs)
            torch_supa._C._transfer.device_type(status)
            return result

        torch.distributed.batch_isend_irecv = batch_isend_irecv

    if torch_version_le(2, 7, 1):
        # in torch <= 2.7, split_group does not support custom backend
        @_time_logger
        def split_group(
            parent_pg: Optional[ProcessGroup] = None,
            split_ranks: Optional[list] = None,
            timeout: Optional[timedelta] = None,
            pg_options: Optional[Any] = None,
            group_desc: Optional[str] = None,
        ) -> Optional[ProcessGroup]:
            # check inputs
            if split_ranks is None:
                raise ValueError("split_ranks cannot be None")

            default_pg = _get_default_group()
            device_id = default_pg.bound_device_id
            if not device_id:
                raise RuntimeError(
                    "No device associated with the default pg, not safe to split any process groups"
                )
            _default_backend, default_store = _world.pg_map[default_pg]
            global_rank = default_pg.rank()
            global_world_size = default_pg.size()

            if not parent_pg:
                parent_pg = default_pg
            if parent_pg not in _world.pg_group_ranks:
                raise ValueError(f"Group {parent_pg} is not registered")

            parent_global_to_group_ranks = _world.pg_group_ranks[parent_pg]
            parent_group_to_global_ranks = {
                group_rank: global_rank
                for global_rank, group_rank in parent_global_to_group_ranks.items()
            }

            if global_rank not in parent_global_to_group_ranks:
                raise ValueError(
                    f"Global rank {global_rank} is not part of the parent group {parent_pg}"
                )

            parent_group_rank = parent_global_to_group_ranks[global_rank]
            parent_backend = parent_pg._get_backend(torch.device("cuda"))

            # if the parent backend does not support splitting, raise error
            # currently this API only support NCCL backend
            if not parent_backend or not parent_backend.supports_splitting:
                raise RuntimeError(
                    "No backend for the parent process group or its backend does not support splitting"
                )

            # set the group_desc before the color or no_cloor split
            group_desc = (
                f"{parent_pg.group_desc}:split:{parent_backend.comm_split_count()}"  # type: ignore[attr-defined]
                if group_desc is None
                else group_desc
            )

            parent_backend_str, _ = _world.pg_map[parent_pg]
            # same type of backend as the parent process group
            backend = Backend(parent_backend_str)
            backend_config = BackendConfig(backend)

            if pg_options is None:
                # default pg_options same as the parent process group
                pg_options = parent_backend.options

            # this timeout defaulting/validation is used for all the new_groups/new_subgroups variants,
            # which may just pass their timeout value (or None)
            if timeout is None:
                timeout = _get_default_timeout(backend)
            _check_valid_timeout(timeout)

            # find my group of ranks and my group local rank in split_ranks
            my_group = None
            group_rank = -1

            for split_group in split_ranks:
                if len(split_group) == 0:
                    raise ValueError("the split group cannot be empty")
                if len(split_group) > global_world_size:
                    raise ValueError(
                        "the split group's size should be less or equal to the world_size set by init_process_group"
                    )
                if len(split_group) != len(set(split_group)):
                    raise ValueError("the split group cannot have duplicate ranks")
                split_group = sorted(split_group)
                if parent_group_rank in split_group:
                    my_group = split_group
                    group_rank = split_group.index(parent_group_rank)
                    break
            # if my rank does not belong to any sub group,
            # no_color split should be called
            if my_group is None or group_rank == -1:
                parent_backend.perform_nocolor_split(device_id)  # type: ignore[attr-defined]
                return None

            group_name = _process_group_name(my_group, use_hashed_name=False)
            global_ranks_in_my_group = [parent_group_to_global_ranks[rank] for rank in my_group]

            prefix_store = PrefixStore(f"{group_name}/", default_store)
            # We register the backend after initializing and timeout is set in pg_options.
            pg: ProcessGroup = ProcessGroup(
                prefix_store,
                group_rank,
                len(my_group),
            )
            pg.bound_device_id = device_id  # type: ignore[union-attr]
            pg_options._timeout = timeout  # type: ignore[union-attr]
            pg_options.split_from = parent_backend  # type: ignore[union-attr]
            pg_options.split_color = _process_group_color(my_group)  # type: ignore[union-attr]
            pg_options.global_ranks_in_group = global_ranks_in_my_group  # type: ignore[union-attr]
            pg_options.group_name = group_name  # type: ignore[union-attr]

            assert parent_backend_str.upper() in Backend._plugins, (
                f"Unknown c10d backend type {parent_backend_str.upper()}"
            )
            backend_plugin = Backend._plugins[parent_backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM
            if not extended_api:
                backend_class = creator_fn(prefix_store, group_rank, len(my_group), timeout)
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = len(my_group)
                backend_class = creator_fn(dist_backend_opts, pg_options)

            pg._set_default_backend(backend_type)
            backend_class._set_sequence_number_for_group()

            pg._register_backend(torch.device("cuda"), backend_type, backend_class)

            # set group_name and group_desc to backend
            assert group_name is not None
            assert group_desc is not None
            pg._set_group_name(group_name)
            pg._set_group_desc(group_desc)

            # always eagerly initialize the backend in split_group
            eager_backend = pg._get_backend(device_id)
            eager_backend.eager_connect_single_device(device_id)

            # update global state
            _world.pg_map[pg] = (backend, prefix_store)
            _world.pg_names[pg] = group_name
            _register_process_group(group_name, pg)
            _world.pg_backend_config[pg] = str(backend_config)
            pg_tag = f"ptd:{group_name}"
            _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
            _world.pg_to_tag[pg] = pg_tag

            # Create the global rank to group rank mapping
            _world.pg_group_ranks[pg] = {
                global_rank: group_rank
                for group_rank, global_rank in enumerate(global_ranks_in_my_group)
            }

            return pg

        torch.distributed.split_group = split_group

    def patch_barrier(self, opts=None):
        if opts is None and self._get_backend_name() == "custom":
            device = torch._C._get_accelerator()
            opts = BarrierOptions()
            if getattr(self, "bound_device_id", None) is not None:
                opts.device = self.bound_device_id
            else:
                opts.device = device
                warnings.warn( 
                    "No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. "
                )

        return original_barrier(self, opts)

    torch.distributed.distributed_c10d.ProcessGroup.barrier = patch_barrier
