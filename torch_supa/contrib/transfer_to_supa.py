# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import warnings
import json
import collections
import importlib.metadata
from functools import wraps
from typing import Callable, cast, Optional
from contextlib import contextmanager

import torch
from torch.utils._device import _device_constructors
from torch.nn.parameter import UninitializedTensorMixin
from torch._utils import _get_device_module
import torch_supa

try:
    from packaging.version import Version as Version
except ImportError:
    from distutils.version import LooseVersion as Version

_device_constructors()

warnings.filterwarnings(action='once')

__all__ = []

is_available = torch.cuda.is_available
cur_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cur_path, 'apis_config.json')


def _is_silence_enabled():
    return os.getenv('BRTB_TRANSFER_SILENCE', '').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def _get_function_from_string(attribute_string):
    try:
        module_path, _, attr_name = attribute_string.rpartition('.')
        module = importlib.import_module(module_path)
        return [module, attr_name]
    except Exception:
        return []


def _get_method_from_string(attribute_string):
    try:
        parts = attribute_string.split('.')
        module_path = '.'.join(parts[:-2])
        class_name = parts[-2]
        attr_name = parts[-1]
        module = getattr(importlib.import_module(module_path), class_name)
        return [module, attr_name]
    except Exception:
        return []


def _get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compare_versions(current_version, version):
    return Version(current_version) >= Version(version)


def _check_input_file_valid(file_path):
    if os.path.islink(os.path.abspath(file_path)):
        return False
    input_path = os.path.realpath(file_path)
    if not os.path.exists(input_path):
        return False
    if not os.access(input_path, os.R_OK):
        return False
    if not len(os.path.basename(input_path)) <= 200:
        return False
    if os.path.getsize(input_path) > 10 * 1024 ** 2:
        return False
    return True


def _load_json_file(file_path):
    if not _check_input_file_valid(file_path):
        return {}
    try:
        with open(file_path, 'r') as file:
            file_dict = json.load(file)
            if not isinstance(file_dict, dict):
                return {}
            return file_dict
    except json.JSONDecodeError:
        return {}


def _wrapper_libraries_func(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        patched_is_available = torch.cuda.is_available
        torch.cuda.is_available = is_available
        result = fn(*args, **kwargs)
        torch.cuda.is_available = patched_is_available
        return result

    return decorated


def _do_wrapper_libraries_func(json_dict):
    for key, value in json_dict.items():
        current_version = _get_package_version(key)
        if not current_version:
            continue
        version = value.get('version')
        apis = value.get('apis')
        if version and apis and _compare_versions(current_version, version):
            for full_name, api_type in apis.items():
                modules = None
                if api_type == 'method':
                    modules = _get_method_from_string(full_name)
                elif api_type == 'function':
                    modules = _get_function_from_string(full_name)
                if modules and getattr(modules[0], modules[1], None):
                    setattr(modules[0], modules[1], _wrapper_libraries_func(getattr(modules[0], modules[1])))

def _wrapper_cuda(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args = list(x.replace("cuda", "supa") if (isinstance(x, str) and "cuda" in x) else x for x in args)
        return fn(*args, **kwargs)

    return decorated

def _wrapper_bccl(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if type(arg) == str and 'nccl' in arg:
                    args_new[idx] = arg.replace('nccl', 'bccl')
            args = args_new
        if kwargs:
            backend = kwargs.get('backend', None)
            if type(backend) == str and 'nccl' in backend:
                kwargs['backend'] = backend.replace('nccl', 'bccl')
        return fn(*args, **kwargs)

    return decorated


def _wrapper_profiler(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if kwargs:
            activities = kwargs.get("activities", [])
            for i in range(activities.__len__()):
                if activities[i].name == "CUDA":
                    activities[i] = torch.profiler.ProfilerActivity.SUPA
        return fn(*args, **kwargs)
    decorated.export_chrome_trace = getattr(fn, 'export_chrome_trace', None)
    return decorated


def _jit_script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    return obj


def _jit_script_method(fn):
    return fn


def _patch_jit_script():
    msg = ('torch.jit.script and torch.jit.script_method will be disabled by transfer_to_supa, '
           'which currently does not support them, if you need to enable them, please do not use transfer_to_supa.')
    if not _is_silence_enabled():
        warnings.warn(msg, RuntimeWarning)
    torch.jit.script = _jit_script
    torch.jit.script_method = _jit_script_method


def _cpu_has_avx_isa():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            return " avx " in f" {f.read().lower()} "
    except OSError:
        return True


def _disable_onednn_for_cpu_without_avx_isa():
    if _cpu_has_avx_isa() or not hasattr(torch._C, "_supported_qengines"):
        return

    qengine_onednn = 3
    supported_qengines = torch._C._supported_qengines
    if qengine_onednn not in supported_qengines():
        return

    def supported_qengines_without_onednn():
        return [
            qengine
            for qengine in supported_qengines()
            if qengine != qengine_onednn
        ]

    torch._C._supported_qengines = supported_qengines_without_onednn
    if torch.backends.quantized.engine == "onednn":
        for qengine in torch.backends.quantized.supported_engines:
            if qengine not in ("none", "onednn"):
                torch.backends.quantized.engine = qengine
                break


def _patch_get_available_device_type():
    if torch.supa.is_available():
        return 'supa'
    return None

def _patch_cpp_extension():
    from torch.utils import cpp_extension
    from torch_supa.utils import cpp_extension as supa_cpp_extension

    def patched_load_inline(*args, **kwargs):
        kwargs = {
            k.replace('cuda', 'supa'): v
            for k, v in kwargs.items()
        }
        return supa_cpp_extension.load_inline(*args, **kwargs)

    cpp_extension.load_inline = patched_load_inline

def _patch_OverlappingCpuLoader_init_(self, resolve_fun: Callable, stream: Optional[torch.Stream] = None,
                                      inflight_threshhold: int = 1_000_000) -> None:
    self.resolve_fun = resolve_fun
    self.items: list[tuple[int, object]] = []
    self.inflight_threshhold = inflight_threshhold
    self.in_flight_data = 0
    self.current_items: collections.deque = collections.deque()
    self.idx = 0
    self.started = False
    self.device_type = (
        stream.device_type if stream else _patch_get_available_device_type()
    )
    self.device_module = _get_device_module(self.device_type)
    self.stream = cast(
        torch.cuda.Stream, stream or self.device_module.current_stream()
    )
    if self.stream != self.device_module.current_stream():
        self.stream.wait_stream(self.device_module.current_stream())


def _patch_cuda():
    patchs = [
        ['cuda', torch_supa.supa], ['cuda.amp', torch_supa.supa.amp],
        ['cuda.memory', torch_supa.supa.memory],
        ['cuda.random', torch_supa.supa.random],
        ['cuda.amp.autocast_mode', torch_supa.supa.amp.autocast_mode],
        ['cuda.amp.common', torch_supa.supa.amp.common],
        ['cuda.amp.grad_scaler', torch_supa.supa.amp.grad_scaler],
        ['cuda._gpu_trace', torch_supa.supa._gpu_trace],
        ['cuda.nccl', torch_supa.supa.bccl]
    ]
    torch_supa._apply_patches(patchs)
    torch.cuda.is_tf32_supported = lambda: True


def _patch_torch_backends():
    patchs = [
        ['backends.cuda', torch_supa.backends.supa]
    ]
    torch_supa._apply_patches(patchs)


def _patch_profiler():
    patchs = [
        ['profiler.ProfilerActivity.CUDA', torch_supa.profiler.ProfilerActivity.SUPA],
        ['profiler.ProfilerActivity.CPU', torch_supa.profiler.ProfilerActivity.CPU]
    ]
    torch_supa._apply_patches(patchs)


def _warning_fn(msg, rank0=True):
    if _is_silence_enabled():
        return

    is_distributed = torch.distributed.is_available() and \
        torch.distributed.is_initialized() and \
        torch.distributed.get_world_size() > 1
    env_rank = os.getenv('RANK', None)

    if rank0 and is_distributed:
        if torch.distributed.get_rank() == 0:
            warnings.warn(msg, ImportWarning)
    elif rank0 and env_rank:
        if env_rank == '0':
            warnings.warn(msg, ImportWarning)
    else:
        warnings.warn(msg, ImportWarning)


def _replace_to_method_in_allowed_methods():
    for i, method in enumerate(UninitializedTensorMixin._allowed_methods):
        if method.__name__ == "to":
            UninitializedTensorMixin._allowed_methods[i] = torch.Tensor.to
            break

def _load_torch_cuda():
    return torch.cuda

class PatchCUDA_default_generators(torch.cuda.__class__):
    @property
    def default_generators(self):
        return torch.supa.default_generators

    def __reduce__(self):
        return (_load_torch_cuda, ())

    def __reduce_ex__(self, protocol):
        return (_load_torch_cuda, ())


def property_is_cuda(self: torch.Tensor) -> bool:
    if torch_supa._C._transfer.device_type_status() is True:
        return self.device.type == "cuda"
    else:
        return self.is_supa


@contextmanager
def device_type_context():
    """context to manage device_type wrap, set to status and restore previous status after exits.
    Note:: device_type wrap works in thread scope."""
    try:
        if (pre_status := torch_supa._C._transfer.device_type_status()) is False:
            torch_supa._C._transfer.device_type(True)
        yield
    finally:
        if pre_status is False:
            torch_supa._C._transfer.device_type(pre_status)


def _init():
    _warning_fn('''
    *************************************************************************************************************
    The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.supa and torch.nn.Module.supa now..
    The torch.cuda.DoubleTensor is replaced with torch.supa.FloatTensor cause the double type is not supported now..
    The backend in torch.distributed.init_process_group set to bccl now..
    The torch.cuda.* and torch.cuda.amp.* are replaced with torch.supa.* and torch.supa.amp.* now..
    The device parameters have been replaced with supa in the function
    *************************************************************************************************************
    ''')

    # torch.cuda.*
    _patch_cuda()
    torch_supa._C._transfer.device()
    if os.getenv("BRTB_NATIVE_CI") == "1":
        import torchvision  # noqa: F401
        torch_supa._C._transfer.device_type(True)
        _disable_onednn_for_cpu_without_avx_isa()
    # transfer torch.cuda.current_stream(idx).cuda_stream
    torch.cuda.Stream.cuda_stream = torch_supa.supa.Stream.supa_stream
    torch.version.cuda = torch_supa.version.__cuda_version__

    # torch._C.*
    torch._C._has_cuda = True
    torch._C._cuda_CUDAAllocator_AllocatorState = torch_supa._C._supa_SUPAAllocator_AllocatorState
    torch._C._cuda_clearCublasWorkspaces = torch_supa._C._supa_clearSublasWorkspaces
    torch._C._cuda_getCheckpointState = torch_supa._C._supa_getCheckpointState
    torch._C._cuda_beginAllocateCurrentThreadToPool = torch_supa._C._supa_beginAllocateCurrentThreadToPool
    torch._C._cuda_setStream = torch_supa._C._supa_setStream
    torch._C._cuda_setDevice = torch_supa._C._supa_setDevice
    torch._C._cuda_synchronize = torch_supa._C._supa_synchronize
    torch._C._cuda_getCompiledVersion = torch_supa._C._supa_getCompiledVersion
    torch._C._cuda_canDeviceAccessPeer = torch_supa._C._supa_canDeviceAccessPeer
    torch._C._cuda_hasPrimaryContext = torch_supa._C._supa_hasPrimaryContext
    torch._C._cuda_endAllocateToPool = torch_supa._C._supa_endAllocateToPool
    torch._C._cuda_releasePool = torch_supa._C._supa_releasePool
    torch._C._cuda_getDeviceCount = torch_supa._C._supa_getDeviceCount
    torch._C._cuda_checkPoolLiveAllocations = torch_supa._C._supa_checkPoolLiveAllocations
    torch._C._cuda_get_conv_benchmark_empty_cache = torch_supa._C._supa_get_conv_benchmark_empty_cache
    torch._C._cudnn_set_conv_benchmark_empty_cache = torch_supa._C._sudnn_set_conv_benchmark_empty_cache
    torch._C._is_flash_attention_available = torch_supa._C._is_flash_attention_available
    torch._C._can_use_flash_attention = torch_supa._C._can_use_flash_attention
    torch._C._can_use_mem_efficient_attention = torch_supa._C._can_use_mem_efficient_attention
    torch._C._can_use_cudnn_attention = torch_supa._C._can_use_cudnn_attention
    torch._C._cuda_getCurrentRawStream = torch_supa._C._supa_getCurrentRawStream
    torch._C._cuda_customAllocator = torch_supa._C._supa_customAllocator
    torch._C._host_emptyCache = torch_supa._C._host_emptyCache
    torch._C._get_device_properties = torch_supa._C._supa_getDeviceProperties
    torch._C._graph_pool_handle = torch_supa._C._graph_pool_handle
    torch._C._cuda_ipc_collect = torch_supa._C._supa_ipc_collect

    # torch.backend.*
    _patch_torch_backends()

    # torch.profiler.*
    _patch_profiler()
    torch.profiler.profile = _wrapper_profiler(torch.profiler.profile)

    # nvtx
    torch.cuda.nvtx.range_push = torch.supa.brtx.range_push
    torch.cuda.nvtx.mark = torch.supa.brtx.mark
    torch.cuda.nvtx.range_pop = torch.supa.brtx.range_pop
    torch.cuda.nvtx.range_start = torch.supa.brtx.range_start
    torch.cuda.nvtx.range_end = torch.supa.brtx.range_end

    # torch.Tensor.*
    torch.amp.autocast_mode.autocast.__init__ = _wrapper_cuda(torch.amp.autocast_mode.autocast.__init__)

    torch.is_autocast_enabled = torch_supa._C.torch_is_autocast_enabled
    torch.set_autocast_enabled = torch_supa._C.torch_set_autocast_enabled

    torch.Tensor.cuda = torch.Tensor.supa
    torch.Tensor.is_cuda = property(property_is_cuda)
    torch.cuda.DoubleTensor = torch.supa.FloatTensor

    # torch.nn.Module.*
    torch.nn.Module.cuda = torch.nn.Module.supa

    # ipc
    torch.cuda.ipc_collect = torch_supa.supa.ipc_collect

    torch.cuda._sleep = torch_supa.supa._sleep

    # torch.distributed
    torch.distributed.init_process_group = _wrapper_bccl(torch.distributed.init_process_group)
    torch.distributed.new_group = _wrapper_bccl(torch.distributed.new_group)
    torch.distributed.new_subgroups = _wrapper_bccl(torch.distributed.new_subgroups)
    torch.distributed.new_subgroups_by_enumeration = _wrapper_bccl(torch.distributed.new_subgroups_by_enumeration)
    torch.distributed.is_nccl_available = torch_supa.distributed.is_bccl_available
    torch.distributed.ProcessGroupNCCL = torch_supa.distributed.ProcessGroupBCCL
    torch.distributed.constants.default_pg_nccl_timeout = torch_supa.distributed.constants.default_pg_bccl_timeout
    torch.distributed.device_mesh.DeviceMesh.__init__ = _wrapper_cuda(torch.distributed.device_mesh.DeviceMesh.__init__)
    torch._C._distributed_c10d._dump_nccl_trace = torch_supa._C._distributed_c10d._dump_bccl_trace
    torch._C._distributed_c10d._dump_nccl_trace_json = torch_supa._C._distributed_c10d._dump_bccl_trace_json
    torch._C._distributed_c10d._get_intra_node_comm_usage_counter = torch_supa._C._distributed_c10d._get_intra_node_comm_usage_counter

    # CUDAGraph
    torch.cuda.CUDAGraph = torch.supa.SUPAGraph

    _patch_cpp_extension()

    _patch_jit_script()

    torch._dynamo.trace_rules._disallowed_callable_ids.function_ids = None

    _do_wrapper_libraries_func(_load_json_file(config_path))

    setattr(torch._utils, '_get_available_device_type', _patch_get_available_device_type)

    _replace_to_method_in_allowed_methods()

    torch.cuda.__class__ = PatchCUDA_default_generators


_init()
