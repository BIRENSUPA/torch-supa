# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# mypy: allow-untyped-defs
import gzip
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing_extensions import Self
from warnings import warn

import torch
import torch_supa

if not torch_supa._C._profiler_init():
    raise RuntimeError("proflier initialization failed")

from . import profiler_kineto as prof
from torch._C._profiler import (
    _ExperimentalConfig,
)
from torch_supa._C._profiler import (
    _supported_activities,
    ProfilerActivity,
    _add_metadata_json,
)
import torch_supa
from torch.autograd import ProfilerState
from torch.profiler.profiler import ProfilerAction, _default_schedule_fn, _ITraceObserver
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline

__all__ = [
    "supported_activities",
    "ProfilerActivity",
    "profile",
]
PROFILER_STEP_NAME = "ProfilerStep"

class _NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for numpy types (np.int, np.float, np.array etc.)
    Returns default encoder if numpy is not available
    """

    def default(self, obj):
        """Encode NumPy types to JSON"""
        try:
            import numpy as np
        except ImportError:
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)
        

def supported_activities():
    return _supported_activities()

class _KinetoProfile:
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, SUPA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.SUPA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.SUPA
            or (when available) ProfilerActivity.XPU.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles


    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """

    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        use_supa_simple: bool = True,
        experimental_config: Optional[_ExperimentalConfig] = None,
        execution_trace_observer: Optional[_ITraceObserver] = None,
        acc_events: bool = False,
        custom_trace_id_callback: Optional[Callable[[], str]] = None,
    ):
        self.activities = set(activities) if activities else supported_activities()
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_supa_simple = use_supa_simple
        self.experimental_config = experimental_config
        self.execution_trace_observer = execution_trace_observer
        self.acc_events = acc_events
        self.custom_trace_id_callback = custom_trace_id_callback
        self.profiler: Optional[prof.profile] = None
        self.mem_tl: Optional[MemoryProfileTimeline] = None
        self.use_device = None
        if ProfilerActivity.SUPA in self.activities:
            self.use_device = "supa"

        # user-defined metadata to be amended to the trace
        self.preset_metadata: Dict[str, str] = {}

    def start(self):
        self.prepare_trace()
        self.start_trace()

    def stop(self):
        self.stop_trace()

    def prepare_trace(self):
        if (self.profiler is None) or (not self.acc_events):
            self.profiler = prof.profile(
                use_cpu=(ProfilerActivity.CPU in self.activities),
                use_device=self.use_device,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
                use_supa_simple=self.use_supa_simple,
                use_kineto=True,
                experimental_config=self.experimental_config,
                acc_events=self.acc_events,
                custom_trace_id_callback=self.custom_trace_id_callback,
            )
        self.profiler._prepare_trace()

    def start_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.start()
        assert self.profiler is not None
        self.profiler._start_trace()

        if self.profile_memory:
            self.add_metadata_json("profile_memory", "1")
        if self.with_stack:
            self.add_metadata_json("with_stack", "1")
        if self.record_shapes:
            self.add_metadata_json("record_shapes", "1")
        if self.with_modules:
            self.add_metadata_json("with_modules", "1")
        if self.with_flops:
            self.add_metadata_json("with_flops", "1")

        if True: # kineto_available
            dist_info = self._get_distributed_info()
            if dist_info:
                
                self.add_metadata_json(
                    "distributedInfo", json.dumps(dist_info, cls=_NumpyEncoder)
                )

            # Insert the preset user metadata to the trace
            for k, v in self.preset_metadata.items():
                self.add_metadata_json(k, v)

    def stop_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.stop()
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        assert self.profiler
        if path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+b", suffix=".json", delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
            with open(fp.name, "rb") as fin:
                with gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        else:
            return self.profiler.export_chrome_trace(path)

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_supa_time_total"
        """
        assert self.profiler
        return self.profiler.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enable: bool, activities: Iterable[ProfilerActivity]
    ):
        """Toggle collection of activities on/off at any point of collection. Currently supports toggling Torch Ops
        (CPU) and SUPA activity supported in Kineto

        Args:
            activities (iterable): list of activity groups to use in profiling, supported values:
                ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.SUPA``
        Examples:

        .. code-block:: python

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.SUPA,
                ]
            ) as p:
                code_to_profile_0()
                // turn off collection of all SUPA activity
                p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.SUPA])
                code_to_profile_1()
                // turn on collection of all SUPA activity
                p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.SUPA])
                code_to_profile_2()
            print(p.key_averages().table(
                sort_by="self_supa_time_total", row_limit=-1))
        """
        if not self.profiler:
            return
        self.profiler.toggle_collection_dynamic(enable, activities)

    def key_averages(
        self, group_by_input_shape: bool = False, group_by_stack_n: int = 0
    ):
        """Averages events, grouping them by operator name and (optionally) input shapes and
        stack.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        assert self.profiler
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        # torch.autograd._add_metadata_json(key, wrapped_value)
        assert self.profiler
        wrapped_value = value.replace('"', '\\"')
        return self.profiler.add_metadata(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        # torch.autograd._add_metadata_json(key, value)
        assert self.profiler
        return self.profiler.add_metadata(key, value)

    def preset_metadata_json(self, key: str, value: str):
        """
        Preset a user defined metadata when the profiler is not started
        and added into the trace file later.
        Metadata is in the format of a string key and a valid json value
        """
        self.preset_metadata[key] = value

    def _get_distributed_info(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        backend = dist.get_backend()
        dist_info = {
            "backend": backend,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pg_count": dist.get_pg_count(),
            "pg_config": dist.distributed_c10d._get_all_pg_configs(),
        }
        return dist_info

    def _memory_profile(self) -> MemoryProfile:
        required = ("record_shapes", "profile_memory", "with_stack")
        missing = [f"{i}=True" for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f"{', '.join(missing)} required for memory profiling.")

        assert self.profiler is not None and self.profiler.kineto_results is not None
        return MemoryProfile(self.profiler.kineto_results)

    def export_memory_timeline(self, path: str, device: Optional[str] = None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None and self.use_device and self.use_device != "supa":
            device = self.use_device + ":0"

        if device is None:
            device = "supa:0" if torch_supa.supa.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
            fp.close()
            if path.endswith("raw.json.gz"):
                self.mem_tl.export_memory_timeline_raw(fp.name, device)
            else:
                self.mem_tl.export_memory_timeline(fp.name, device)
            with open(fp.name) as fin:
                with gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
        else:
            self.mem_tl.export_memory_timeline(path, device)


class profile(_KinetoProfile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, SUPA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.SUPA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.SUPA
            or (when available) ProfilerActivity.XPU.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        use_supa (bool):
            .. deprecated:: 1.8.1
                use ``activities`` instead.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.


    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.SUPA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_supa_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_supa_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.SUPA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()

    The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

    .. code-block:: python

        with torch.profiler.profile(
            ...
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            ),
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                p.step()

    You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
    Note: One can also pass any object satisfying the _ITraceObserver interface.
    """

    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        execution_trace_observer: Optional[_ITraceObserver] = None,
        acc_events: bool = False,
        # deprecated:
        use_supa: Optional[bool] = None,
        custom_trace_id_callback: Optional[Callable[[], str]] = None,
    ):
        activities_set = set(activities) if activities else supported_activities()
        if use_supa is not None:
            warn(
                "`use_supa` is deprecated, use `activities` argument instead",
                FutureWarning,
                stacklevel=2,
            )
            if use_supa:
                activities_set.add(ProfilerActivity.SUPA)
            elif ProfilerActivity.SUPA in activities_set:
                activities_set.remove(ProfilerActivity.SUPA)
        assert len(activities_set) > 0, "No valid profiler activities found"

        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer,
            acc_events=acc_events,
            custom_trace_id_callback=custom_trace_id_callback,
        )

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self.step_rec_fn: Optional[prof.record_function] = None

        self.action_map: Dict[
            Tuple[ProfilerAction, Optional[ProfilerAction]], List[Any]
        ] = {
            # key is (prev_action, current_action), value is action list corresponding to the state pair.
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_trace],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self.start_trace, self.stop_trace],
            (ProfilerAction.RECORD, None): [self.stop_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                self.stop_trace,
                self._trace_ready,
            ],
        }
        # Start tracking increments to profiler step, this will be used
        # by Kineto
        prof.KinetoStepTracker.init_step_count(PROFILER_STEP_NAME)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()

    def start(self):
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)

    def step(self):
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)

        self._transit_action(prev_action, self.current_action)
        prof.KinetoStepTracker.increment_step(PROFILER_STEP_NAME)

        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def set_custom_trace_id_callback(self, callback):
        """
        Sets a callback to be called when a new trace ID is generated.
        """
        self.custom_trace_id_callback = callback

    def get_trace_id(self):
        """
        Returns the current trace ID.
        """
        if self.profiler is None:
            return None
        return self.profiler.trace_id

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action):
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _stats(self) -> Optional[prof._ProfilerStats]:
        if self.profiler is None:
            return None
        return self.profiler._stats
