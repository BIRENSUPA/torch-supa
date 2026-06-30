# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import uuid
from collections import defaultdict
import json
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Optional
from warnings import warn

import torch
from torch._C._autograd import DeviceType
from torch._C._profiler import (
    _ExperimentalConfig,
)
from torch.autograd import ProfilerState
from torch.autograd.profiler import record_function
import torch_supa
from torch_supa._C._profiler import (
    _kineto_step,
    _ProfilerResult,
    _supported_activities,
    ProfilerActivity,
    ProfilerConfig,
    _prepare_profiler,
    _disable_profiler,
    _enable_profiler,
    _add_metadata_json,
    _toggle_collection_dynamic
)
from .profiler_util import ( 
    MEMORY_EVENT_NAME, 
    OUT_OF_MEMORY_EVENT_NAME,
    _filter_name,
    _rewrite_name,
    _filter_stack_entry,
    EventList,
    FunctionEvent,
    MemRecordsAcc
) 

__all__ = ["profile", "record_function", "KinetoStepTracker"]

# To capture triton cpu op, we wish to set this attribute.
# check the source code torch/_inductor/runtime/triton_heuristics.py
# Could we find a better way to patch _is_profiler_enabled ?
def _run_on_profiler_start():
    torch.autograd.profiler._set_is_profiler_enabled(True)


def _run_on_profiler_stop():
    torch.autograd.profiler._set_is_profiler_enabled(False)


@dataclass
class _ProfilerStats:
    "Profiler timing and stats used by developers to catch issues/regressions"
    profiling_window_duration_sec: float = 0
    number_of_events: int = 0
    profiler_prepare_call_duration_us: int = 0
    profiler_enable_call_duration_us: int = 0
    profiler_disable_call_duration_us: int = 0
    parse_kineto_call_duration_us: int = 0
    function_events_build_tree_call_duration_us: int = 0


class profile:
    """Context manager that manages autograd profiler state and holds a summary of results.

    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.

        use_supa (bool, optional): Enables timing of SUPA events as well
            using the supaEvent API. (will be deprecated)

        use_device (str, optional): Enables timing of device events.
            Adds approximately 4us of overhead to each tensor operation when use supa.
            The valid devices options are 'supa', 'xpu', 'mtia' and 'privateuseone'.

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        with_flops (bool, optional): If with_flops is set, the profiler will estimate
            the FLOPs (floating point operations) value using the operator's input shape.
            This allows one to estimate the hardware performance. Currently,
            this option only works for the matrix multiplication and 2D convolution operators.

        profile_memory (bool, optional): track tensor memory allocation/deallocation.

        with_stack (bool, optional): record source information (file and line number) for the ops.

        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.

        use_kineto (bool, optional): experimental, enable profiling with Kineto profiler.

        use_cpu (bool, optional): profile CPU events; setting to ``False`` requires
            ``use_kineto=True`` and can be used to lower the overhead for GPU-only profiling.

        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.

        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles


    .. warning:
        Enabling memory profiling or source attribution incurs additional profiler
        overhead

    .. warning:
        This context managers should not be called recursively, i.e. no nested
        instances are allowed

    .. warning:
        Due to some SUPA multiprocessing limitations (multiprocessing-supa-note_),
        one cannot use the profiler with ``use_device = 'supa'`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_device = None`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        >>>     for _ in range(100):  # any normal python code, really!
        >>>         y = x ** 2
        >>>         y.backward()
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total   CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        mul                                  32.048ms         32.048ms         200
        pow                                  27.041ms         27.041ms         200
        PowBackward0                         9.727ms          55.483ms         100
        torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
        torch::autograd::GraphRoot           691.816us        691.816us        100
        -----------------------------------  ---------------  ---------------  ---------------

    """

    def __init__(
        self,
        enabled=True,
        *,
        use_supa=False,  # Deprecated
        use_device=None,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        use_kineto=True,
        use_cpu=True,
        use_supa_simple=False,
        experimental_config=None,
        acc_events=False,
        custom_trace_id_callback=None,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_supa = use_supa
        if self.use_supa:
            warn(
                "The attribute `use_supa` will be deprecated soon, "
                "please use ``use_device = 'supa'`` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_device: Optional[str] = "supa"
        else:
            self.use_device = use_device
        # TODO Consider changing _function_events into data structure with size cap
        self._function_events: Optional[EventList] = None
        self._old_function_events: Optional[EventList] = None
        # Function event processing is done lazily
        self._needs_processing = False
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_supa_simple = use_supa_simple
        self.use_cpu = use_cpu
        self.acc_events = acc_events
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.kineto_results: Optional[_ProfilerResult] = None
        self.profiling_start_time_ns = 0
        self.profiling_end_time_ns = 0
        self._stats = _ProfilerStats()
        self.custom_trace_id_callback = custom_trace_id_callback
        self.trace_id = ""
        if not self.use_cpu:
            assert (
                use_kineto
            ), "Device-only events supported only with Kineto (use_kineto=True)"

        if self.use_device is not None:
            VALID_DEVICE_OPTIONS = ["supa"]
            if self.use_device not in VALID_DEVICE_OPTIONS:
                warn(f"The {self.use_device} is not a valid device option.")
                self.use_device = None

            if self.use_device == "supa" and not torch_supa.supa.is_available():
                warn("SUPA is not available, disabling SUPA profiling")
                self.use_supa = False
                self.use_device = None

        self.kineto_activities = set()
        if self.use_cpu:
            self.kineto_activities.add(ProfilerActivity.CPU)

        self.profiler_kind = ProfilerState.KINETO
        if self.use_device == "supa":
            if not use_kineto or ProfilerActivity.SUPA not in _supported_activities():
                assert self.use_cpu, "Legacy SUPA profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_GPU_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.SUPA)

        assert (
            len(self.kineto_activities) > 0
        ), "No activities specified for the profiler"

    def default_trace_id(self):
        # Generate a UUID
        uuid_raw = uuid.uuid4()

        return f"{uuid_raw.int:032X}"

    def create_trace_id(self):
        if self.custom_trace_id_callback:
            return self.custom_trace_id_callback()
        return self.default_trace_id()

    def config(self, create_trace_id=False):
        # only need to generate new trace id upon prepare trace not start trace
        if create_trace_id:
            trace_id = self.create_trace_id()
            self.trace_id = trace_id
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.use_supa_simple,
            self.experimental_config,
            self.trace_id,
        )

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self._prepare_trace()
        self._start_trace()
        return self

    def _prepare_trace(self):
        self.entered = True
        t0 = perf_counter_ns()
        _prepare_profiler(self.config(create_trace_id=True), self.kineto_activities)
        t1 = perf_counter_ns()
        self._stats.profiler_prepare_call_duration_us = int((t1 - t0) / 1000)

    def _start_trace(self):
        self.entered = True
        _run_on_profiler_start()
        t0 = perf_counter_ns()
        _enable_profiler(self.config(create_trace_id=False), self.kineto_activities)
        t1 = perf_counter_ns()
        self._stats.profiler_enable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_start_time_ns = t1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_device and hasattr(torch, self.use_device):
            device_module = getattr(torch, self.use_device)
            if hasattr(device_module, "synchronize"):
                device_module.synchronize()

        if self._function_events and self.acc_events:
            self._old_function_events = self._function_events
        self._function_events = None
        self._needs_processing = True

        t0 = perf_counter_ns()

        self.kineto_results = _disable_profiler()
        t1 = perf_counter_ns()
        self._stats.profiler_disable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_end_time_ns = t0

        _run_on_profiler_stop()

        self._stats.profiling_window_duration_sec = (
            (self.profiling_end_time_ns - self.profiling_start_time_ns) * 1.0 / 1e9
        )

        # If we plan to accumulate events we should post process the function events
        # right away to retain the state across mulitple start/stop calls
        if self.acc_events:
            self._ensure_function_events()
        return False

    def __repr__(self):
        if self._needs_processing:
            self._ensure_function_events()
        if self._function_events is None:
            return "<unfinished torch.autograd.profile>"
        return repr(self._function_events)

    def __str__(self):
        if self._needs_processing:
            self._ensure_function_events()
        if self._function_events is None:
            return "<unfinished torch.autograd.profile>"
        return str(self._function_events)

    def _ensure_function_events(self):
        """Process function events lazily if required"""
        if self._function_events is not None:
            return
        self._needs_processing = False

        t0 = perf_counter_ns()
        parsed_results = []
        if self.kineto_results:
            parsed_results = self._parse_kineto_results(self.kineto_results)
        t1 = perf_counter_ns()
        self._stats.parse_kineto_call_duration_us = int((t1 - t0) / 1000)

        self._function_events = EventList(
            parsed_results,
            use_device=self.use_device,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        t0 = perf_counter_ns()
        self._function_events._build_tree()
        t1 = perf_counter_ns()
        self._stats.function_events_build_tree_call_duration_us = int((t1 - t0) / 1000)
        self._stats.number_of_events = len(self._function_events)

        if self._old_function_events and self.acc_events:
            for evt in self._old_function_events:
                self._function_events.append(evt)
            self._old_function_events = None

        if self._function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    @property
    def function_events(self):
        if self._function_events is None or self._needs_processing:
            self._ensure_function_events()
        return self._function_events

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        self._ensure_function_events()
        assert self._function_events is not None
        return self._function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        # use_kineto is True 
        self.kineto_results.save(path)  # type: ignore[union-attr]

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._ensure_function_events()
        assert self._function_events is not None, "Expected profiling results"
        assert self.with_stack, "export_stacks() requires with_stack=True"
        return self._function_events.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enabled: bool, activities: Iterable[ProfilerActivity]
    ):
        """
        Toggles the collection of activities for the current profiler instance.
        """
        return _toggle_collection_dynamic(enabled, set(activities))

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._ensure_function_events()
        assert self._function_events is not None, "Expected profiling results"
        return self._function_events.key_averages(
            group_by_input_shape, group_by_stack_n
        )

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._ensure_function_events()
        assert self._function_events is not None, "Expected profiling results"
        return self._function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Returns total time spent on CPU.

        The total time is a sum of all self times across all the events.
        """
        self._ensure_function_events()
        assert self._function_events is not None
        return self._function_events.self_cpu_time_total

    def _parse_kineto_results(self, result: _ProfilerResult):
        # result.events() has most of the events - PyTorch op-level and device-level events

        trace_start_ns = result.trace_start_ns()
        mem_records = [
            [evt, False] for evt in result.events() if evt.name() == MEMORY_EVENT_NAME
        ]
        oom_records = [
            evt for evt in result.events() if evt.name() == OUT_OF_MEMORY_EVENT_NAME
        ]
        mem_records_acc = MemRecordsAcc(mem_records)

        def _cpu_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]
                else 0
            )

        def _device_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.PrivateUse1, DeviceType.HIP]
                else 0
            )

        # Create and return FunctionEvent list, which contains all function events
        # Here 2 function events are created:
        # all_function_events contains all events associated with each kineto event from result
        all_function_events = []
        # frontend_function_events contains the events in aten or torch frontend level,
        # whose correlation id is 0
        frontend_function_events = []
        device_corr_map: Dict[int, List[FunctionEvent]] = {}
        max_evt_id = 0
        for kineto_event in result.events():
            if _filter_name(kineto_event.name()):
                continue
            rel_start_ns = kineto_event.start_ns() - trace_start_ns
            rel_end_ns = kineto_event.end_ns() - trace_start_ns
            abs_end_ns = kineto_event.end_ns()

            cpu_memory_usage = 0
            device_memory_usage = 0
            if kineto_event.device_type() == DeviceType.CPU:
                # find the corresponding memory allocation events
                for mem_record in mem_records_acc.in_interval(
                    kineto_event.start_ns() / 1000, abs_end_ns / 1000
                ):
                    cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                    device_memory_usage += _device_memory_usage(mem_record[0])
                    mem_record[1] = True

            is_async = kineto_event.is_async() or (
                kineto_event.start_thread_id() != kineto_event.end_thread_id()
            )

            fe = FunctionEvent(
                id=kineto_event.correlation_id(),
                name=_rewrite_name(name=kineto_event.name(), with_wildcard=True),
                trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False),
                thread=kineto_event.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_end_ns / 1000,
                fwd_thread=kineto_event.fwd_thread_id(),
                input_shapes=kineto_event.shapes(),
                concrete_inputs=kineto_event.concrete_inputs(),
                kwinputs=kineto_event.kwinputs(),
                stack=[
                    entry
                    for entry in kineto_event.stack()
                    if _filter_stack_entry(entry)
                ],
                scope=kineto_event.scope(),
                use_device=self.use_device,
                cpu_memory_usage=cpu_memory_usage,
                device_memory_usage=device_memory_usage,
                is_async=is_async,
                sequence_nr=kineto_event.sequence_nr(),
                device_type=kineto_event.device_type(),
                device_index=kineto_event.device_index(),
                device_resource_id=kineto_event.device_resource_id(),
                flops=kineto_event.flops(),
                is_user_annotation=kineto_event.is_user_annotation(),
            )
            max_evt_id = max(max_evt_id, fe.id)
            if fe.device_type == DeviceType.CPU and not fe.is_async:
                if self.use_device == "privateuseone":
                    privateuse1_time = kineto_event.privateuse1_elapsed_us()
                    if privateuse1_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, privateuse1_time)
                        fe.is_legacy = True
                elif self.use_device == "supa":
                    # Check if we have SUPA time as a fallback
                    supa_time = kineto_event.supa_elapsed_us()
                    if supa_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, supa_time)
                        fe.is_legacy = True
            all_function_events.append(fe)
            corr_id = kineto_event.linked_correlation_id()
            if corr_id > 0:
                if corr_id not in device_corr_map:
                    device_corr_map[corr_id] = []
                device_corr_map[corr_id].append(fe)
            elif corr_id == 0:
                frontend_function_events.append(fe)
            else:
                raise RuntimeError(
                    f"Got negative correlation id {corr_id} in profiler post processing"
                )

        # associate device kernels and device runtime (CPU) with CPU events
        for fe in frontend_function_events:
            if (
                fe.device_type == DeviceType.CPU
                and not fe.is_async
                and fe.id in device_corr_map
            ):
                for f_evt in device_corr_map[fe.id]:
                    if (
                        f_evt.device_type == DeviceType.PrivateUse1
                    ):
                        fe.append_kernel(
                            f_evt.name,
                            f_evt.device_index,
                            f_evt.time_range.end - f_evt.time_range.start,
                        )
                    elif f_evt.device_type == DeviceType.CPU:
                        # make sure that 'thread' of a CPU Kineto (e.g. Device Runtime) event is associated
                        # with the 'thread' of the corresponding linked PyTorch event to properly track
                        # parents and children
                        f_evt.thread = fe.thread

        def createFunctionEventForMemoryEvents(evt):
            rel_start_ns = evt.start_ns() - trace_start_ns
            fe = FunctionEvent(
                id=max_evt_id,
                name=evt.name(),
                trace_name=None,  # not outputting in the trace
                thread=evt.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_start_ns / 1000,  # no duration
                fwd_thread=evt.start_thread_id(),
                input_shapes=[],
                stack=[],
                scope=0,  # RecordScope::FUNCTION
                use_device=self.use_device,
                cpu_memory_usage=_cpu_memory_usage(evt),
                device_memory_usage=_device_memory_usage(evt),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            return fe

        # output top-level memory events
        for mem_record in mem_records:
            if not mem_record[1]:
                max_evt_id += 1
                fe = createFunctionEventForMemoryEvents(mem_record[0])
                all_function_events.append(fe)

        for oom_record in oom_records:
            max_evt_id += 1
            fe = createFunctionEventForMemoryEvents(oom_record)
            all_function_events.append(fe)

        all_function_events.sort(
            key=lambda evt: [evt.time_range.start, -evt.time_range.end]
        )
        return all_function_events

    def add_metadata(self, key: str, value: Any):
        if isinstance(value, str):
            value = f'"{value}"'
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)
        else:
            value = str(value)

        _add_metadata_json(key, value)


class KinetoStepTracker:
    """Provides an abstraction for incrementing the step count globally.

    Previously, we only had one place to mark that a step() has occurred
    in the program via pytorch profiler step(). We will now add step hooks
    in the Optimizer class https://github.com/pytorch/pytorch/issues/88446

    - This could mean programs that already call profiler.step() every
      iteration can end up double incrementing step count.
    - If a model uses multiple optimizers we can also have double or more
      counting of the step.

    We fix this by adding a layer of abstraction before calling step()
    to the kineto library. The idea is to maintain steps per requester in a dict:

    .. code-block::

        {
           "ProfilerStep": 100,  # triggered by profiler step() call
           "Optimizer1Step": 100,   # Optimizer 1 or 2 are just examples, could be SGD, Adam etc
           "Optimizer2Step": 100,
        }

    To figure out the global step count just take the max of dict values (100).

    If one of the count increments the max will go up.

    .. code-block::

        {
           "ProfilerStep": 100,
           "Optimizer1Step": 101,   # Optimizer1 got incremented first say
           "Optimizer2Step": 100,
        }

    Then global step count is 101
    We only call the kineto step() function when global count increments.

    NOTE: Please do not use the KinetoStepTracker in modules beside the Optimizer
    for now. The result could be incorrect increments of the step count.
    """

    _current_step = 0
    _step_dict: Dict[str, int] = defaultdict(int)

    @classmethod
    def init_step_count(cls, requester: str):
        r"""
        Initialize for a given requester.
        """
        cls._step_dict[requester] = cls._current_step

    @classmethod
    def erase_step_count(cls, requester: str) -> bool:
        r"""
        Remove a given requester.
        """
        return cls._step_dict.pop(requester, None) is not None

    @classmethod
    def increment_step(cls, requester: str) -> int:
        """Increments the step count for the requester.

        Additionally if the max over all step counts has incremented then
        trigger the _kineto_step() returns global step count
        """
        if requester not in cls._step_dict:
            cls.init_step_count(requester)
        cls._step_dict[requester] += 1

        new_step = max(cls._step_dict.values())
        if new_step > cls._current_step:
            delta = new_step - cls._current_step
            if delta > 1:
                warn(
                    "Profiler step count has increased more than 1 - "
                    f"current_step = {cls._current_step} step dict =  {cls._step_dict}"
                )
            for _ in range(0, delta):
                _kineto_step()
            cls._current_step = new_step
        return cls._current_step

    @classmethod
    def current_step(cls) -> int:
        r"""
        Get the latest step for any requester
        """
        return cls._current_step
