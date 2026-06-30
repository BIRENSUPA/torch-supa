# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.testing._internal.distributed.multi_threaded_pg as multi_threaded_pg


def patch_multi_threaded_pg():
    patch_multi_threaded_pg_Collective_join()


def patch_multi_threaded_pg_Collective_join():
    def join(self, rank, data):
        device_module = _get_stream_sync_handle(data)
        input_event = None
        if device_module is not None:
            input_event = device_module.Event()
            input_event.record(device_module.current_stream())

        with self._start_cond:
            self._data[rank] = (data, input_event)
            self._count += 1

            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()

            if rank == 0:
                self._start_cond.wait_for(
                    lambda: self._count == self._world_size
                    or self._pg._terminate.is_set()
                )
                if self._pg._terminate.is_set():
                    raise SystemExit("Test termination event occurs.")

        with self._done_cond:
            if rank > 0:
                self._done_cond.wait_for(
                    lambda: self._done or self._pg._terminate.is_set()
                )
                if self._pg._terminate.is_set():
                    raise SystemExit("Test termination event occurs.")
                done_event = getattr(self, "_device_done_event", None)
                done_event_device_module = getattr(self, "_device_done_event_module", None)
                if done_event is not None and done_event_device_module is not None:
                    _wait_event(done_event_device_module, done_event)
            else:
                collective_device_module = None
                for _, maybe_event in self._data:
                    if maybe_event is not None:
                        collective_device_module = device_module
                        if collective_device_module is not None:
                            _wait_event(collective_device_module, maybe_event)
                self._collective.work([rank_data for rank_data, _ in self._data])
                self._device_done_event = None
                self._device_done_event_module = None
                if collective_device_module is not None:
                    self._device_done_event = collective_device_module.Event()
                    self._device_done_event.record(collective_device_module.current_stream())
                    self._device_done_event_module = collective_device_module
                self._done = True
                self._done_cond.notify_all()
        return multi_threaded_pg.ret_work(data)

    multi_threaded_pg.Collective.join = join


def _wait_event(device_module, event):
    event.wait(device_module.current_stream())


def _get_stream_sync_handle(obj):
    if isinstance(obj, torch.Tensor):
        device_module = getattr(torch, obj.device.type, None)
        if device_module is None:
            return None
        required_attrs = ("Event", "current_stream")
        if all(hasattr(device_module, attr) for attr in required_attrs):
            return device_module
        return None
    if isinstance(obj, dict):
        for key, value in obj.items():
            device_module = _get_stream_sync_handle(key) or _get_stream_sync_handle(value)
            if device_module is not None:
                return device_module
    if isinstance(obj, (list, tuple)):
        for value in obj:
            device_module = _get_stream_sync_handle(value)
            if device_module is not None:
                return device_module
    return None
