# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import ctypes

import torch_supa
import torch_supa._C

__all__ = ["Stream", "Event", "ExternalStream"]

class Stream(torch_supa._C._SUPAStreamBase):
    r"""Wrapper around a SUPA stream.

    A SUPA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`supa-semantics` for
    details.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch_supa.supa.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch_supa.supa.Event): an event to wait for.

        .. note:: This is a wrapper around ``supaStreamWaitEvent()``: see
           `SUPA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _SUPA Stream documentation:
           https://docs.nvidia.com/supa/supa-runtime-api/group__SUPART__STREAM.html
        """
        event.wait(self)

    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch_supa.supa.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``supaStreamSynchronize()``: see
           `SUPA Stream documentation`_ for more info.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.supa_stream)

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.supa_stream, self.device))

    def __repr__(self):
        return f"<torch_supa.supa.Stream device={self.device} supa_stream={self.supa_stream:#x}>"


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated SUPA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `supaStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch_supa.supa.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


class Event(torch_supa._C._SUPAEventBase):
    r"""Wrapper around a SUPA event.

    SUPA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize SUPA
    streams.

    The underlying SUPA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    .. _SUPA Event Documentation:
       https://docs.nvidia.com/supa/supa-runtime-api/group__SUPART__EVENT.html
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Record the event in a given stream.

        Uses ``torch_supa.supa.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch_supa.supa.current_stream()
        super().record(stream)

    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch_supa.supa.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``supaStreamWaitEvent()``: see
            `SUPA Event documentation`_ for more info.
        """
        if stream is None:
            stream = torch_supa.supa.current_stream()
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``supaEventSynchronize()``: see
            `SUPA Event documentation`_ for more info.
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.supa_event)

    def __repr__(self) -> str:
        if self.supa_event:
            return f"<torch_supa.supa.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch_supa.supa.Event uninitialized>"