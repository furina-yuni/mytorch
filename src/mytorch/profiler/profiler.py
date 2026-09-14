"""CUDA-event based profiling for MyTorch operations."""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import cupy as cp


@dataclass(frozen=True)
class FunctionEvent:
    """One measured GPU operation."""

    name: str
    device: int
    cuda_time_ms: float
    input_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FunctionEventAvg:
    """Aggregated measurements for operations sharing the same name."""

    name: str
    count: int
    cuda_time_total_ms: float
    cuda_time_avg_ms: float


@dataclass
class _PendingEvent:
    name: str
    device: int
    start: Any
    end: Any
    input_shapes: tuple[tuple[int, ...], ...]


_active_profiler: ContextVar[profile | None] = ContextVar(
    "mytorch_active_profiler", default=None
)


class profile:
    """Measure MyTorch operator CUDA time inside a context manager."""

    def __init__(self, *, enabled: bool = True, record_shapes: bool = False) -> None:
        if not isinstance(enabled, bool) or not isinstance(record_shapes, bool):
            raise TypeError("enabled and record_shapes must be bool values")
        self.enabled = enabled
        self.record_shapes = record_shapes
        self._pending: list[_PendingEvent] = []
        self._events: list[FunctionEvent] | None = None
        self._token: Token | None = None
        self._entered = False

    def __enter__(self) -> profile:
        if self._entered:
            raise RuntimeError("a profiler instance cannot be entered more than once")
        if self.enabled and _active_profiler.get() is not None:
            raise RuntimeError("nested profiler contexts are not supported")
        self._entered = True
        if self.enabled:
            self._token = _active_profiler.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._token is not None:
            _active_profiler.reset(self._token)
            self._token = None
        if self.enabled:
            self._finalize()

    def _append(
        self,
        name: str,
        device: int,
        start: Any,
        end: Any,
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        self._pending.append(_PendingEvent(name, device, start, end, input_shapes))

    def _finalize(self) -> None:
        if self._events is not None:
            return
        events: list[FunctionEvent] = []
        for pending in self._pending:
            with cp.cuda.Device(pending.device):
                pending.end.synchronize()
                duration = float(cp.cuda.get_elapsed_time(pending.start, pending.end))
            events.append(
                FunctionEvent(
                    pending.name,
                    pending.device,
                    duration,
                    pending.input_shapes,
                )
            )
        self._events = events
        self._pending.clear()

    def events(self) -> tuple[FunctionEvent, ...]:
        """Return individual operator measurements in execution order."""
        if self.enabled and self._token is not None:
            raise RuntimeError("events are available after the profiler context exits")
        self._finalize()
        return tuple(self._events or ())

    def key_averages(self) -> tuple[FunctionEventAvg, ...]:
        """Aggregate measurements by operation name."""
        totals: dict[str, list[float]] = defaultdict(list)
        for event in self.events():
            totals[event.name].append(event.cuda_time_ms)
        return tuple(
            FunctionEventAvg(name, len(values), sum(values), sum(values) / len(values))
            for name, values in sorted(totals.items())
        )

    def summary(self, *, sort_by: str = "cuda_time_total") -> str:
        """Return a compact text table of aggregated CUDA timings."""
        if sort_by not in {"cuda_time_total", "cuda_time_avg", "count", "name"}:
            raise ValueError(
                "sort_by must be cuda_time_total, cuda_time_avg, count, or name"
            )
        rows = list(self.key_averages())
        keys = {
            "cuda_time_total": lambda item: item.cuda_time_total_ms,
            "cuda_time_avg": lambda item: item.cuda_time_avg_ms,
            "count": lambda item: item.count,
            "name": lambda item: item.name,
        }
        rows.sort(key=keys[sort_by], reverse=sort_by != "name")
        lines = [
            "Name                         Calls   CUDA total   CUDA avg",
            "---------------------------  ------  -----------  -----------",
        ]
        lines.extend(
            f"{row.name[:27]:27}  {row.count:6d}  "
            f"{row.cuda_time_total_ms:9.3f} ms  {row.cuda_time_avg_ms:9.3f} ms"
            for row in rows
        )
        return "\n".join(lines)


def _record_operation(name: str, device: int, arrays: list[Any], operation):
    active = _active_profiler.get()
    if active is None or not active.enabled:
        return operation()
    shapes = (
        tuple(tuple(value.shape) for value in arrays if isinstance(value, cp.ndarray))
        if active.record_shapes
        else ()
    )
    with cp.cuda.Device(device):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        try:
            result = operation()
        finally:
            end.record()
    active._append(name, device, start, end, shapes)
    return result
