"""Checkpointable GPU random-number generation for MyTorch."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Mapping
from typing import Any

import cupy as cp

_STATE_VERSION = 1
_MASK_64 = (1 << 64) - 1
_lock = threading.Lock()
_seed = secrets.randbits(64)
_counters: dict[int, int] = {}


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def manual_seed(seed: int) -> None:
    """Reset every MyTorch GPU random stream to a reproducible seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    global _seed
    with _lock:
        _seed = seed & _MASK_64
        _counters.clear()


def get_rng_state() -> dict[str, Any]:
    """Return a safe, serializable snapshot of all device random streams."""
    with _lock:
        return {
            "version": _STATE_VERSION,
            "seed": _seed,
            "counters": dict(_counters),
        }


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state previously returned by :func:`get_rng_state`."""
    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping")
    if set(state) != {"version", "seed", "counters"}:
        raise ValueError("RNG state has missing or unexpected fields")
    if state["version"] != _STATE_VERSION:
        raise ValueError(f"unsupported RNG state version: {state['version']!r}")
    seed = state["seed"]
    counters = state["counters"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("RNG seed must be a non-negative integer")
    if not isinstance(counters, Mapping):
        raise TypeError("RNG counters must be a mapping")
    normalized: dict[int, int] = {}
    for device, counter in counters.items():
        if (
            not isinstance(device, int)
            or isinstance(device, bool)
            or device < 0
            or not isinstance(counter, int)
            or isinstance(counter, bool)
            or counter < 0
        ):
            raise ValueError(
                "RNG device ids and counters must be non-negative integers"
            )
        normalized[device] = counter
    global _seed
    with _lock:
        _seed = seed & _MASK_64
        _counters.clear()
        _counters.update(normalized)


def _generator(device: int | None = None) -> cp.random.RandomState:
    device_index = int(cp.cuda.Device().id) if device is None else int(device)
    with _lock:
        counter = _counters.get(device_index, 0)
        _counters[device_index] = counter + 1
        local_seed = _mix64(_seed ^ _mix64(device_index) ^ _mix64(counter))
    return cp.random.RandomState(local_seed)


def random(
    shape: tuple[int, ...], *, device: int | None = None, dtype: Any = cp.float64
) -> cp.ndarray:
    device_index = int(cp.cuda.Device().id) if device is None else device
    with cp.cuda.Device(device_index):
        return _generator(device_index).random_sample(shape, dtype=dtype)


def standard_normal(
    shape: tuple[int, ...], *, device: int | None = None, dtype: Any = cp.float64
) -> cp.ndarray:
    device_index = int(cp.cuda.Device().id) if device is None else device
    with cp.cuda.Device(device_index):
        return _generator(device_index).standard_normal(shape, dtype=dtype)


def normal(
    mean: float,
    std: float,
    shape: tuple[int, ...],
    *,
    device: int | None = None,
    dtype: Any = cp.float64,
) -> cp.ndarray:
    device_index = int(cp.cuda.Device().id) if device is None else device
    with cp.cuda.Device(device_index):
        return _generator(device_index).normal(mean, std, shape, dtype=dtype)


def uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: int | None = None,
    dtype: Any = cp.float64,
) -> cp.ndarray:
    device_index = int(cp.cuda.Device().id) if device is None else device
    with cp.cuda.Device(device_index):
        return _generator(device_index).uniform(low, high, shape, dtype=dtype)
