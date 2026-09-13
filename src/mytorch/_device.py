"""CUDA device parsing shared by tensor creation and operations."""

from __future__ import annotations

import re

import cupy as cp

from .cuda import CudaUnavailableError

_CUDA_DEVICE = re.compile(r"^cuda(?::(?P<index>\d+))?$")


def parse_device(device: str | int | None) -> int:
    """Normalize a CUDA device specification to an available device index."""
    if device is None:
        index = 0
    elif isinstance(device, bool):
        raise TypeError("device must be a CUDA device string or integer index")
    elif isinstance(device, int):
        index = device
    elif isinstance(device, str):
        match = _CUDA_DEVICE.fullmatch(device.lower())
        if match is None:
            raise ValueError(
                f"MyTorch is GPU-only; expected 'cuda:N', but received {device!r}"
            )
        index = int(match.group("index") or 0)
    else:
        raise TypeError("device must be a CUDA device string or integer index")

    if index < 0:
        raise ValueError(f"CUDA device index must be non-negative, got {index}")

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise CudaUnavailableError("CUDA could not be initialized") from exc
    if index >= count:
        raise CudaUnavailableError(
            f"CUDA device {index} is unavailable; detected {count} device(s)"
        )
    return index


def format_device(index: int) -> str:
    return f"cuda:{index}"
