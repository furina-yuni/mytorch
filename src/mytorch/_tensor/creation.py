"""Tensor creation and random initialization helpers."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch._device import parse_device
from mytorch.tensor import DTypeLike, ShapeLike, Tensor, _shape_from_args


def tensor(
    data: Any,
    *,
    dtype: DTypeLike | None = None,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def empty(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _create(cp.empty, shape, dtype, device, requires_grad)


def zeros(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _create(cp.zeros, shape, dtype, device, requires_grad)


def ones(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _create(cp.ones, shape, dtype, device, requires_grad)


def full(
    shape: ShapeLike,
    fill_value: Any,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _create(
        cp.full,
        (shape,),
        dtype,
        device,
        requires_grad,
        fill_value=fill_value,
    )


def _create(
    operation: Any,
    shape: tuple[ShapeLike, ...],
    dtype: DTypeLike,
    device: str | int,
    requires_grad: bool,
    **kwargs: Any,
) -> Tensor:
    device_index = parse_device(device)
    normalized = _shape_from_args(shape)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            operation(normalized, dtype=dtype, **kwargs),
            requires_grad=requires_grad,
        )


def arange(
    start: float,
    end: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    device_index = parse_device(device)
    if end is None:
        start, end = 0, start
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            cp.arange(start, end, step, dtype=dtype), requires_grad=requires_grad
        )


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    device_index = parse_device(device)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            cp.linspace(start, end, steps, dtype=dtype),
            requires_grad=requires_grad,
        )


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    device_index = parse_device(device)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            cp.eye(n, m, dtype=dtype), requires_grad=requires_grad
        )


def rand(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _random_create(cp.random.random, shape, dtype, device, requires_grad)


def randn(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
    requires_grad: bool = False,
) -> Tensor:
    return _random_create(
        cp.random.standard_normal, shape, dtype, device, requires_grad
    )


def _random_create(
    operation: Any,
    shape: tuple[ShapeLike, ...],
    dtype: DTypeLike,
    device: str | int,
    requires_grad: bool,
) -> Tensor:
    device_index = parse_device(device)
    normalized = _shape_from_args(shape)
    resolved_dtype = cp.dtype(dtype)
    generation_dtype = cp.float32 if resolved_dtype == cp.float16 else resolved_dtype
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            operation(normalized, dtype=generation_dtype).astype(
                resolved_dtype, copy=False
            ),
            requires_grad=requires_grad,
        )


def _like(
    operation: Any,
    source: Tensor,
    dtype: DTypeLike | None,
    device: str | int | None,
    requires_grad: bool,
) -> Tensor:
    if not isinstance(source, Tensor):
        raise TypeError("like creation expects a Tensor")
    device_index = source._device_index if device is None else parse_device(device)
    resolved_dtype = source.dtype if dtype is None else dtype
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            operation(source.shape, dtype=resolved_dtype),
            requires_grad=requires_grad,
        )


def empty_like(
    source: Tensor,
    *,
    dtype: DTypeLike | None = None,
    device: str | int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return _like(cp.empty, source, dtype, device, requires_grad)


def zeros_like(
    source: Tensor,
    *,
    dtype: DTypeLike | None = None,
    device: str | int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return _like(cp.zeros, source, dtype, device, requires_grad)


def ones_like(
    source: Tensor,
    *,
    dtype: DTypeLike | None = None,
    device: str | int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return _like(cp.ones, source, dtype, device, requires_grad)


def manual_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    current_device = int(cp.cuda.Device().id)
    try:
        for device_index in range(int(cp.cuda.runtime.getDeviceCount())):
            with cp.cuda.Device(device_index):
                cp.random.seed(seed)
    finally:
        cp.cuda.Device(current_device).use()
