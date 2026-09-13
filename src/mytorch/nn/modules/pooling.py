"""N-dimensional pooling implementations."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import cupy as cp

from mytorch import _ops
from mytorch.tensor import Tensor, stack

from .base import Module
from .convolution import _output_shape, _tuple


def pool_nd(
    input: Tensor,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    mode: str,
) -> Tensor:
    dims = input.ndim - 2
    kernel = _tuple(kernel_size, dims, "kernel_size")
    steps = kernel if stride is None else _tuple(stride, dims, "stride")
    pads = _tuple(padding, dims, "padding")
    output_spatial = _output_shape(input.shape[2:], kernel, steps, pads, (1,) * dims)
    pad_value = -cp.inf if mode == "max" else 0
    pad_spec = ((0, 0), (0, 0)) + tuple((value, value) for value in pads)

    def forward(source):
        padded = cp.pad(source, pad_spec, constant_values=pad_value)
        windows = []
        for kernel_index in itertools.product(*(range(size) for size in kernel)):
            slices = tuple(
                slice(offset, offset + output_size * step, step)
                for offset, output_size, step in zip(
                    kernel_index, output_spatial, steps, strict=True
                )
            )
            windows.append(padded[(slice(None), slice(None), *slices)])
        values = cp.stack(windows, axis=2)
        return values.max(axis=2) if mode == "max" else values.mean(axis=2)

    def backward(gradient, result, arrays):
        source = arrays[0]
        padded = cp.pad(source, pad_spec, constant_values=pad_value)
        grad_padded = cp.zeros_like(padded)
        count = math.prod(kernel)
        kernel_indices = list(itertools.product(*(range(size) for size in kernel)))
        winners = None
        if mode == "max":
            windows = []
            for kernel_index in kernel_indices:
                winner_slices = tuple(
                    slice(offset, offset + output_size * step, step)
                    for offset, output_size, step in zip(
                        kernel_index, output_spatial, steps, strict=True
                    )
                )
                windows.append(padded[(slice(None), slice(None), *winner_slices)])
            winners = cp.stack(windows, axis=2).argmax(axis=2)
        for flat_index, kernel_index in enumerate(kernel_indices):
            slices = tuple(
                slice(offset, offset + output_size * step, step)
                for offset, output_size, step in zip(
                    kernel_index, output_spatial, steps, strict=True
                )
            )
            if mode == "max":
                contribution = gradient * (winners == flat_index)
            else:
                contribution = gradient / count
            grad_padded[(slice(None), slice(None), *slices)] += contribution
        crop = (slice(None), slice(None)) + tuple(
            slice(pad, pad + size)
            for pad, size in zip(pads, source.shape[2:], strict=True)
        )
        return (grad_padded[crop],)

    return _ops.apply(forward, input, backward=backward, name=f"{mode}_pool{dims}d")


def adaptive_pool_nd(
    input: Tensor, output_size: int | Sequence[int], mode: str
) -> Tensor:
    dims = input.ndim - 2
    target = _tuple(output_size, dims, "output_size")
    values = []
    for output_index in itertools.product(*(range(size) for size in target)):
        slices = tuple(
            slice(
                math.floor(index * input_size / output_count),
                math.ceil((index + 1) * input_size / output_count),
            )
            for index, input_size, output_count in zip(
                output_index, input.shape[2:], target, strict=True
            )
        )
        region = input[(slice(None), slice(None), *slices)]
        axes = tuple(range(2, region.ndim))
        values.append(region.max(axes) if mode == "max" else region.mean(axes))
    return stack(values, dim=-1).reshape(input.shape[:2] + target)


class _PoolNd(Module):
    dims = 0
    mode = "max"

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != self.dims + 2:
            raise ValueError("pooling input has the wrong rank")
        return pool_nd(input, self.kernel_size, self.stride, self.padding, self.mode)


class MaxPool1d(_PoolNd):
    dims = 1


class MaxPool2d(_PoolNd):
    dims = 2


class MaxPool3d(_PoolNd):
    dims = 3


class AvgPool1d(_PoolNd):
    dims = 1
    mode = "avg"


class AvgPool2d(_PoolNd):
    dims = 2
    mode = "avg"


class AvgPool3d(_PoolNd):
    dims = 3
    mode = "avg"


class _AdaptivePoolNd(Module):
    dims = 0
    mode = "max"

    def __init__(self, output_size: int | Sequence[int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != self.dims + 2:
            raise ValueError("adaptive pooling input has the wrong rank")
        return adaptive_pool_nd(input, self.output_size, self.mode)


class AdaptiveMaxPool1d(_AdaptivePoolNd):
    dims = 1


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    dims = 2


class AdaptiveMaxPool3d(_AdaptivePoolNd):
    dims = 3


class AdaptiveAvgPool1d(_AdaptivePoolNd):
    dims = 1
    mode = "avg"


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    dims = 2
    mode = "avg"


class AdaptiveAvgPool3d(_AdaptivePoolNd):
    dims = 3
    mode = "avg"
