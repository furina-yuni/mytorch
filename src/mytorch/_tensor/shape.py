"""Tensor shape, indexing, and joining operations."""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from typing import Any

import cupy as cp

from mytorch import _autograd, _ops
from mytorch.tensor import ShapeLike, Tensor, _shape_from_args

from .elementwise import where


def contiguous(input: Tensor) -> Tensor:
    return _ops.apply(
        cp.ascontiguousarray,
        input,
        backward=lambda gradient, _result, _arrays: (gradient,),
        name="contiguous",
    )


def expand(input: Tensor, *shape: ShapeLike) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target = tuple(shape[0])
    else:
        target = tuple(shape)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in target
    ):
        raise TypeError("expand dimensions must be integers")
    if len(target) < input.ndim:
        raise ValueError("expand cannot remove Tensor dimensions")
    padded = (1,) * (len(target) - input.ndim) + input.shape
    resolved = tuple(
        source if requested == -1 else requested
        for source, requested in zip(padded, target, strict=True)
    )
    if any(value < 0 for value in resolved):
        raise ValueError("expand dimensions must be non-negative or -1")
    for source, requested in zip(padded, resolved, strict=True):
        if source != requested and source != 1:
            raise ValueError(f"cannot expand shape {input.shape} to {resolved}")
    return _ops.apply(
        lambda array: cp.broadcast_to(array.reshape(padded), resolved),
        input,
        backward=lambda gradient, _result, arrays: (
            _autograd.sum_to_shape(gradient, padded).reshape(arrays[0].shape),
        ),
        name="expand",
    )


def repeat(input: Tensor, *repeats: ShapeLike) -> Tensor:
    normalized = _shape_from_args(repeats)
    if len(normalized) < input.ndim:
        raise ValueError("repeat expects at least as many values as Tensor dimensions")
    if any(value < 0 for value in normalized):
        raise ValueError("repeat counts must be non-negative")
    padded_shape = (1,) * (len(normalized) - input.ndim) + input.shape

    def backward(gradient, _result, arrays):
        view_shape: list[int] = []
        reduction_axes: list[int] = []
        for axis, (count, size) in enumerate(
            zip(normalized, padded_shape, strict=True)
        ):
            view_shape.extend((count, size))
            reduction_axes.append(axis * 2)
        result = gradient.reshape(view_shape).sum(axis=tuple(reduction_axes))
        return (result.reshape(arrays[0].shape),)

    return _ops.apply(
        lambda array: cp.tile(array.reshape(padded_shape), normalized),
        input,
        backward=backward,
        name="repeat",
    )


def pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    if mode != "constant":
        raise ValueError("only constant padding is supported")
    if not isinstance(pad, Sequence) or len(pad) % 2 or len(pad) > 2 * input.ndim:
        raise ValueError("pad must contain pairs for trailing dimensions")
    if not all(isinstance(item, int) and item >= 0 for item in pad):
        raise ValueError("padding values must be non-negative integers")
    pairs = [(0, 0)] * (input.ndim - len(pad) // 2)
    trailing = list(zip(pad[::2], pad[1::2], strict=True))[::-1]
    pairs.extend(trailing)
    slices = tuple(
        slice(before, before + size)
        for (before, _), size in zip(pairs, input.shape, strict=True)
    )
    return _ops.apply(
        lambda array: cp.pad(
            array, tuple(pairs), mode="constant", constant_values=value
        ),
        input,
        backward=lambda gradient, _result, _arrays: (gradient[slices],),
        name="pad",
    )


def masked_fill(input: Tensor, mask: Tensor, value: Any) -> Tensor:
    if not isinstance(mask, Tensor) or mask.dtype != cp.bool_:
        raise TypeError("masked_fill mask must be a boolean Tensor")
    return where(mask, value, input)


def _gather_coordinates(index: cp.ndarray, axis: int) -> tuple[cp.ndarray, ...]:
    coordinates = []
    for dim, size in enumerate(index.shape):
        if dim == axis:
            coordinates.append(index)
        else:
            shape = [1] * index.ndim
            shape[dim] = size
            coordinates.append(cp.arange(size).reshape(shape))
    return tuple(coordinates)


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    if not isinstance(index, Tensor) or index.dtype.kind not in "iu":
        raise TypeError("gather index must be an integer Tensor")
    axis = _ops.normalize_dims(dim, input.ndim)
    assert isinstance(axis, int)
    if index.ndim != input.ndim:
        raise ValueError(
            "gather input and index must have the same number of dimensions"
        )
    for current, (index_size, input_size) in enumerate(
        zip(index.shape, input.shape, strict=True)
    ):
        if current != axis and index_size > input_size:
            raise ValueError("gather index is too large for a non-gather dimension")
    if input._device_index != index._device_index:
        raise ValueError("gather input and index must be on the same device")

    def backward(gradient, _result, arrays):
        source, indices = arrays
        result = cp.zeros_like(source)
        cp.add.at(result, _gather_coordinates(indices, axis), gradient)
        return result, None

    return _ops.apply(
        lambda array, indices: cp.take_along_axis(array, indices, axis=axis),
        input,
        index,
        backward=backward,
        name="gather",
    )


def scatter_add(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    if not isinstance(index, Tensor) or index.dtype.kind not in "iu":
        raise TypeError("scatter_add index must be an integer Tensor")
    if not isinstance(source, Tensor):
        raise TypeError("scatter_add source must be a Tensor")
    axis = _ops.normalize_dims(dim, input.ndim)
    assert isinstance(axis, int)
    if index.shape != source.shape or index.ndim != input.ndim:
        raise ValueError(
            "scatter_add index and source must have matching input-rank shapes"
        )

    def forward(base, indices, values):
        result = base.copy()
        cp.add.at(result, _gather_coordinates(indices, axis), values)
        return result

    def backward(gradient, _result, arrays):
        _, indices, _ = arrays
        return gradient, None, cp.take_along_axis(gradient, indices, axis=axis)

    return _ops.apply(
        forward,
        input,
        index,
        source,
        backward=backward,
        name="scatter_add",
    )


def topk(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[Tensor, Tensor]:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(largest, bool) or not isinstance(sorted, bool):
        raise TypeError("largest and sorted must be bool values")
    axis = _ops.normalize_dims(dim, input.ndim)
    assert isinstance(axis, int)
    if k > input.shape[axis]:
        raise ValueError("k cannot exceed the selected dimension")
    array = input._array
    partition = cp.argpartition(-array if largest else array, k - 1, axis=axis)
    selection = [slice(None)] * input.ndim
    selection[axis] = slice(0, k)
    indices_array = partition[tuple(selection)]
    values_array = cp.take_along_axis(array, indices_array, axis=axis)
    if sorted:
        order = cp.argsort(-values_array if largest else values_array, axis=axis)
        indices_array = cp.take_along_axis(indices_array, order, axis=axis)
        values_array = cp.take_along_axis(values_array, order, axis=axis)
    indices = Tensor._from_array(indices_array.astype(cp.int64, copy=False))
    if not input.requires_grad or not _autograd.is_grad_enabled():
        return Tensor._from_array(values_array), indices

    def backward(gradient):
        result = cp.zeros_like(array)
        cp.add.at(result, _gather_coordinates(indices_array, axis), gradient)
        return (result,)

    node = _autograd.Node(
        name="topk",
        parents=(input,),
        backward_fn=backward,
        versions=(input._version,),
    )
    return Tensor._from_array(values_array, requires_grad=True, grad_fn=node), indices


def split(
    input: Tensor,
    split_size_or_sections: int | Sequence[int],
    dim: int = 0,
) -> tuple[Tensor, ...]:
    axis = _ops.normalize_dims(dim, input.ndim)
    assert isinstance(axis, int)
    length = input.shape[axis]
    if isinstance(split_size_or_sections, int) and not isinstance(
        split_size_or_sections, bool
    ):
        if split_size_or_sections <= 0:
            raise ValueError("split size must be positive")
        sizes = [split_size_or_sections] * (length // split_size_or_sections)
        if length % split_size_or_sections:
            sizes.append(length % split_size_or_sections)
    elif isinstance(split_size_or_sections, Sequence):
        sizes = list(split_size_or_sections)
        if not sizes or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in sizes
        ):
            raise ValueError("split sections must be non-negative integers")
        if builtins.sum(sizes) != length:
            raise ValueError("split sections must sum to the selected dimension")
    else:
        raise TypeError("split size must be an int or sequence of ints")
    outputs = []
    start = 0
    for size in sizes:
        index = [slice(None)] * input.ndim
        index[axis] = slice(start, start + size)
        outputs.append(input[tuple(index)])
        start += size
    return tuple(outputs)


def chunk(input: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks <= 0:
        raise ValueError("chunks must be a positive integer")
    axis = _ops.normalize_dims(dim, input.ndim)
    assert isinstance(axis, int)
    length = input.shape[axis]
    if length == 0:
        return split(input, [0] * chunks, dim=axis)
    split_size = math.ceil(length / chunks)
    return split(input, split_size, dim=axis)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.concatenate(tensors, dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.stack(tensors, dim=dim)
