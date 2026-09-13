"""Centralized forward operations for GPU tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor

ArrayOperation = Callable[..., Any]


def _tensor_type():
    from .tensor import Tensor

    return Tensor


def _tensor_operands(values: Sequence[Any]) -> list[Tensor]:
    tensor_type = _tensor_type()
    return [value for value in values if isinstance(value, tensor_type)]


def _device_for(values: Sequence[Any]) -> int:
    tensors = _tensor_operands(values)
    if not tensors:
        raise TypeError("at least one operand must be a Tensor")
    device = tensors[0]._device_index
    for tensor in tensors[1:]:
        if tensor._device_index != device:
            raise ValueError(
                "all Tensor operands must be on the same CUDA device; "
                f"got cuda:{device} and cuda:{tensor._device_index}"
            )
    return device


def _unwrap(value: Any, device: int) -> Any:
    tensor_type = _tensor_type()
    if isinstance(value, tensor_type):
        if value._device_index != device:
            raise ValueError(
                "Tensor operands cannot be moved between devices implicitly"
            )
        return value._array
    if value is None or isinstance(value, (Number, np.generic)):
        return value
    if isinstance(value, (cp.ndarray, np.ndarray, list, tuple)):
        raise TypeError("non-scalar operands must be wrapped with mytorch.tensor()")
    raise TypeError(f"unsupported operand type: {type(value).__name__}")


def apply(operation: ArrayOperation, *inputs: Any, **kwargs: Any) -> Tensor:
    """Run one forward operation and wrap its result without a host copy."""
    tensor_type = _tensor_type()
    device = _device_for(inputs)
    arrays = [_unwrap(value, device) for value in inputs]
    with cp.cuda.Device(device):
        result = operation(*arrays, **kwargs)
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
    return tensor_type._from_array(result)


def binary(operation: ArrayOperation, left: Any, right: Any) -> Tensor:
    return apply(operation, left, right)


def unary(operation: ArrayOperation, tensor: Tensor) -> Tensor:
    return apply(operation, tensor)


def reduction(
    operation: ArrayOperation,
    tensor: Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    **kwargs: Any,
) -> Tensor:
    axes = normalize_dims(dim, tensor.ndim)
    return apply(operation, tensor, axis=axes, keepdims=keepdim, **kwargs)


def normalize_dims(
    dim: int | tuple[int, ...] | None, ndim: int
) -> int | tuple[int, ...] | None:
    if dim is None:
        return None
    dims = (dim,) if isinstance(dim, int) and not isinstance(dim, bool) else dim
    if not isinstance(dims, tuple) or not all(
        isinstance(axis, int) and not isinstance(axis, bool) for axis in dims
    ):
        raise TypeError("dim must be an int, tuple of ints, or None")
    normalized = tuple(axis + ndim if axis < 0 else axis for axis in dims)
    if any(axis < 0 or axis >= ndim for axis in normalized):
        raise ValueError(f"dim {dim!r} is out of range for a {ndim}D Tensor")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"dim contains duplicates: {dim!r}")
    return normalized[0] if isinstance(dim, int) else normalized


def matrix_binary(
    name: str,
    operation: ArrayOperation,
    left: Tensor,
    right: Tensor,
    *,
    required_ndim: int | None = None,
    matching_batch: bool = False,
) -> Tensor:
    tensor_type = _tensor_type()
    if not isinstance(left, tensor_type) or not isinstance(right, tensor_type):
        raise TypeError(f"{name} expects two Tensor operands")
    if required_ndim is not None and (
        left.ndim != required_ndim or right.ndim != required_ndim
    ):
        raise ValueError(
            f"{name} expects two {required_ndim}D Tensors, "
            f"got shapes {left.shape} and {right.shape}"
        )
    if matching_batch and left.shape[0] != right.shape[0]:
        raise ValueError(
            f"{name} expects matching batch sizes, "
            f"got shapes {left.shape} and {right.shape}"
        )
    try:
        return apply(operation, left, right)
    except ValueError as exc:
        raise ValueError(
            f"{name} cannot operate on shapes {left.shape} and {right.shape}: {exc}"
        ) from exc


def concatenate(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("cat expects a non-empty sequence of Tensors")
    if not all(isinstance(tensor, _tensor_type()) for tensor in tensors):
        raise TypeError("cat expects a sequence containing only Tensors")
    axis = normalize_dims(dim, tensors[0].ndim)
    return apply(lambda *arrays: cp.concatenate(arrays, axis=axis), *tensors)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("stack expects a non-empty sequence of Tensors")
    if not all(isinstance(tensor, _tensor_type()) for tensor in tensors):
        raise TypeError("stack expects a sequence containing only Tensors")
    ndim = tensors[0].ndim + 1
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dim {dim} is out of range for stack result with {ndim} dims")
    return apply(lambda *arrays: cp.stack(arrays, axis=normalized), *tensors)


def require_floating(tensor: Tensor, operation: str) -> None:
    if not isinstance(tensor, _tensor_type()):
        raise TypeError(f"{operation} expects a Tensor")
    if tensor.dtype.kind != "f":
        raise TypeError(
            f"{operation} expects a floating-point Tensor, got {tensor.dtype}"
        )
