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
BackwardOperation = Callable[
    [cp.ndarray, cp.ndarray, tuple[Any, ...]], Sequence[cp.ndarray | None]
]


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


def apply(
    operation: ArrayOperation,
    *inputs: Any,
    backward: BackwardOperation | None = None,
    name: str | None = None,
    differentiable: bool = True,
    **kwargs: Any,
) -> Tensor:
    """Run one forward operation and optionally attach its reverse-mode VJP."""
    from . import _autograd

    tensor_type = _tensor_type()
    device = _device_for(inputs)
    arrays = [_unwrap(value, device) for value in inputs]
    from .amp.autocast_mode import _cast_arrays
    from .profiler.profiler import _record_operation

    arrays = _cast_arrays(arrays, name or getattr(operation, "__name__", None))
    with cp.cuda.Device(device):
        operation_name = name or getattr(operation, "__name__", "operation")
        result = _record_operation(
            operation_name,
            device,
            arrays,
            lambda: operation(*arrays, **kwargs),
        )
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
    tracked = [
        (index, value)
        for index, value in enumerate(inputs)
        if isinstance(value, tensor_type) and value.requires_grad
    ]
    requires_grad = bool(
        tracked
        and differentiable
        and result.dtype.kind == "f"
        and _autograd.is_grad_enabled()
    )
    if not requires_grad:
        return tensor_type._from_array(result)
    if backward is None:
        raise NotImplementedError(
            f"backward is not implemented for {name or operation.__name__}"
        )

    tracked_indices = tuple(index for index, _ in tracked)
    parents = tuple(value for _, value in tracked)
    saved_arrays = tuple(arrays)

    def vjp(gradient: cp.ndarray) -> Sequence[cp.ndarray | None]:
        all_gradients = backward(gradient, result, saved_arrays)
        if len(all_gradients) != len(inputs):
            raise RuntimeError("an operation returned the wrong number of gradients")
        return tuple(all_gradients[index] for index in tracked_indices)

    node = _autograd.Node(
        name=name or getattr(operation, "__name__", "operation"),
        parents=parents,
        backward_fn=vjp,
        versions=tuple(parent._version for parent in parents),
        forward_trace=_autograd.capture_forward_trace(),
    )
    return tensor_type._from_array(result, requires_grad=True, grad_fn=node)


def binary(
    operation: ArrayOperation,
    left: Any,
    right: Any,
    *,
    backward: BackwardOperation | None = None,
    name: str | None = None,
    differentiable: bool = True,
) -> Tensor:
    return apply(
        operation,
        left,
        right,
        backward=backward,
        name=name,
        differentiable=differentiable,
    )


def unary(
    operation: ArrayOperation,
    tensor: Tensor,
    *,
    backward: BackwardOperation | None = None,
    name: str | None = None,
) -> Tensor:
    return apply(operation, tensor, backward=backward, name=name)


def reduction(
    operation: ArrayOperation,
    tensor: Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    backward: BackwardOperation | None = None,
    name: str | None = None,
    differentiable: bool = True,
    **kwargs: Any,
) -> Tensor:
    axes = normalize_dims(dim, tensor.ndim)
    return apply(
        operation,
        tensor,
        axis=axes,
        keepdims=keepdim,
        backward=backward,
        name=name,
        differentiable=differentiable,
        **kwargs,
    )


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
    backward: BackwardOperation | None = None,
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
        return apply(operation, left, right, backward=backward, name=name)
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
    sizes = tuple(tensor.shape[axis] for tensor in tensors)

    def backward(gradient, _result, _arrays):
        boundaries = []
        current = 0
        for size in sizes[:-1]:
            current += size
            boundaries.append(current)
        return tuple(cp.split(gradient, boundaries, axis=axis))

    return apply(
        lambda *arrays: cp.concatenate(arrays, axis=axis),
        *tensors,
        backward=backward,
        name="cat",
    )


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("stack expects a non-empty sequence of Tensors")
    if not all(isinstance(tensor, _tensor_type()) for tensor in tensors):
        raise TypeError("stack expects a sequence containing only Tensors")
    ndim = tensors[0].ndim + 1
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dim {dim} is out of range for stack result with {ndim} dims")

    def backward(gradient, _result, arrays):
        return tuple(
            cp.take(gradient, index, axis=normalized) for index in range(len(arrays))
        )

    return apply(
        lambda *arrays: cp.stack(arrays, axis=normalized),
        *tensors,
        backward=backward,
        name="stack",
    )


def require_floating(tensor: Tensor, operation: str) -> None:
    if not isinstance(tensor, _tensor_type()):
        raise TypeError(f"{operation} expects a Tensor")
    if tensor.dtype.kind != "f":
        raise TypeError(
            f"{operation} expects a floating-point Tensor, got {tensor.dtype}"
        )
