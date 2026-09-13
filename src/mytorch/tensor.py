"""GPU-only Tensor type and creation functions."""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np
from cupyx.scipy.special import logsumexp as cp_logsumexp

from . import _autograd, _ops
from ._device import format_device, parse_device

DTypeLike = Any
ShapeLike = int | tuple[int, ...] | list[int]


def _shape_from_args(shape: tuple[ShapeLike, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        result = tuple(shape[0])
    else:
        result = tuple(shape)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in result
    ):
        raise TypeError("shape dimensions must be integers")
    if any(value < 0 for value in result):
        raise ValueError(f"shape dimensions must be non-negative, got {result}")
    return result


def _default_dtype(data: Any, dtype: DTypeLike | None) -> DTypeLike | None:
    if dtype is not None:
        return dtype
    if isinstance(data, Tensor):
        return None
    if isinstance(data, (cp.ndarray, np.ndarray)):
        return None
    return cp.float32


def _add_backward(gradient, _result, _arrays):
    return gradient, gradient


def _sub_backward(gradient, _result, _arrays):
    return gradient, -gradient


def _mul_backward(gradient, _result, arrays):
    left, right = arrays
    return gradient * right, gradient * left


def _div_backward(gradient, _result, arrays):
    left, right = arrays
    return gradient / right, -gradient * left / (right**2)


def _pow_backward(gradient, result, arrays):
    left, right = arrays
    left_gradient = gradient * right * cp.power(left, right - 1)
    right_gradient = gradient * result * cp.log(left)
    return left_gradient, right_gradient


def _expand_reduction_gradient(
    gradient: cp.ndarray,
    input_shape: tuple[int, ...],
    axes: int | tuple[int, ...] | None,
    keepdim: bool,
) -> cp.ndarray:
    if axes is None:
        normalized = tuple(range(len(input_shape)))
    elif isinstance(axes, int):
        normalized = (axes,)
    else:
        normalized = axes
    if not keepdim:
        for axis in sorted(normalized):
            gradient = cp.expand_dims(gradient, axis=axis)
    return cp.broadcast_to(gradient, input_shape)


def _matmul_backward(gradient, _result, arrays):
    left, right = arrays
    left_was_vector = left.ndim == 1
    right_was_vector = right.ndim == 1
    left_matrix = left[None, :] if left_was_vector else left
    right_matrix = right[:, None] if right_was_vector else right
    if left_was_vector and right_was_vector:
        grad_matrix = gradient.reshape(1, 1)
    elif left_was_vector:
        grad_matrix = cp.expand_dims(gradient, axis=-2)
    elif right_was_vector:
        grad_matrix = cp.expand_dims(gradient, axis=-1)
    else:
        grad_matrix = gradient
    grad_left = cp.matmul(grad_matrix, cp.swapaxes(right_matrix, -1, -2))
    grad_right = cp.matmul(cp.swapaxes(left_matrix, -1, -2), grad_matrix)
    if left_was_vector:
        grad_left = cp.squeeze(grad_left, axis=-2)
    if right_was_vector:
        grad_right = cp.squeeze(grad_right, axis=-1)
    return grad_left, grad_right


class Tensor:
    """A thin, out-of-place wrapper around a CUDA-resident CuPy array."""

    __array_priority__ = 1000
    __hash__ = None

    def __init__(
        self,
        data: Any,
        *,
        dtype: DTypeLike | None = None,
        device: str | int = "cuda:0",
        requires_grad: bool = False,
    ) -> None:
        device_index = parse_device(device)
        resolved_dtype = _default_dtype(data, dtype)
        with cp.cuda.Device(device_index):
            if isinstance(data, Tensor):
                if data._device_index != device_index:
                    raise ValueError(
                        "Tensor construction does not implicitly copy devices"
                    )
                array = cp.array(data._array, dtype=resolved_dtype, copy=True)
            elif isinstance(data, cp.ndarray):
                if int(data.device.id) != device_index:
                    raise ValueError(
                        "CuPy array construction does not implicitly copy devices"
                    )
                array = cp.array(data, dtype=resolved_dtype, copy=True)
            else:
                array = cp.asarray(data, dtype=resolved_dtype)
        self.__array = array
        self._initialize_autograd(requires_grad=requires_grad)

    @classmethod
    def _from_array(
        cls,
        array: cp.ndarray,
        *,
        requires_grad: bool = False,
        grad_fn: _autograd.Node | None = None,
    ) -> Tensor:
        if not isinstance(array, cp.ndarray):
            raise TypeError("internal Tensor storage must be a cupy.ndarray")
        instance = cls.__new__(cls)
        instance.__array = array
        instance._initialize_autograd(requires_grad=requires_grad, grad_fn=grad_fn)
        return instance

    def _initialize_autograd(
        self,
        *,
        requires_grad: bool,
        grad_fn: _autograd.Node | None = None,
    ) -> None:
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if requires_grad and self.__array.dtype.kind != "f":
            raise TypeError("only floating-point Tensors can require gradients")
        self._requires_grad = requires_grad
        self._grad: Tensor | None = None
        self._grad_fn = grad_fn
        self._version = 0

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> Tensor | None:
        return self._grad

    @property
    def grad_fn(self) -> _autograd.Node | None:
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        if not self.is_leaf:
            raise RuntimeError("requires_grad_() can only change leaf Tensors")
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if requires_grad and self.dtype.kind != "f":
            raise TypeError("only floating-point Tensors can require gradients")
        self._requires_grad = requires_grad
        if not requires_grad:
            self._grad = None
        return self

    def detach(self) -> Tensor:
        return Tensor._from_array(self.__array)

    def backward(
        self, gradient: Tensor | None = None, *, retain_graph: bool = False
    ) -> None:
        if not isinstance(retain_graph, bool):
            raise TypeError("retain_graph must be a bool")
        _autograd.backward(self, gradient, retain_graph)

    def _accumulate_grad(self, gradient: cp.ndarray) -> None:
        with cp.cuda.Device(self._device_index):
            value = gradient.astype(self.dtype, copy=False)
            if self._grad is None:
                self._grad = Tensor._from_array(cp.array(value, copy=True))
            else:
                self._grad.__array += value

    def _clear_grad(self, *, set_to_none: bool = True) -> None:
        if set_to_none:
            self._grad = None
        elif self._grad is not None:
            self._grad.__array.fill(0)

    def _optimizer_update(self, update: cp.ndarray) -> None:
        self.__array += update.astype(self.dtype, copy=False)
        self._version += 1

    @property
    def _array(self) -> cp.ndarray:
        """Internal GPU array access; callers must not mutate it."""
        return self.__array

    @property
    def _device_index(self) -> int:
        return int(self.__array.device.id)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__array.shape

    @property
    def ndim(self) -> int:
        return self.__array.ndim

    @property
    def dtype(self) -> cp.dtype:
        return self.__array.dtype

    @property
    def device(self) -> str:
        return format_device(self._device_index)

    @property
    def T(self) -> Tensor:
        if self.ndim != 2:
            raise ValueError(
                f"T is only defined for 2D Tensors, got shape {self.shape}"
            )
        return self.transpose(0, 1)

    def size(self, dim: int | None = None) -> tuple[int, ...] | int:
        if dim is None:
            return self.shape
        axis = _ops.normalize_dims(dim, self.ndim)
        assert isinstance(axis, int)
        return self.shape[axis]

    def numel(self) -> int:
        return int(self.__array.size)

    def numpy(self) -> np.ndarray:
        return cp.asnumpy(self.__array)

    def item(self) -> Any:
        if self.numel() != 1:
            raise ValueError(
                f"item() requires one element, but Tensor has {self.numel()}"
            )
        return self.__array.item()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        del dtype, copy
        raise TypeError(
            "implicit CPU conversion is disabled; call tensor.numpy() explicitly"
        )

    def __repr__(self) -> str:
        grad = ", requires_grad=True" if self.requires_grad else ""
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"device='{self.device}'{grad})"
        )

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-dimensional Tensor")
        return self.shape[0]

    def __getitem__(self, index: Any) -> Tensor:
        def unwrap(item: Any) -> Any:
            if isinstance(item, Tensor):
                if item._device_index != self._device_index:
                    raise ValueError("index Tensor must be on the same CUDA device")
                return item._array
            if isinstance(item, tuple):
                return tuple(unwrap(part) for part in item)
            if item is None or item is Ellipsis or isinstance(item, (int, slice)):
                return item
            raise TypeError(
                "indices must be integers, slices, ellipsis, None, or GPU Tensors"
            )

        normalized_index = unwrap(index)

        def backward(gradient, _result, arrays):
            source = arrays[0]
            result = cp.zeros_like(source)
            cp.add.at(result, normalized_index, gradient)
            return (result,)

        return _ops.apply(
            lambda array: array[normalized_index],
            self,
            backward=backward,
            name="getitem",
        )

    def __add__(self, other: Any) -> Tensor:
        return _ops.binary(cp.add, self, other, backward=_add_backward, name="add")

    def __radd__(self, other: Any) -> Tensor:
        return _ops.binary(cp.add, self, other, backward=_add_backward, name="add")

    def __sub__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.subtract, self, other, backward=_sub_backward, name="subtract"
        )

    def __rsub__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.subtract, other, self, backward=_sub_backward, name="subtract"
        )

    def __mul__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.multiply, self, other, backward=_mul_backward, name="multiply"
        )

    def __rmul__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.multiply, self, other, backward=_mul_backward, name="multiply"
        )

    def __truediv__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.true_divide, self, other, backward=_div_backward, name="divide"
        )

    def __rtruediv__(self, other: Any) -> Tensor:
        return _ops.binary(
            cp.true_divide, other, self, backward=_div_backward, name="divide"
        )

    def __pow__(self, other: Any) -> Tensor:
        return _ops.binary(cp.power, self, other, backward=_pow_backward, name="power")

    def __rpow__(self, other: Any) -> Tensor:
        return _ops.binary(cp.power, other, self, backward=_pow_backward, name="power")

    def __neg__(self) -> Tensor:
        return _ops.unary(
            cp.negative,
            self,
            backward=lambda grad, _result, _arrays: (-grad,),
            name="negative",
        )

    def __abs__(self) -> Tensor:
        return _ops.unary(
            cp.abs,
            self,
            backward=lambda grad, _result, arrays: (grad * cp.sign(arrays[0]),),
            name="abs",
        )

    def __eq__(self, other: Any) -> Tensor:
        return _ops.binary(cp.equal, self, other, differentiable=False)

    def __ne__(self, other: Any) -> Tensor:
        return _ops.binary(cp.not_equal, self, other, differentiable=False)

    def __lt__(self, other: Any) -> Tensor:
        return _ops.binary(cp.less, self, other, differentiable=False)

    def __le__(self, other: Any) -> Tensor:
        return _ops.binary(cp.less_equal, self, other, differentiable=False)

    def __gt__(self, other: Any) -> Tensor:
        return _ops.binary(cp.greater, self, other, differentiable=False)

    def __ge__(self, other: Any) -> Tensor:
        return _ops.binary(cp.greater_equal, self, other, differentiable=False)

    def __matmul__(self, other: Tensor) -> Tensor:
        return matmul(self, other)

    def exp(self) -> Tensor:
        return exp(self)

    def log(self) -> Tensor:
        return log(self)

    def log1p(self) -> Tensor:
        return log1p(self)

    def logsumexp(self, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
        return logsumexp(self, dim=dim, keepdim=keepdim)

    def sqrt(self) -> Tensor:
        return sqrt(self)

    def square(self) -> Tensor:
        return square(self)

    def sin(self) -> Tensor:
        return sin(self)

    def cos(self) -> Tensor:
        return cos(self)

    def sign(self) -> Tensor:
        return sign(self)

    def norm(
        self,
        p: float = 2.0,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        return norm(self, p=p, dim=dim, keepdim=keepdim)

    def normalize(self, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
        return normalize(self, p=p, dim=dim, eps=eps)

    def clip(self, minimum: Any, maximum: Any) -> Tensor:
        return clip(self, minimum, maximum)

    def sum(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        axes = _ops.normalize_dims(dim, self.ndim)
        return _ops.reduction(
            cp.sum,
            self,
            dim=dim,
            keepdim=keepdim,
            backward=lambda grad, _result, arrays: (
                _expand_reduction_gradient(grad, arrays[0].shape, axes, keepdim),
            ),
            name="sum",
        )

    def mean(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        axes = _ops.normalize_dims(dim, self.ndim)
        if axes is None:
            count = self.numel()
        elif isinstance(axes, int):
            count = self.shape[axes]
        else:
            count = math.prod(self.shape[axis] for axis in axes)
        return _ops.reduction(
            cp.mean,
            self,
            dim=dim,
            keepdim=keepdim,
            backward=lambda grad, _result, arrays: (
                _expand_reduction_gradient(grad, arrays[0].shape, axes, keepdim)
                / count,
            ),
            name="mean",
        )

    def prod(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        axes = _ops.normalize_dims(dim, self.ndim)

        def backward(gradient, _result, arrays):
            source = arrays[0]
            normalized = (
                tuple(range(source.ndim))
                if axes is None
                else ((axes,) if isinstance(axes, int) else axes)
            )
            expanded = _expand_reduction_gradient(gradient, source.shape, axes, keepdim)
            zero_count = cp.sum(source == 0, axis=normalized, keepdims=True)
            nonzero_product = cp.prod(
                cp.where(source == 0, 1, source), axis=normalized, keepdims=True
            )
            safe = cp.where(source == 0, 1, source)
            derivative = cp.where(
                zero_count == 0,
                nonzero_product / safe,
                cp.where((zero_count == 1) & (source == 0), nonzero_product, 0),
            )
            return (expanded * derivative,)

        return _ops.reduction(
            cp.prod,
            self,
            dim=dim,
            keepdim=keepdim,
            backward=backward,
            name="prod",
        )

    def max(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return self._extreme_reduction(cp.max, dim, keepdim, "max")

    def min(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return self._extreme_reduction(cp.min, dim, keepdim, "min")

    def _extreme_reduction(
        self,
        operation: Any,
        dim: int | tuple[int, ...] | None,
        keepdim: bool,
        name: str,
    ) -> Tensor:
        axes = _ops.normalize_dims(dim, self.ndim)

        def backward(gradient, result, arrays):
            source = arrays[0]
            expanded_gradient = _expand_reduction_gradient(
                gradient, source.shape, axes, keepdim
            )
            expanded_result = _expand_reduction_gradient(
                result, source.shape, axes, keepdim
            )
            mask = source == expanded_result
            normalized = (
                tuple(range(source.ndim))
                if axes is None
                else ((axes,) if isinstance(axes, int) else axes)
            )
            ties = cp.sum(mask, axis=normalized, keepdims=True)
            return (expanded_gradient * mask / ties,)

        return _ops.reduction(
            operation,
            self,
            dim=dim,
            keepdim=keepdim,
            backward=backward,
            name=name,
        )

    def var(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        *,
        correction: int = 0,
    ) -> Tensor:
        if (
            not isinstance(correction, int)
            or isinstance(correction, bool)
            or correction < 0
        ):
            raise ValueError("correction must be a non-negative integer")
        axes = _ops.normalize_dims(dim, self.ndim)
        normalized = (
            tuple(range(self.ndim))
            if axes is None
            else ((axes,) if isinstance(axes, int) else axes)
        )
        count = math.prod(self.shape[axis] for axis in normalized)

        def backward(gradient, _result, arrays):
            source = arrays[0]
            mean = cp.mean(source, axis=normalized, keepdims=True)
            expanded = _expand_reduction_gradient(gradient, source.shape, axes, keepdim)
            return (expanded * 2 * (source - mean) / (count - correction),)

        return _ops.reduction(
            cp.var,
            self,
            dim=dim,
            keepdim=keepdim,
            ddof=correction,
            backward=backward,
            name="var",
        )

    def std(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        *,
        correction: int = 0,
    ) -> Tensor:
        if (
            not isinstance(correction, int)
            or isinstance(correction, bool)
            or correction < 0
        ):
            raise ValueError("correction must be a non-negative integer")
        axes = _ops.normalize_dims(dim, self.ndim)
        normalized = (
            tuple(range(self.ndim))
            if axes is None
            else ((axes,) if isinstance(axes, int) else axes)
        )
        count = math.prod(self.shape[axis] for axis in normalized)

        def backward(gradient, result, arrays):
            source = arrays[0]
            mean = cp.mean(source, axis=normalized, keepdims=True)
            expanded_gradient = _expand_reduction_gradient(
                gradient, source.shape, axes, keepdim
            )
            expanded_result = _expand_reduction_gradient(
                result, source.shape, axes, keepdim
            )
            derivative = cp.where(
                expanded_result == 0,
                0,
                (source - mean) / ((count - correction) * expanded_result),
            )
            return (expanded_gradient * derivative,)

        return _ops.reduction(
            cp.std,
            self,
            dim=dim,
            keepdim=keepdim,
            ddof=correction,
            backward=backward,
            name="std",
        )

    def argmax(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(
            cp.argmax,
            self,
            dim=dim,
            keepdim=keepdim,
            differentiable=False,
        )

    def argmin(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(
            cp.argmin,
            self,
            dim=dim,
            keepdim=keepdim,
            differentiable=False,
        )

    def reshape(self, *shape: ShapeLike) -> Tensor:
        normalized = _shape_from_args(shape)
        return _ops.apply(
            lambda array: array.reshape(normalized),
            self,
            backward=lambda grad, _result, arrays: (grad.reshape(arrays[0].shape),),
            name="reshape",
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        if self.ndim == 0:
            return self.reshape(1)
        start = _ops.normalize_dims(start_dim, self.ndim)
        end = _ops.normalize_dims(end_dim, self.ndim)
        assert isinstance(start, int) and isinstance(end, int)
        if start > end:
            raise ValueError("start_dim cannot come after end_dim")
        flat_size = math.prod(self.shape[start : end + 1])
        shape = self.shape[:start] + (flat_size,) + self.shape[end + 1 :]
        return self.reshape(shape)

    def squeeze(self, dim: int | tuple[int, ...] | None = None) -> Tensor:
        axes = _ops.normalize_dims(dim, self.ndim)
        return _ops.apply(
            lambda array: cp.squeeze(array, axis=axes),
            self,
            backward=lambda grad, _result, arrays: (grad.reshape(arrays[0].shape),),
            name="squeeze",
        )

    def unsqueeze(self, dim: int) -> Tensor:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        ndim = self.ndim + 1
        axis = dim + ndim if dim < 0 else dim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"dim {dim} is out of range for unsqueeze")
        return _ops.apply(
            lambda array: cp.expand_dims(array, axis=axis),
            self,
            backward=lambda grad, _result, _arrays: (cp.squeeze(grad, axis=axis),),
            name="unsqueeze",
        )

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        axis0 = _ops.normalize_dims(dim0, self.ndim)
        axis1 = _ops.normalize_dims(dim1, self.ndim)
        assert isinstance(axis0, int) and isinstance(axis1, int)
        return _ops.apply(
            lambda array: cp.swapaxes(array, axis0, axis1),
            self,
            backward=lambda grad, _result, _arrays: (cp.swapaxes(grad, axis0, axis1),),
            name="transpose",
        )

    def permute(self, *dims: int) -> Tensor:
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        try:
            axes = _ops.normalize_dims(tuple(dims), self.ndim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"permute requires each dimension exactly once, got {dims}"
            ) from exc
        assert isinstance(axes, tuple)
        if len(axes) != self.ndim or set(axes) != set(range(self.ndim)):
            raise ValueError(
                f"permute requires each dimension exactly once, got {dims}"
            )
        inverse = tuple(axes.index(axis) for axis in range(self.ndim))
        return _ops.apply(
            lambda array: cp.transpose(array, axes),
            self,
            backward=lambda grad, _result, _arrays: (cp.transpose(grad, inverse),),
            name="permute",
        )

    def split(
        self, split_size_or_sections: int | Sequence[int], dim: int = 0
    ) -> tuple[Tensor, ...]:
        return split(self, split_size_or_sections, dim=dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
        return chunk(self, chunks, dim=dim)

    def matmul(self, other: Tensor) -> Tensor:
        return matmul(self, other)

    def dot(self, other: Tensor) -> Tensor:
        return dot(self, other)

    def mm(self, other: Tensor) -> Tensor:
        return mm(self, other)

    def bmm(self, other: Tensor) -> Tensor:
        return bmm(self, other)

    def outer(self, other: Tensor) -> Tensor:
        return outer(self, other)

    def relu(self) -> Tensor:
        from .nn.functional import relu

        return relu(self)

    def sigmoid(self) -> Tensor:
        from .nn.functional import sigmoid

        return sigmoid(self)

    def tanh(self) -> Tensor:
        from .nn.functional import tanh

        return tanh(self)

    def softmax(self, dim: int = -1) -> Tensor:
        from .nn.functional import softmax

        return softmax(self, dim=dim)

    def log_softmax(self, dim: int = -1) -> Tensor:
        from .nn.functional import log_softmax

        return log_softmax(self, dim=dim)


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
    with cp.cuda.Device(device_index):
        return Tensor._from_array(
            operation(normalized, dtype=dtype), requires_grad=requires_grad
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


def exp(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.exp,
        input,
        backward=lambda grad, result, _arrays: (grad * result,),
        name="exp",
    )


def log(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.log,
        input,
        backward=lambda grad, _result, arrays: (grad / arrays[0],),
        name="log",
    )


def log1p(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.log1p,
        input,
        backward=lambda grad, _result, arrays: (grad / (1 + arrays[0]),),
        name="log1p",
    )


def logaddexp(left: Tensor, right: Any) -> Tensor:
    def backward(gradient, _result, arrays):
        first, second = arrays
        first_weight = 1 / (1 + cp.exp(second - first))
        return gradient * first_weight, gradient * (1 - first_weight)

    return _ops.binary(cp.logaddexp, left, right, backward=backward, name="logaddexp")


def logsumexp(
    input: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
) -> Tensor:
    axes = _ops.normalize_dims(dim, input.ndim)
    if axes is None:
        raise TypeError("logsumexp dim must be an int or tuple of ints")

    def backward(gradient, result, arrays):
        source = arrays[0]
        expanded_gradient = _expand_reduction_gradient(
            gradient, source.shape, axes, keepdim
        )
        expanded_result = _expand_reduction_gradient(
            result, source.shape, axes, keepdim
        )
        return (expanded_gradient * cp.exp(source - expanded_result),)

    return _ops.apply(
        lambda array: cp_logsumexp(array, axis=axes, keepdims=keepdim),
        input,
        backward=backward,
        name="logsumexp",
    )


def sqrt(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.sqrt,
        input,
        backward=lambda grad, result, _arrays: (grad * 0.5 / result,),
        name="sqrt",
    )


def square(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.square,
        input,
        backward=lambda grad, _result, arrays: (grad * 2 * arrays[0],),
        name="square",
    )


def sin(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.sin,
        input,
        backward=lambda grad, _result, arrays: (grad * cp.cos(arrays[0]),),
        name="sin",
    )


def cos(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.cos,
        input,
        backward=lambda grad, _result, arrays: (-grad * cp.sin(arrays[0]),),
        name="cos",
    )


def sign(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.sign,
        input,
        backward=lambda grad, _result, arrays: (cp.zeros_like(arrays[0]),),
        name="sign",
    )


def where(condition: Tensor, input: Any, other: Any) -> Tensor:
    if not isinstance(condition, Tensor) or condition.dtype.kind != "b":
        raise TypeError("where condition must be a boolean Tensor")

    def backward(gradient, _result, arrays):
        mask = arrays[0]
        return None, cp.where(mask, gradient, 0), cp.where(mask, 0, gradient)

    return _ops.apply(
        cp.where,
        condition,
        input,
        other,
        backward=backward,
        name="where",
    )


def norm(
    input: Tensor,
    p: float = 2.0,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    _ops.require_floating(input, "norm")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or p < 1:
        raise ValueError("p must be a real number greater than or equal to 1")
    axes = _ops.normalize_dims(dim, input.ndim)

    def forward(array):
        return cp.sum(cp.abs(array) ** p, axis=axes, keepdims=keepdim) ** (1.0 / p)

    def backward(gradient, result, arrays):
        source = arrays[0]
        expanded_gradient = _expand_reduction_gradient(
            gradient, source.shape, axes, keepdim
        )
        expanded_result = _expand_reduction_gradient(
            result, source.shape, axes, keepdim
        )
        safe_norm = cp.where(expanded_result == 0, 1, expanded_result)
        derivative = cp.sign(source) * cp.abs(source) ** (p - 1) * safe_norm ** (1 - p)
        derivative = cp.where(expanded_result == 0, 0, derivative)
        return (expanded_gradient * derivative,)

    return _ops.apply(forward, input, backward=backward, name="norm")


def normalize(
    input: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise ValueError("eps must be a positive real number")
    denominator = maximum(norm(input, p=p, dim=dim, keepdim=True), eps)
    return input / denominator


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


def maximum(left: Tensor, right: Any) -> Tensor:
    def backward(gradient, _result, arrays):
        first, second = arrays
        return (
            gradient * (first > second) + gradient * 0.5 * (first == second),
            gradient * (second > first) + gradient * 0.5 * (first == second),
        )

    return _ops.binary(cp.maximum, left, right, backward=backward, name="maximum")


def minimum(left: Tensor, right: Any) -> Tensor:
    def backward(gradient, _result, arrays):
        first, second = arrays
        return (
            gradient * (first < second) + gradient * 0.5 * (first == second),
            gradient * (second < first) + gradient * 0.5 * (first == second),
        )

    return _ops.binary(cp.minimum, left, right, backward=backward, name="minimum")


def clip(input: Tensor, minimum: Any, maximum: Any) -> Tensor:
    for bound in (minimum, maximum):
        if isinstance(bound, Tensor) and bound.requires_grad:
            raise ValueError("clip bounds cannot require gradients")

    def backward(gradient, _result, arrays):
        source, lower, upper = arrays
        return gradient * ((source >= lower) & (source <= upper)), None, None

    return _ops.apply(cp.clip, input, minimum, maximum, backward=backward, name="clip")


def abs(input: Tensor) -> Tensor:
    return input.__abs__()


def sum(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.sum(dim=dim, keepdim=keepdim)


def mean(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.mean(dim=dim, keepdim=keepdim)


def prod(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.prod(dim=dim, keepdim=keepdim)


def max(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.max(dim=dim, keepdim=keepdim)


def min(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.min(dim=dim, keepdim=keepdim)


def var(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    correction: int = 0,
) -> Tensor:
    return input.var(dim=dim, keepdim=keepdim, correction=correction)


def std(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    correction: int = 0,
) -> Tensor:
    return input.std(dim=dim, keepdim=keepdim, correction=correction)


def argmax(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.argmax(dim=dim, keepdim=keepdim)


def argmin(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    return input.argmin(dim=dim, keepdim=keepdim)


def matmul(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary(
        "matmul", cp.matmul, left, right, backward=_matmul_backward
    )


def dot(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary(
        "dot",
        cp.dot,
        left,
        right,
        required_ndim=1,
        backward=_matmul_backward,
    )


def mm(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary(
        "mm",
        cp.matmul,
        left,
        right,
        required_ndim=2,
        backward=_matmul_backward,
    )


def bmm(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary(
        "bmm",
        cp.matmul,
        left,
        right,
        required_ndim=3,
        matching_batch=True,
        backward=_matmul_backward,
    )


def outer(left: Tensor, right: Tensor) -> Tensor:
    def backward(gradient, _result, arrays):
        first, second = arrays
        return gradient @ second, first @ gradient

    return _ops.matrix_binary(
        "outer",
        cp.outer,
        left,
        right,
        required_ndim=1,
        backward=backward,
    )


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.concatenate(tensors, dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.stack(tensors, dim=dim)
