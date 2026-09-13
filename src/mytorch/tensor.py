"""GPU-only Tensor type and creation functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np

from . import _autograd, _ops
from ._device import format_device, parse_device

DTypeLike = Any
ShapeLike = int | tuple[int, ...] | list[int]


def _shape_from_args(
    shape: tuple[ShapeLike, ...], *, allow_infer: bool = False
) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        result = tuple(shape[0])
    else:
        result = tuple(shape)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in result
    ):
        raise TypeError("shape dimensions must be integers")
    minimum = -1 if allow_infer else 0
    if any(value < minimum for value in result) or (
        allow_infer and result.count(-1) > 1
    ):
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

    def to(
        self,
        device: str | int | None = None,
        dtype: DTypeLike | None = None,
    ) -> Tensor:
        """Copy or cast a Tensor without ever staging through CPU memory."""
        target_device = self._device_index if device is None else parse_device(device)
        target_dtype = self.dtype if dtype is None else cp.dtype(dtype)
        if target_device == self._device_index and target_dtype == self.dtype:
            return self
        with cp.cuda.Device(target_device):
            result = cp.asarray(self.__array, dtype=target_dtype)
            if result is self.__array:
                result = result.copy()
        requires_grad = (
            self.requires_grad
            and target_dtype.kind == "f"
            and _autograd.is_grad_enabled()
        )
        if not requires_grad:
            return Tensor._from_array(result)

        source_device = self._device_index

        def backward(gradient: cp.ndarray):
            with cp.cuda.Device(source_device):
                return (cp.asarray(gradient, dtype=self.dtype),)

        node = _autograd.Node(
            name="to",
            parents=(self,),
            backward_fn=backward,
            versions=(self._version,),
        )
        return Tensor._from_array(result, requires_grad=True, grad_fn=node)

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

    def _copy_from(self, source: Tensor | cp.ndarray) -> None:
        array = source._array if isinstance(source, Tensor) else source
        if not isinstance(array, cp.ndarray):
            raise TypeError("Tensor data must come from GPU storage")
        if array.shape != self.shape:
            raise ValueError(
                f"shape mismatch: expected {self.shape}, got {array.shape}"
            )
        with cp.cuda.Device(self._device_index):
            self.__array[...] = cp.asarray(array, dtype=self.dtype)
        self._version += 1

    def _replace_array(self, array: cp.ndarray) -> None:
        if not isinstance(array, cp.ndarray):
            raise TypeError("internal Tensor storage must be a cupy.ndarray")
        if self.requires_grad and array.dtype.kind != "f":
            raise TypeError("a trainable Tensor must remain floating point")
        self.__array = array
        self._grad = None
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

    def rsqrt(self) -> Tensor:
        return rsqrt(self)

    def round(self) -> Tensor:
        return round(self)

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

    def contiguous(self) -> Tensor:
        return contiguous(self)

    def expand(self, *shape: ShapeLike) -> Tensor:
        return expand(self, *shape)

    def repeat(self, *repeats: ShapeLike) -> Tensor:
        return repeat(self, *repeats)

    def masked_fill(self, mask: Tensor, value: Any) -> Tensor:
        return masked_fill(self, mask, value)

    def pad(
        self,
        pad_width: Sequence[int],
        mode: str = "constant",
        value: float = 0.0,
    ) -> Tensor:
        return pad(self, pad_width, mode=mode, value=value)

    def gather(self, dim: int, index: Tensor) -> Tensor:
        return gather(self, dim, index)

    def scatter_add(self, dim: int, index: Tensor, source: Tensor) -> Tensor:
        return scatter_add(self, dim, index, source)

    def topk(
        self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[Tensor, Tensor]:
        return topk(self, k, dim=dim, largest=largest, sorted=sorted)

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
        normalized = _shape_from_args(shape, allow_infer=True)
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


from ._tensor import creation as _creation  # noqa: E402
from ._tensor import elementwise as _elementwise  # noqa: E402
from ._tensor import linalg as _linalg  # noqa: E402
from ._tensor import reductions as _reductions  # noqa: E402
from ._tensor import shape as _shape  # noqa: E402

tensor = _creation.tensor
empty = _creation.empty
zeros = _creation.zeros
ones = _creation.ones
full = _creation.full
arange = _creation.arange
linspace = _creation.linspace
eye = _creation.eye
rand = _creation.rand
randn = _creation.randn
empty_like = _creation.empty_like
zeros_like = _creation.zeros_like
ones_like = _creation.ones_like
manual_seed = _creation.manual_seed

exp = _elementwise.exp
log = _elementwise.log
log1p = _elementwise.log1p
rsqrt = _elementwise.rsqrt
round = _elementwise.round
sqrt = _elementwise.sqrt
square = _elementwise.square
sin = _elementwise.sin
cos = _elementwise.cos
sign = _elementwise.sign
where = _elementwise.where
maximum = _elementwise.maximum
minimum = _elementwise.minimum
clip = _elementwise.clip
abs = _elementwise.abs

contiguous = _shape.contiguous
expand = _shape.expand
repeat = _shape.repeat
pad = _shape.pad
masked_fill = _shape.masked_fill
gather = _shape.gather
scatter_add = _shape.scatter_add
topk = _shape.topk
split = _shape.split
chunk = _shape.chunk
cat = _shape.cat
stack = _shape.stack

logaddexp = _reductions.logaddexp
logsumexp = _reductions.logsumexp
norm = _reductions.norm
normalize = _reductions.normalize
sum = _reductions.sum
mean = _reductions.mean
prod = _reductions.prod
max = _reductions.max
min = _reductions.min
var = _reductions.var
std = _reductions.std
argmax = _reductions.argmax
argmin = _reductions.argmin

matmul = _linalg.matmul
dot = _linalg.dot
mm = _linalg.mm
bmm = _linalg.bmm
outer = _linalg.outer

__all__ = [
    "Tensor",
    "abs",
    "arange",
    "argmax",
    "argmin",
    "bmm",
    "cat",
    "chunk",
    "clip",
    "contiguous",
    "cos",
    "dot",
    "empty",
    "empty_like",
    "exp",
    "expand",
    "eye",
    "full",
    "gather",
    "linspace",
    "log",
    "log1p",
    "logaddexp",
    "logsumexp",
    "manual_seed",
    "masked_fill",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mm",
    "norm",
    "normalize",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "prod",
    "rand",
    "randn",
    "repeat",
    "round",
    "rsqrt",
    "scatter_add",
    "sign",
    "sin",
    "split",
    "sqrt",
    "square",
    "stack",
    "std",
    "sum",
    "tensor",
    "topk",
    "var",
    "where",
    "zeros",
    "zeros_like",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
