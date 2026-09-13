"""GPU-only Tensor type and creation functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np

from . import _ops
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

    @classmethod
    def _from_array(cls, array: cp.ndarray) -> Tensor:
        if not isinstance(array, cp.ndarray):
            raise TypeError("internal Tensor storage must be a cupy.ndarray")
        instance = cls.__new__(cls)
        instance.__array = array
        return instance

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
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device='{self.device}')"

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

        return _ops.apply(lambda array: array[unwrap(index)], self)

    def __add__(self, other: Any) -> Tensor:
        return _ops.binary(cp.add, self, other)

    def __radd__(self, other: Any) -> Tensor:
        return _ops.binary(cp.add, self, other)

    def __sub__(self, other: Any) -> Tensor:
        return _ops.binary(cp.subtract, self, other)

    def __rsub__(self, other: Any) -> Tensor:
        return _ops.binary(cp.subtract, other, self)

    def __mul__(self, other: Any) -> Tensor:
        return _ops.binary(cp.multiply, self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return _ops.binary(cp.multiply, self, other)

    def __truediv__(self, other: Any) -> Tensor:
        return _ops.binary(cp.true_divide, self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        return _ops.binary(cp.true_divide, other, self)

    def __pow__(self, other: Any) -> Tensor:
        return _ops.binary(cp.power, self, other)

    def __rpow__(self, other: Any) -> Tensor:
        return _ops.binary(cp.power, other, self)

    def __neg__(self) -> Tensor:
        return _ops.unary(cp.negative, self)

    def __abs__(self) -> Tensor:
        return _ops.unary(cp.abs, self)

    def __eq__(self, other: Any) -> Tensor:
        return _ops.binary(cp.equal, self, other)

    def __ne__(self, other: Any) -> Tensor:
        return _ops.binary(cp.not_equal, self, other)

    def __lt__(self, other: Any) -> Tensor:
        return _ops.binary(cp.less, self, other)

    def __le__(self, other: Any) -> Tensor:
        return _ops.binary(cp.less_equal, self, other)

    def __gt__(self, other: Any) -> Tensor:
        return _ops.binary(cp.greater, self, other)

    def __ge__(self, other: Any) -> Tensor:
        return _ops.binary(cp.greater_equal, self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return matmul(self, other)

    def exp(self) -> Tensor:
        return exp(self)

    def log(self) -> Tensor:
        return log(self)

    def sqrt(self) -> Tensor:
        return sqrt(self)

    def square(self) -> Tensor:
        return square(self)

    def sin(self) -> Tensor:
        return sin(self)

    def cos(self) -> Tensor:
        return cos(self)

    def clip(self, minimum: Any, maximum: Any) -> Tensor:
        return clip(self, minimum, maximum)

    def sum(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.sum, self, dim=dim, keepdim=keepdim)

    def mean(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.mean, self, dim=dim, keepdim=keepdim)

    def prod(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.prod, self, dim=dim, keepdim=keepdim)

    def max(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.max, self, dim=dim, keepdim=keepdim)

    def min(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.min, self, dim=dim, keepdim=keepdim)

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
        return _ops.reduction(cp.var, self, dim=dim, keepdim=keepdim, ddof=correction)

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
        return _ops.reduction(cp.std, self, dim=dim, keepdim=keepdim, ddof=correction)

    def argmax(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.argmax, self, dim=dim, keepdim=keepdim)

    def argmin(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> Tensor:
        return _ops.reduction(cp.argmin, self, dim=dim, keepdim=keepdim)

    def reshape(self, *shape: ShapeLike) -> Tensor:
        normalized = _shape_from_args(shape)
        return _ops.apply(lambda array: array.reshape(normalized), self)

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
        return _ops.apply(lambda array: cp.squeeze(array, axis=axes), self)

    def unsqueeze(self, dim: int) -> Tensor:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        ndim = self.ndim + 1
        axis = dim + ndim if dim < 0 else dim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"dim {dim} is out of range for unsqueeze")
        return _ops.apply(lambda array: cp.expand_dims(array, axis=axis), self)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        axis0 = _ops.normalize_dims(dim0, self.ndim)
        axis1 = _ops.normalize_dims(dim1, self.ndim)
        assert isinstance(axis0, int) and isinstance(axis1, int)
        return _ops.apply(lambda array: cp.swapaxes(array, axis0, axis1), self)

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
        return _ops.apply(lambda array: cp.transpose(array, axes), self)

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
) -> Tensor:
    return Tensor(data, dtype=dtype, device=device)


def empty(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _create(cp.empty, shape, dtype, device)


def zeros(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _create(cp.zeros, shape, dtype, device)


def ones(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _create(cp.ones, shape, dtype, device)


def full(
    shape: ShapeLike,
    fill_value: Any,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _create(cp.full, (shape,), dtype, device, fill_value=fill_value)


def _create(
    operation: Any,
    shape: tuple[ShapeLike, ...],
    dtype: DTypeLike,
    device: str | int,
    **kwargs: Any,
) -> Tensor:
    device_index = parse_device(device)
    normalized = _shape_from_args(shape)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(operation(normalized, dtype=dtype, **kwargs))


def arange(
    start: float,
    end: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    device_index = parse_device(device)
    if end is None:
        start, end = 0, start
    with cp.cuda.Device(device_index):
        return Tensor._from_array(cp.arange(start, end, step, dtype=dtype))


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    device_index = parse_device(device)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(cp.linspace(start, end, steps, dtype=dtype))


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    device_index = parse_device(device)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(cp.eye(n, m, dtype=dtype))


def rand(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _random_create(cp.random.random, shape, dtype, device)


def randn(
    *shape: ShapeLike,
    dtype: DTypeLike = cp.float32,
    device: str | int = "cuda:0",
) -> Tensor:
    return _random_create(cp.random.standard_normal, shape, dtype, device)


def _random_create(
    operation: Any,
    shape: tuple[ShapeLike, ...],
    dtype: DTypeLike,
    device: str | int,
) -> Tensor:
    device_index = parse_device(device)
    normalized = _shape_from_args(shape)
    with cp.cuda.Device(device_index):
        return Tensor._from_array(operation(normalized, dtype=dtype))


def _like(
    operation: Any,
    source: Tensor,
    dtype: DTypeLike | None,
    device: str | int | None,
) -> Tensor:
    if not isinstance(source, Tensor):
        raise TypeError("like creation expects a Tensor")
    device_index = source._device_index if device is None else parse_device(device)
    resolved_dtype = source.dtype if dtype is None else dtype
    with cp.cuda.Device(device_index):
        return Tensor._from_array(operation(source.shape, dtype=resolved_dtype))


def empty_like(
    source: Tensor, *, dtype: DTypeLike | None = None, device: str | int | None = None
) -> Tensor:
    return _like(cp.empty, source, dtype, device)


def zeros_like(
    source: Tensor, *, dtype: DTypeLike | None = None, device: str | int | None = None
) -> Tensor:
    return _like(cp.zeros, source, dtype, device)


def ones_like(
    source: Tensor, *, dtype: DTypeLike | None = None, device: str | int | None = None
) -> Tensor:
    return _like(cp.ones, source, dtype, device)


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
    return _ops.unary(cp.exp, input)


def log(input: Tensor) -> Tensor:
    return _ops.unary(cp.log, input)


def sqrt(input: Tensor) -> Tensor:
    return _ops.unary(cp.sqrt, input)


def square(input: Tensor) -> Tensor:
    return _ops.unary(cp.square, input)


def sin(input: Tensor) -> Tensor:
    return _ops.unary(cp.sin, input)


def cos(input: Tensor) -> Tensor:
    return _ops.unary(cp.cos, input)


def maximum(left: Tensor, right: Any) -> Tensor:
    return _ops.binary(cp.maximum, left, right)


def minimum(left: Tensor, right: Any) -> Tensor:
    return _ops.binary(cp.minimum, left, right)


def clip(input: Tensor, minimum: Any, maximum: Any) -> Tensor:
    return _ops.apply(cp.clip, input, minimum, maximum)


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
    return _ops.matrix_binary("matmul", cp.matmul, left, right)


def dot(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary("dot", cp.dot, left, right, required_ndim=1)


def mm(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary("mm", cp.matmul, left, right, required_ndim=2)


def bmm(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary(
        "bmm", cp.matmul, left, right, required_ndim=3, matching_batch=True
    )


def outer(left: Tensor, right: Tensor) -> Tensor:
    return _ops.matrix_binary("outer", cp.outer, left, right, required_ndim=1)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.concatenate(tensors, dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return _ops.stack(tensors, dim=dim)
