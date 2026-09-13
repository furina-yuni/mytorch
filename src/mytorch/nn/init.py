"""GPU parameter initialization utilities."""

from __future__ import annotations

import math

import cupy as cp

from mytorch import _random
from mytorch.tensor import Tensor


def _write(tensor: Tensor, values: cp.ndarray) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("initializer expects a Tensor")
    tensor._copy_from(values.astype(tensor.dtype, copy=False))
    return tensor


def constant_(tensor: Tensor, value: float) -> Tensor:
    with cp.cuda.Device(tensor._device_index):
        return _write(tensor, cp.full(tensor.shape, value, dtype=tensor.dtype))


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    if a > b:
        raise ValueError("uniform_ expects a <= b")
    with cp.cuda.Device(tensor._device_index):
        return _write(
            tensor,
            _random.uniform(a, b, tensor.shape, device=tensor._device_index),
        )


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    if std < 0:
        raise ValueError("normal_ expects a non-negative std")
    with cp.cuda.Device(tensor._device_index):
        return _write(
            tensor,
            _random.normal(mean, std, tensor.shape, device=tensor._device_index),
        )


def _fan_in_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("fan calculation requires at least two dimensions")
    receptive = math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1
    return tensor.shape[1] * receptive, tensor.shape[0] * receptive


def _gain(nonlinearity: str, a: float = 0.0) -> float:
    if nonlinearity in {"linear", "sigmoid"}:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1.0 + a * a))
    raise ValueError(f"unsupported nonlinearity: {nonlinearity}")


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _fan_in_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _fan_in_out(tensor)
    return normal_(tensor, 0.0, gain * math.sqrt(2.0 / (fan_in + fan_out)))


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else None
    if fan is None:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    bound = math.sqrt(3.0) * _gain(nonlinearity, a) / math.sqrt(fan)
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else None
    if fan is None:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    return normal_(tensor, 0.0, _gain(nonlinearity, a) / math.sqrt(fan))
