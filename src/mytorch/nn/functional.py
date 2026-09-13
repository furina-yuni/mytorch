"""Numerically stable, out-of-place activation functions."""

from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.special import erf, expit

from mytorch import _ops
from mytorch.tensor import Tensor


def relu(input: Tensor) -> Tensor:
    _ops.require_floating(input, "relu")
    return _ops.apply(lambda array: _same_dtype(cp.maximum(array, 0), array), input)


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    _ops.require_floating(input, "leaky_relu")
    if not isinstance(negative_slope, (int, float)) or isinstance(negative_slope, bool):
        raise TypeError("negative_slope must be a real number")
    return _ops.apply(
        lambda array: _same_dtype(
            cp.where(array >= 0, array, negative_slope * array), array
        ),
        input,
    )


def sigmoid(input: Tensor) -> Tensor:
    _ops.require_floating(input, "sigmoid")
    return _ops.apply(lambda array: _same_dtype(expit(array), array), input)


def tanh(input: Tensor) -> Tensor:
    _ops.require_floating(input, "tanh")
    return _ops.apply(lambda array: _same_dtype(cp.tanh(array), array), input)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    _ops.require_floating(input, "softmax")
    axis = _single_dim(dim, input, "softmax")

    def forward(array: cp.ndarray) -> cp.ndarray:
        shifted = array - cp.max(array, axis=axis, keepdims=True)
        exponentials = cp.exp(shifted)
        result = exponentials / cp.sum(exponentials, axis=axis, keepdims=True)
        return _same_dtype(result, array)

    return _ops.apply(forward, input)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    _ops.require_floating(input, "log_softmax")
    axis = _single_dim(dim, input, "log_softmax")

    def forward(array: cp.ndarray) -> cp.ndarray:
        shifted = array - cp.max(array, axis=axis, keepdims=True)
        normalizer = cp.log(cp.sum(cp.exp(shifted), axis=axis, keepdims=True))
        return _same_dtype(shifted - normalizer, array)

    return _ops.apply(forward, input)


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    _ops.require_floating(input, "gelu")
    if approximate == "none":
        return _ops.apply(
            lambda array: _same_dtype(
                0.5 * array * (1.0 + erf(array / math.sqrt(2.0))), array
            ),
            input,
        )
    if approximate == "tanh":
        coefficient = math.sqrt(2.0 / math.pi)
        return _ops.apply(
            lambda array: _same_dtype(
                0.5
                * array
                * (1.0 + cp.tanh(coefficient * (array + 0.044715 * array**3))),
                array,
            ),
            input,
        )
    raise ValueError("approximate must be either 'none' or 'tanh'")


def silu(input: Tensor) -> Tensor:
    _ops.require_floating(input, "silu")
    return _ops.apply(lambda array: _same_dtype(array * expit(array), array), input)


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    _ops.require_floating(input, "softplus")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta <= 0:
        raise ValueError("beta must be a positive real number")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a real number")

    def forward(array: cp.ndarray) -> cp.ndarray:
        scaled = beta * array
        stable = cp.logaddexp(0, scaled) / beta
        return _same_dtype(cp.where(scaled > threshold, array, stable), array)

    return _ops.apply(forward, input)


def _single_dim(dim: int, input: Tensor, operation: str) -> int:
    normalized = _ops.normalize_dims(dim, input.ndim)
    if not isinstance(normalized, int):
        raise TypeError(f"{operation} dim must be a single integer")
    return normalized


def _same_dtype(result: cp.ndarray, input: cp.ndarray) -> cp.ndarray:
    return result.astype(input.dtype, copy=False)
