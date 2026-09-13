"""Tensor reductions and normalization operations."""

from __future__ import annotations

from typing import Any

import cupy as cp
from cupyx.scipy.special import logsumexp as cp_logsumexp

from mytorch import _ops
from mytorch.tensor import Tensor, _expand_reduction_gradient

from .elementwise import maximum


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
