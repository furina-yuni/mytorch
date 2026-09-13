"""Elementwise Tensor operations."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch import _ops
from mytorch.tensor import Tensor


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


def rsqrt(input: Tensor) -> Tensor:
    return _ops.unary(
        cp.reciprocal,
        sqrt(input),
        backward=lambda gradient, _result, arrays: (-gradient / (arrays[0] ** 2),),
        name="reciprocal_sqrt",
    )


def round(input: Tensor) -> Tensor:
    """Round values; like PyTorch, its mathematical gradient is zero."""
    return _ops.unary(
        cp.round,
        input,
        backward=lambda gradient, _result, arrays: (cp.zeros_like(arrays[0]),),
        name="round",
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
