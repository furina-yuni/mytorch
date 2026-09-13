"""Tensor linear-algebra operations."""

from __future__ import annotations

import cupy as cp

from mytorch import _ops
from mytorch.tensor import Tensor, _matmul_backward


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
