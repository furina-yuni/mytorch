"""Context-local CUDA autocast state and operation policies."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any

import cupy as cp

_enabled: ContextVar[bool] = ContextVar("mytorch_autocast_enabled", default=False)
_dtype: ContextVar[cp.dtype | None] = ContextVar("mytorch_autocast_dtype", default=None)

_LOW_PRECISION = {
    "bilinear",
    "bmm",
    "dot",
    "linear",
    "matmul",
    "mm",
    "scaled_dot_product_attention",
}
_FLOAT32 = {
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "exp",
    "log",
    "log1p",
    "log_softmax",
    "logaddexp",
    "logsumexp",
    "mean",
    "multi_margin_loss",
    "nll_loss",
    "norm",
    "prod",
    "reciprocal_sqrt",
    "safe_softmax",
    "softmax",
    "sqrt",
    "std",
    "sum",
    "var",
}


def is_autocast_enabled() -> bool:
    return _enabled.get()


def get_autocast_dtype() -> cp.dtype:
    return _dtype.get() or cp.dtype(cp.float16)


def _policy(name: str | None) -> str:
    operation = name or ""
    if operation in _LOW_PRECISION or operation.startswith("conv"):
        return "lower"
    if operation in _FLOAT32 or operation.endswith("_loss"):
        return "float32"
    return "promote"


def _cast_arrays(arrays: list[Any], name: str | None) -> list[Any]:
    if not is_autocast_enabled():
        return arrays
    policy = _policy(name)
    if policy == "lower":
        target = get_autocast_dtype()
        return [
            value.astype(target, copy=False)
            if isinstance(value, cp.ndarray) and value.dtype == cp.float32
            else value
            for value in arrays
        ]
    if policy == "float32":
        return [
            value.astype(cp.float32, copy=False)
            if isinstance(value, cp.ndarray) and value.dtype == cp.float16
            else value
            for value in arrays
        ]
    return arrays


class autocast(AbstractContextManager["autocast"]):
    """Run eligible CUDA operations in float16 while preserving stable ops."""

    def __init__(self, dtype: Any = cp.float16, enabled: bool = True) -> None:
        resolved = cp.dtype(dtype)
        if resolved != cp.dtype(cp.float16):
            raise TypeError("MyTorch autocast currently supports only float16")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.dtype = resolved
        self.enabled = enabled
        self._enabled_token: Token[bool] | None = None
        self._dtype_token: Token[cp.dtype | None] | None = None

    def __enter__(self) -> autocast:
        self._dtype_token = _dtype.set(self.dtype)
        self._enabled_token = _enabled.set(self.enabled)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self._enabled_token is None or self._dtype_token is None:
            raise RuntimeError("autocast context was not entered")
        _enabled.reset(self._enabled_token)
        _dtype.reset(self._dtype_token)
        self._enabled_token = None
        self._dtype_token = None


__all__ = ["autocast", "get_autocast_dtype", "is_autocast_enabled"]
