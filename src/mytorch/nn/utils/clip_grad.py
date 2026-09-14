"""Gradient clipping utilities for dense CUDA Parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import cupy as cp

from mytorch.tensor import Tensor


def _gradients(parameters: Tensor | Iterable[Tensor]) -> list[Tensor]:
    values = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("parameters must contain only Tensor values")
    gradients = [value.grad for value in values if value.grad is not None]
    if gradients:
        device = gradients[0]._device_index
        if any(gradient._device_index != device for gradient in gradients[1:]):
            raise ValueError("all gradients must be on the same CUDA device")
    return gradients


def clip_grad_norm_(
    parameters: Tensor | Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    """Clip the aggregate gradient norm and return its value before clipping."""
    if not isinstance(error_if_nonfinite, bool):
        raise TypeError("error_if_nonfinite must be a bool")
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if max_norm < 0 or math.isnan(max_norm):
        raise ValueError("max_norm must be non-negative")
    if norm_type <= 0 or math.isnan(norm_type):
        raise ValueError("norm_type must be positive")
    gradients = _gradients(parameters)
    if not gradients:
        return Tensor._from_array(cp.asarray(0.0, dtype=cp.float32))

    with cp.cuda.Device(gradients[0]._device_index):
        if math.isinf(norm_type):
            total_norm = cp.max(
                cp.stack([cp.max(cp.abs(gradient._array)) for gradient in gradients])
            )
        else:
            powered = cp.stack(
                [
                    cp.sum(cp.abs(gradient._array).astype(cp.float64) ** norm_type)
                    for gradient in gradients
                ]
            ).sum()
            total_norm = powered ** (1.0 / norm_type)
        finite = bool(cp.isfinite(total_norm).item())
        if error_if_nonfinite and not finite:
            raise RuntimeError("the total gradient norm is non-finite")
        coefficient = max_norm / (float(total_norm.item()) + 1e-6)
        if coefficient < 1.0:
            for gradient in gradients:
                gradient._array[...] *= coefficient
        return Tensor._from_array(cp.asarray(total_norm))


def clip_grad_value_(parameters: Tensor | Iterable[Tensor], clip_value: float) -> None:
    """Clamp every dense gradient element to ``[-clip_value, clip_value]``."""
    clip_value = float(clip_value)
    if clip_value < 0 or not math.isfinite(clip_value):
        raise ValueError("clip_value must be a finite non-negative number")
    for gradient in _gradients(parameters):
        gradient._array[...] = cp.clip(gradient._array, -clip_value, clip_value)
