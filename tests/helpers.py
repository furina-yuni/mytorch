"""Shared numerical assertions for GPU tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import mytorch as mt


def finite_difference(
    function: Callable[[mt.Tensor], mt.Tensor],
    values: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute a central finite-difference gradient for a scalarized function."""
    result = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_result = function(mt.tensor(plus))
        minus_result = function(mt.tensor(minus))
        result[index] = (plus_result.sum().item() - minus_result.sum().item()) / (
            2 * epsilon
        )
    return result
