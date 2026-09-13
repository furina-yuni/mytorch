"""Shared validation helpers for neural-network modules."""

from collections.abc import Sequence


def positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def shape(value: int | Sequence[int]) -> tuple[int, ...]:
    result = (value,) if isinstance(value, int) else tuple(value)
    if not result or any(item <= 0 for item in result):
        raise ValueError("shape dimensions must be positive")
    return result
