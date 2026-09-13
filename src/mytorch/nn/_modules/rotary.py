"""Rotary positional embedding support."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch._device import parse_device
from mytorch.tensor import Tensor, cat, tensor

from .base import Module


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("rotary dimension must be a positive even integer")
        self.dim = dim
        self.base = base
        self.dtype = dtype
        self.device = device
        self._build(max_seq_len)

    def _build(self, length: int) -> None:
        with cp.cuda.Device(parse_device(self.device)):
            inverse = 1 / (self.base ** (cp.arange(0, self.dim, 2) / self.dim))
            angles = cp.arange(length)[:, None] * inverse[None, :]
            angles = cp.repeat(angles, 2, axis=-1).astype(self.dtype)
        self.register_buffer(
            "cos_cached", tensor(cp.cos(angles), device=self.device), False
        )
        self.register_buffer(
            "sin_cached", tensor(cp.sin(angles), device=self.device), False
        )

    def to(self, device: str | int | None = None, dtype: Any = None) -> RotaryEmbedding:
        super().to(device=device, dtype=dtype)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def forward(
        self, input: Tensor, seq_len: int | None = None
    ) -> tuple[Tensor, Tensor]:
        length = input.shape[-2] if seq_len is None else seq_len
        if length > self.cos_cached.shape[0]:
            self._build(length)
        return self.cos_cached[:length], self.sin_cached[:length]


def apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    def rotate(value: Tensor) -> Tensor:
        even, odd = value[..., ::2], value[..., 1::2]
        return stack_rotary(-odd, even)

    return query * cosine + rotate(query) * sine, key * cosine + rotate(key) * sine


def stack_rotary(even: Tensor, odd: Tensor) -> Tensor:
    return cat((even.unsqueeze(-1), odd.unsqueeze(-1)), -1).flatten(-2, -1)
