"""Low-rank, LoRA, and gated feed-forward layers."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, full, rand, zeros

from ..functional import activations as A
from ..functional import layers as F
from .base import Module, Parameter
from .linear import Linear
from .regularization import Dropout


class LowRankLinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(
                "rank must be positive and no larger than either dimension"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        bound = 1 / math.sqrt(in_features)
        self.left = Parameter(
            (rand(rank, in_features, device=device, dtype=dtype) * 2 - 1) * bound
        )
        self.right = Parameter(
            (rand(out_features, rank, device=device, dtype=dtype) * 2 - 1)
            / math.sqrt(rank)
        )
        self.bias = (
            Parameter(zeros(out_features, device=device, dtype=dtype)) if bias else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(F.linear(input, self.left), self.right, self.bias)


class LoRALinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = Linear(in_features, out_features, bias, device=device, dtype=dtype)
        self.lora_a = Parameter(
            (rand(rank, in_features, device=device, dtype=dtype) * 2 - 1)
            / math.sqrt(in_features)
        )
        self.lora_b = Parameter(zeros(out_features, rank, device=device, dtype=dtype))
        self.adapter_dropout = Dropout(dropout)
        self.scaling = float(alpha) / rank
        self.register_buffer(
            "merged_state", full((1,), 0, device=device, dtype=cp.int8)
        )
        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

    def _delta(self) -> Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    @property
    def merged(self) -> bool:
        return bool(self.merged_state.item())

    def merge(self) -> LoRALinear:
        if self.merged:
            return self
        if self.lora_a.grad is not None or self.lora_b.grad is not None:
            raise RuntimeError("clear adapter gradients before merging LoRA weights")
        delta = (self.lora_b._array @ self.lora_a._array) * self.scaling
        self.base.weight._copy_from(self.base.weight._array + delta)
        self.merged_state._copy_from(
            full((1,), 1, device=self.merged_state.device, dtype=cp.int8)
        )
        return self

    def unmerge(self) -> LoRALinear:
        if not self.merged:
            return self
        if self.lora_a.grad is not None or self.lora_b.grad is not None:
            raise RuntimeError("clear adapter gradients before unmerging LoRA weights")
        delta = (self.lora_b._array @ self.lora_a._array) * self.scaling
        self.base.weight._copy_from(self.base.weight._array - delta)
        self.merged_state._copy_from(
            full((1,), 0, device=self.merged_state.device, dtype=cp.int8)
        )
        return self

    def forward(self, input: Tensor) -> Tensor:
        result = self.base(input)
        if self.merged:
            return result
        adapter = F.linear(
            F.linear(self.adapter_dropout(input), self.lora_a), self.lora_b
        )
        return result + adapter * self.scaling


class SwiGLUFeedForward(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.input_projection = Linear(
            input_dim, hidden_dim * 2, bias, device=device, dtype=dtype
        )
        self.output_projection = Linear(
            hidden_dim,
            input_dim if output_dim is None else output_dim,
            bias,
            device=device,
            dtype=dtype,
        )
        self.dropout = Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        value, gate = self.input_projection(input).chunk(2, -1)
        return self.output_projection(self.dropout(value * A.silu(gate)))
