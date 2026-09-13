"""Dense trainable modules."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, rand

from ..functional import layers as F
from .base import Module, Parameter


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if (
            not isinstance(in_features, int)
            or isinstance(in_features, bool)
            or in_features <= 0
        ):
            raise ValueError("in_features must be a positive integer")
        if (
            not isinstance(out_features, int)
            or isinstance(out_features, bool)
            or out_features <= 0
        ):
            raise ValueError("out_features must be a positive integer")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            (rand(out_features, in_features, dtype=dtype, device=device) * 2 - 1)
            * bound
        )
        self.bias = (
            Parameter((rand(out_features, dtype=dtype, device=device) * 2 - 1) * bound)
            if bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
