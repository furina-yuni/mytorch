"""Bilinear and lazy dense modules."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, rand

from .._functional import layers as F
from .base import Module, Parameter
from .linear import Linear
from .utils import positive


class Bilinear(Module):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.in1_features = positive("in1_features", in1_features)
        self.in2_features = positive("in2_features", in2_features)
        self.out_features = positive("out_features", out_features)
        bound = 1 / math.sqrt(in1_features)
        self.weight = Parameter(
            (
                rand(
                    out_features, in1_features, in2_features, device=device, dtype=dtype
                )
                * 2
                - 1
            )
            * bound
        )
        self.bias = (
            Parameter((rand(out_features, device=device, dtype=dtype) * 2 - 1) * bound)
            if bias
            else None
        )

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)


class LazyLinear(Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.out_features = positive("out_features", out_features)
        self.use_bias = bool(bias)
        self.device = device
        self.dtype = dtype
        self.in_features: int | None = None
        self.weight = None
        self.bias = None

    def _check_materialized(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                "LazyLinear must receive an input before parameters are requested"
            )

    def _materialize(self, in_features: int) -> None:
        layer = Linear(
            in_features,
            self.out_features,
            self.use_bias,
            device=self.device,
            dtype=self.dtype,
        )
        self.in_features = in_features
        self.weight = layer.weight
        self.bias = layer.bias

    def to(self, device: str | int | None = None, dtype: Any = None) -> LazyLinear:
        if self.weight is None:
            if device is not None:
                self.device = device
            if dtype is not None:
                self.dtype = dtype
            return self
        return super().to(device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim < 1:
            raise ValueError("LazyLinear expects an input with at least one dimension")
        if self.weight is None:
            self._materialize(input.shape[-1])
        elif input.shape[-1] != self.in_features:
            raise ValueError(
                "LazyLinear input feature size changed after materialization"
            )
        return F.linear(input, self.weight, self.bias)
