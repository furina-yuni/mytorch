"""Dropout and stochastic-depth modules."""

from __future__ import annotations

import cupy as cp

from mytorch.tensor import Tensor, tensor

from .._functional import layers as F
from .base import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)


class _FeatureDropout(Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return F.feature_dropout(input, self.p, self.training)


class Dropout1d(_FeatureDropout):
    pass


class Dropout2d(_FeatureDropout):
    pass


class Dropout3d(_FeatureDropout):
    pass


class StochasticDepth(Module):
    def __init__(self, p: float, mode: str = "row") -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        if mode not in {"row", "batch"}:
            raise ValueError("mode must be 'row' or 'batch'")
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input
        if self.p == 1:
            return input * 0
        shape = (
            (1,) * input.ndim
            if self.mode == "batch"
            else (input.shape[0],) + (1,) * (input.ndim - 1)
        )
        with cp.cuda.Device(input._device_index):
            mask = tensor(cp.random.random(shape) >= self.p, device=input.device)
        return input * mask / (1 - self.p)
