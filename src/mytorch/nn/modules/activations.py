"""Activation Module wrappers."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, full

from ..functional import activations as F
from .base import Module, Parameter


class _Activation(Module):
    function: Any

    def forward(self, input: Tensor) -> Tensor:
        return self.function(input)


class ReLU(_Activation):
    function = staticmethod(F.relu)


class Sigmoid(_Activation):
    function = staticmethod(F.sigmoid)


class Tanh(_Activation):
    function = staticmethod(F.tanh)


class SiLU(_Activation):
    function = staticmethod(F.silu)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim)


class LogSoftmax(Softmax):
    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim)


class GELU(Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, self.approximate)


class Softplus(Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)


class Threshold(Module):
    def __init__(self, threshold: float, value: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.threshold(input, self.threshold, self.value)


class Hardtanh(Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val)


class ReLU6(_Activation):
    function = staticmethod(F.relu6)


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha)


class SELU(_Activation):
    function = staticmethod(F.selu)


class CELU(ELU):
    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, self.alpha)


class PReLU(Module):
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if (
            not isinstance(num_parameters, int)
            or isinstance(num_parameters, bool)
            or num_parameters <= 0
        ):
            raise ValueError("num_parameters must be a positive integer")
        self.num_parameters = num_parameters
        self.weight = Parameter(
            full(num_parameters, fill_value=init, dtype=dtype, device=device)
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)


class RReLU(Module):
    def __init__(self, lower: float = 1.0 / 8, upper: float = 1.0 / 3) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training)


class Hardsigmoid(_Activation):
    function = staticmethod(F.hardsigmoid)


class Hardswish(_Activation):
    function = staticmethod(F.hardswish)


class Hardshrink(Module):
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)


class Softshrink(Hardshrink):
    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)


class Tanhshrink(_Activation):
    function = staticmethod(F.tanhshrink)


class LogSigmoid(_Activation):
    function = staticmethod(F.logsigmoid)


class Softsign(_Activation):
    function = staticmethod(F.softsign)


class Mish(_Activation):
    function = staticmethod(F.mish)


class Softmin(Softmax):
    def forward(self, input: Tensor) -> Tensor:
        return F.softmin(input, self.dim)


class GLU(Module):
    function = staticmethod(F.glu)

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return self.function(input, self.dim)


class ReGLU(GLU):
    function = staticmethod(F.reglu)


class GEGLU(GLU):
    function = staticmethod(F.geglu)


class SwiGLU(GLU):
    function = staticmethod(F.swiglu)
