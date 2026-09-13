"""PyTorch-like neural-network building blocks."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, rand

from . import functional as F


class Parameter(Tensor):
    """A trainable leaf Tensor registered by Module."""

    def __init__(
        self,
        data: Any,
        *,
        dtype: Any = None,
        device: str | int | None = None,
        requires_grad: bool = True,
    ) -> None:
        resolved_device = (
            data.device if device is None and isinstance(data, Tensor) else device
        )
        super().__init__(
            data,
            dtype=dtype,
            device="cuda:0" if resolved_device is None else resolved_device,
            requires_grad=requires_grad,
        )


class Module:
    """Base class for composable neural-network layers."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_parameters", "_modules", "training"}:
            object.__setattr__(self, name, value)
            return
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is not None:
            parameters.pop(name, None)
        if modules is not None:
            modules.pop(name, None)
        if isinstance(value, Parameter):
            parameters[name] = value
        elif isinstance(value, Module):
            modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def add_module(self, name: str, module: Module) -> None:
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("module name must be a non-empty string without dots")
        if not isinstance(module, Module):
            raise TypeError("module must be a Module")
        setattr(self, name, module)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()

        def visit(module: Module, current_prefix: str):
            for name, parameter in module._parameters.items():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield current_prefix + name, parameter
            for name, child in module._modules.items():
                yield from visit(child, current_prefix + name + ".")

        yield from visit(self, prefix)

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def children(self) -> Iterator[Module]:
        yield from self._modules.values()

    def train(self, mode: bool = True) -> Module:
        if not isinstance(mode, bool):
            raise TypeError("mode must be a bool")
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if not isinstance(set_to_none, bool):
            raise TypeError("set_to_none must be a bool")
        for parameter in self.parameters():
            parameter._clear_grad(set_to_none=set_to_none)


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


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def forward(self, input: Tensor) -> Tensor:
        result = input
        for module in self._modules.values():
            result = module(result)
        return result

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Sequential index must be an integer")
        return tuple(self._modules.values())[index]


class Identity(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


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


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, self.reduction)


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, self.reduction)
