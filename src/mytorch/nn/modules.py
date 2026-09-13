"""PyTorch-like neural-network building blocks."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import cupy as cp

from mytorch import _autograd
from mytorch.tensor import Tensor, full, rand

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
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent_buffers", set())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "_parameters",
            "_modules",
            "_buffers",
            "_non_persistent_buffers",
            "training",
        }:
            object.__setattr__(self, name, value)
            return
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")
        if parameters is not None:
            parameters.pop(name, None)
        if modules is not None:
            modules.pop(name, None)
        if buffers is not None and name in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"buffer {name!r} must be a Tensor or None")
            buffers[name] = value
            object.__setattr__(self, name, value)
            return
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

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("buffer name must be a non-empty string without dots")
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("buffer must be a Tensor or None")
        if not isinstance(persistent, bool):
            raise TypeError("persistent must be a bool")
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers.discard(name)
        else:
            self._non_persistent_buffers.add(name)
        object.__setattr__(self, name, tensor)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()

        def visit(module: Module, current_prefix: str):
            check = getattr(module, "_check_materialized", None)
            if check is not None:
                check()
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

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        seen: set[int] = set()

        def visit(module: Module, current_prefix: str):
            for name, buffer in module._buffers.items():
                if buffer is not None and id(buffer) not in seen:
                    seen.add(id(buffer))
                    yield current_prefix + name, buffer
            for name, child in module._modules.items():
                yield from visit(child, current_prefix + name + ".")

        yield from visit(self, prefix)

    def buffers(self) -> Iterator[Tensor]:
        for _, buffer in self.named_buffers():
            yield buffer

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        seen: set[int] = set()

        def visit(module: Module, name: str):
            if id(module) in seen:
                return
            seen.add(id(module))
            yield name, module
            for child_name, child in module._modules.items():
                qualified = f"{name}.{child_name}" if name else child_name
                yield from visit(child, qualified)

        yield from visit(self, prefix)

    def modules(self) -> Iterator[Module]:
        for _, module in self.named_modules():
            yield module

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

    def to(self, device: str | int | None = None, dtype: Any = None) -> Module:
        with _autograd.no_grad():
            for parameter in self.parameters():
                converted = parameter.to(device=device, dtype=dtype)
                if converted is not parameter:
                    parameter._replace_array(converted._array)
            for buffer in self.buffers():
                buffer_dtype = dtype if buffer.dtype.kind == "f" else None
                converted = buffer.to(device=device, dtype=buffer_dtype)
                if converted is not buffer:
                    buffer._replace_array(converted._array)
        return self

    def state_dict(self) -> OrderedDict[str, Tensor]:
        result: OrderedDict[str, Tensor] = OrderedDict()

        def visit(module: Module, prefix: str) -> None:
            for name, parameter in module._parameters.items():
                result[prefix + name] = Tensor(
                    parameter, dtype=parameter.dtype, device=parameter.device
                )
            for name, buffer in module._buffers.items():
                if buffer is not None and name not in module._non_persistent_buffers:
                    result[prefix + name] = Tensor(
                        buffer, dtype=buffer.dtype, device=buffer.device
                    )
            for name, child in module._modules.items():
                visit(child, prefix + name + ".")

        visit(self, "")
        return result

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> dict[str, list[str]]:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool")
        destinations: dict[str, Tensor] = {}

        def visit(module: Module, prefix: str) -> None:
            for name, parameter in module._parameters.items():
                destinations[prefix + name] = parameter
            for name, buffer in module._buffers.items():
                if buffer is not None and name not in module._non_persistent_buffers:
                    destinations[prefix + name] = buffer
            for name, child in module._modules.items():
                visit(child, prefix + name + ".")

        visit(self, "")
        missing = sorted(set(destinations) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(destinations))
        errors: list[str] = []
        for name in sorted(set(destinations) & set(state_dict)):
            source = state_dict[name]
            target = destinations[name]
            if not isinstance(source, Tensor):
                errors.append(f"{name}: expected a Tensor")
            elif source.shape != target.shape:
                errors.append(
                    f"{name}: expected shape {target.shape}, got {source.shape}"
                )
            elif source.dtype != target.dtype:
                errors.append(
                    f"{name}: expected dtype {target.dtype}, got {source.dtype}"
                )
            else:
                target._copy_from(source)
        if errors or (strict and (missing or unexpected)):
            details = (
                errors
                + [f"missing keys: {missing}"] * bool(missing)
                + [f"unexpected keys: {unexpected}"] * bool(unexpected)
            )
            raise RuntimeError("error loading state_dict: " + "; ".join(details))
        return {"missing_keys": missing, "unexpected_keys": unexpected}


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


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> ModuleList:
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        for module in modules:
            self.append(module)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, index: int) -> Module:
        return tuple(self._modules.values())[index]


class ParameterList(Module):
    def __init__(self, parameters: Iterable[Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def append(self, parameter: Parameter) -> ParameterList:
        if not isinstance(parameter, Parameter):
            parameter = Parameter(parameter)
        setattr(self, str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable[Parameter]) -> ParameterList:
        for parameter in parameters:
            self.append(parameter)
        return self

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def __getitem__(self, index: int) -> Parameter:
        return tuple(self._parameters.values())[index]


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


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, self.reduction)


class CrossEntropyLoss(Module):
    def __init__(
        self,
        reduction: str = "mean",
        *,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target,
            self.reduction,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class L1Loss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, self.reduction)


class SmoothL1Loss(MSELoss):
    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__(reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, self.reduction, self.beta)


class HuberLoss(MSELoss):
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction)
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, self.reduction, self.delta)


class BCELoss(MSELoss):
    def __init__(self, weight: Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, self.reduction, weight=self.weight)


class BCEWithLogitsLoss(BCELoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__(weight, reduction)
        self.pos_weight = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.reduction,
            weight=self.weight,
            pos_weight=self.pos_weight,
        )


class NLLLoss(CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(
            input,
            target,
            self.reduction,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


class KLDivLoss(MSELoss):
    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__(reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input, target, self.reduction, log_target=self.log_target)


class PoissonNLLLoss(MSELoss):
    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.poisson_nll_loss(
            input,
            target,
            self.reduction,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
        )


class GaussianNLLLoss(MSELoss):
    def __init__(
        self, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super().__init__(reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, variance: Tensor) -> Tensor:
        return F.gaussian_nll_loss(
            input,
            target,
            variance,
            self.reduction,
            full=self.full,
            eps=self.eps,
        )


class HingeEmbeddingLoss(MSELoss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.hinge_embedding_loss(input, target, self.margin, self.reduction)


class MarginRankingLoss(HingeEmbeddingLoss):
    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__(margin, reduction)

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.margin_ranking_loss(
            input1, input2, target, self.margin, self.reduction
        )


class SoftMarginLoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.soft_margin_loss(input, target, self.reduction)


class MultiLabelSoftMarginLoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multilabel_soft_margin_loss(input, target, self.reduction)


class CosineEmbeddingLoss(HingeEmbeddingLoss):
    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__(margin, reduction)

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(
            input1, input2, target, self.margin, self.reduction
        )


class TripletMarginLoss(MSELoss):
    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            self.margin,
            self.p,
            self.eps,
            self.swap,
            self.reduction,
        )


class MultiMarginLoss(MSELoss):
    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction)
        self.p = p
        self.margin = margin
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multi_margin_loss(
            input,
            target,
            self.p,
            self.margin,
            self.reduction,
            weight=self.weight,
        )


class FocalLoss(MSELoss):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.sigmoid_focal_loss(
            input, target, self.alpha, self.gamma, self.reduction
        )


class DiceLoss(MSELoss):
    def __init__(
        self,
        reduction: str = "mean",
        *,
        from_logits: bool = True,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(reduction)
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.dice_loss(
            input,
            target,
            self.reduction,
            from_logits=self.from_logits,
            smooth=self.smooth,
            eps=self.eps,
        )


class ContrastiveLoss(HingeEmbeddingLoss):
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.contrastive_loss(input1, input2, target, self.margin, self.reduction)
