"""Module and Parameter foundations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any

from mytorch import _autograd
from mytorch.tensor import Tensor


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
