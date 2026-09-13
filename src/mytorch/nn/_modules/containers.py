"""Module and Parameter containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from mytorch.tensor import Tensor

from .base import Module, Parameter


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
