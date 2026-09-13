"""Reverse-mode automatic differentiation for GPU tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from .tensor import Tensor

Backward = Callable[[cp.ndarray], Sequence[cp.ndarray | None]]

_grad_enabled: ContextVar[bool] = ContextVar("mytorch_grad_enabled", default=True)


@dataclass
class Node:
    """One result-producing operation in the dynamic computation graph."""

    name: str
    parents: tuple[Tensor, ...]
    backward_fn: Backward | None
    versions: tuple[int, ...]
    released: bool = False

    def apply(self, gradient: cp.ndarray) -> Sequence[cp.ndarray | None]:
        if self.released or self.backward_fn is None:
            raise RuntimeError(
                "the computation graph has already been freed; "
                "pass retain_graph=True to backward() if it must be reused"
            )
        for parent, expected in zip(self.parents, self.versions, strict=True):
            if parent._version != expected:
                raise RuntimeError(
                    f"a Tensor needed by {self.name} backward was modified "
                    "after the forward pass"
                )
        return self.backward_fn(gradient)

    def release(self) -> None:
        self.parents = ()
        self.versions = ()
        self.backward_fn = None
        self.released = True


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def set_grad_enabled(enabled: bool):
    token = _grad_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def no_grad():
    return set_grad_enabled(False)


def enable_grad():
    return set_grad_enabled(True)


def sum_to_shape(gradient: cp.ndarray, shape: tuple[int, ...]) -> cp.ndarray:
    """Undo NumPy-style broadcasting for a vector-Jacobian product."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    axes = tuple(
        axis
        for axis, (grad_size, target_size) in enumerate(
            zip(gradient.shape, shape, strict=True)
        )
        if target_size == 1 and grad_size != 1
    )
    if axes:
        gradient = gradient.sum(axis=axes, keepdims=True)
    return gradient.reshape(shape)


def backward(output: Tensor, gradient: Tensor | None, retain_graph: bool) -> None:
    if not output.requires_grad:
        raise RuntimeError(
            "cannot call backward() on a Tensor that does not require grad"
        )
    if output._grad_fn is not None and output._grad_fn.released:
        output._grad_fn.apply(cp.empty((), dtype=output.dtype))

    if gradient is None:
        if output.numel() != 1:
            raise RuntimeError(
                "backward() requires an explicit gradient for non-scalar outputs"
            )
        with cp.cuda.Device(output._device_index):
            initial = cp.ones(output.shape, dtype=output.dtype)
    else:
        from .tensor import Tensor

        if not isinstance(gradient, Tensor):
            raise TypeError("backward gradient must be a Tensor")
        if gradient.shape != output.shape:
            raise ValueError(
                f"backward gradient shape {gradient.shape} does not match "
                f"output shape {output.shape}"
            )
        if gradient._device_index != output._device_index:
            raise ValueError("backward gradient must be on the same CUDA device")
        if gradient.dtype != output.dtype:
            raise TypeError("backward gradient dtype must match the output dtype")
        initial = gradient._array

    topo: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(output, False)]
    while stack:
        tensor, expanded = stack.pop()
        identity = id(tensor)
        if expanded:
            topo.append(tensor)
            continue
        if identity in visited:
            continue
        visited.add(identity)
        stack.append((tensor, True))
        node = tensor._grad_fn
        if node is not None:
            if node.released:
                node.apply(initial)
            for parent in node.parents:
                stack.append((parent, False))

    gradients: dict[int, cp.ndarray] = {id(output): initial}
    for tensor in reversed(topo):
        tensor_gradient = gradients.get(id(tensor))
        if tensor_gradient is None:
            continue
        node = tensor._grad_fn
        if node is None:
            if tensor.requires_grad:
                tensor._accumulate_grad(tensor_gradient)
            continue
        parent_gradients = node.apply(tensor_gradient)
        if len(parent_gradients) != len(node.parents):
            raise RuntimeError(
                f"{node.name} backward returned the wrong gradient count"
            )
        for parent, parent_gradient in zip(node.parents, parent_gradients, strict=True):
            if parent_gradient is None:
                continue
            parent_gradient = sum_to_shape(parent_gradient, parent.shape).astype(
                parent.dtype, copy=False
            )
            identity = id(parent)
            existing = gradients.get(identity)
            gradients[identity] = (
                parent_gradient if existing is None else existing + parent_gradient
            )

    if not retain_graph:
        for tensor in topo:
            if tensor._grad_fn is not None:
                tensor._grad_fn.release()
