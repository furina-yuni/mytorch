"""Reverse-mode automatic differentiation for GPU tensors."""

from __future__ import annotations

import traceback
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
_inference_enabled: ContextVar[bool] = ContextVar(
    "mytorch_inference_enabled", default=False
)
_anomaly_config: ContextVar[tuple[bool, bool]] = ContextVar(
    "mytorch_anomaly_config", default=(False, True)
)


@dataclass
class Node:
    """One result-producing operation in the dynamic computation graph."""

    name: str
    parents: tuple[Tensor, ...]
    backward_fn: Backward | None
    versions: tuple[int, ...]
    forward_trace: str | None = None
    released: bool = False

    def apply(self, gradient: cp.ndarray) -> Sequence[cp.ndarray | None]:
        if self.released or self.backward_fn is None:
            raise RuntimeError(
                "the computation graph has already been freed; "
                "pass retain_graph=True if it must be reused"
            )
        for parent, expected in zip(self.parents, self.versions, strict=True):
            if parent._version != expected:
                raise RuntimeError(
                    f"a Tensor needed by {self.name} backward was modified "
                    "after the forward pass"
                )
        anomaly_enabled, check_nan = _anomaly_config.get()
        try:
            from .profiler.profiler import _record_operation

            gradients = _record_operation(
                f"{self.name} backward",
                int(gradient.device.id),
                [gradient],
                lambda: self.backward_fn(gradient),
            )
        except Exception as exc:
            if not anomaly_enabled:
                raise
            message = f"error detected in {self.name} backward"
            if self.forward_trace:
                message += f"\nForward operation trace:\n{self.forward_trace}"
            raise RuntimeError(message) from exc
        if anomaly_enabled and check_nan:
            for result in gradients:
                if result is not None and not bool(cp.isfinite(result).all().item()):
                    message = f"non-finite gradient detected in {self.name} backward"
                    if self.forward_trace:
                        message += f"\nForward operation trace:\n{self.forward_trace}"
                    raise RuntimeError(message)
        return gradients

    def release(self) -> None:
        self.parents = ()
        self.versions = ()
        self.backward_fn = None
        self.forward_trace = None
        self.released = True


def is_grad_enabled() -> bool:
    """Return whether forward operations are currently recorded for backward."""
    return _grad_enabled.get()


def is_inference_mode_enabled() -> bool:
    """Return whether the current context creates inference-only tensors."""
    return _inference_enabled.get()


def is_anomaly_enabled() -> bool:
    """Return whether backward anomaly checks are currently active."""
    return _anomaly_config.get()[0]


@contextmanager
def set_grad_enabled(enabled: bool):
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    token = _grad_enabled.set(enabled)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def no_grad():
    """Disable graph recording inside the context while retaining Tensor metadata."""
    return set_grad_enabled(False)


def enable_grad():
    """Enable graph recording inside the context, including inside ``no_grad``."""
    return set_grad_enabled(True)


@contextmanager
def inference_mode(mode: bool = True):
    """Disable autograd bookkeeping for evaluation-only Tensor computations."""
    if not isinstance(mode, bool):
        raise TypeError("mode must be a bool")
    inference_token = _inference_enabled.set(mode)
    grad_token = _grad_enabled.set(not mode)
    try:
        yield
    finally:
        _grad_enabled.reset(grad_token)
        _inference_enabled.reset(inference_token)


@contextmanager
def detect_anomaly(check_nan: bool = True):
    """Add forward traces and non-finite checks to backward error reporting."""
    if not isinstance(check_nan, bool):
        raise TypeError("check_nan must be a bool")
    token = _anomaly_config.set((True, check_nan))
    try:
        yield
    finally:
        _anomaly_config.reset(token)


@contextmanager
def set_detect_anomaly(mode: bool, check_nan: bool = True):
    """Temporarily enable or disable backward anomaly detection."""
    if not isinstance(mode, bool) or not isinstance(check_nan, bool):
        raise TypeError("mode and check_nan must be bool values")
    token = _anomaly_config.set((mode, check_nan))
    try:
        yield
    finally:
        _anomaly_config.reset(token)


def capture_forward_trace() -> str | None:
    """Capture a compact Python stack only while anomaly detection is enabled."""
    if not is_anomaly_enabled():
        return None
    return "".join(traceback.format_stack(limit=12)[:-2])


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


def _as_tuple(value, *, name: str) -> tuple:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return tuple(value)
    return (value,)


def _initial_gradient(output: Tensor, gradient: Tensor | None) -> cp.ndarray:
    from .tensor import Tensor

    if not isinstance(output, Tensor):
        raise TypeError("outputs must contain only Tensor values")
    if not output.requires_grad:
        raise RuntimeError("an output does not require grad")
    if output._grad_fn is not None and output._grad_fn.released:
        output._grad_fn.apply(cp.empty((), dtype=output.dtype))
    if gradient is None:
        if output.numel() != 1:
            raise RuntimeError(
                "an explicit gradient is required for non-scalar outputs"
            )
        with cp.cuda.Device(output._device_index):
            return cp.ones(output.shape, dtype=output.dtype)
    if not isinstance(gradient, Tensor):
        raise TypeError("gradient outputs must contain Tensor values or None")
    if gradient.shape != output.shape:
        raise ValueError(
            f"gradient shape {gradient.shape} does not match output shape "
            f"{output.shape}"
        )
    if gradient._device_index != output._device_index:
        raise ValueError("gradient and output must be on the same CUDA device")
    if gradient.dtype != output.dtype:
        raise TypeError("gradient dtype must match the output dtype")
    return gradient._array


def _topological_order(outputs: tuple[Tensor, ...]) -> list[Tensor]:
    topo: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(output, False) for output in outputs]
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
                node.apply(cp.empty(tensor.shape, dtype=tensor.dtype))
            for parent in node.parents:
                stack.append((parent, False))
    return topo


def _compute_gradients(
    outputs: tuple[Tensor, ...],
    initial_gradients: tuple[cp.ndarray, ...],
) -> tuple[list[Tensor], dict[int, cp.ndarray]]:
    topo = _topological_order(outputs)
    gradients: dict[int, cp.ndarray] = {}
    for output, initial in zip(outputs, initial_gradients, strict=True):
        identity = id(output)
        existing = gradients.get(identity)
        gradients[identity] = initial if existing is None else existing + initial

    for tensor in reversed(topo):
        tensor_gradient = gradients.get(id(tensor))
        if tensor_gradient is None or tensor._grad_fn is None:
            continue
        node = tensor._grad_fn
        parents = node.parents
        parent_gradients = node.apply(tensor_gradient)
        if len(parent_gradients) != len(parents):
            raise RuntimeError(
                f"{node.name} backward returned the wrong gradient count"
            )
        for parent, parent_gradient in zip(parents, parent_gradients, strict=True):
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
    return topo, gradients


def _release_graph(topo: Sequence[Tensor]) -> None:
    for tensor in topo:
        if tensor._grad_fn is not None:
            tensor._grad_fn.release()


def backward(output: Tensor, gradient: Tensor | None, retain_graph: bool) -> None:
    initial = _initial_gradient(output, gradient)
    topo, gradients = _compute_gradients((output,), (initial,))
    for tensor in topo:
        if tensor._grad_fn is None and tensor.requires_grad:
            tensor_gradient = gradients.get(id(tensor))
            if tensor_gradient is not None:
                tensor._accumulate_grad(tensor_gradient)
    if not retain_graph:
        _release_graph(topo)


def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    *,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[Tensor | None, ...]:
    """Return vector-Jacobian products without accumulating into ``Tensor.grad``."""
    from .tensor import Tensor

    if not isinstance(retain_graph, bool) or not isinstance(allow_unused, bool):
        raise TypeError("retain_graph and allow_unused must be bool values")
    output_values = _as_tuple(outputs, name="outputs")
    input_values = _as_tuple(inputs, name="inputs")
    if not all(isinstance(value, Tensor) for value in input_values):
        raise TypeError("inputs must contain only Tensor values")
    if grad_outputs is None:
        gradient_values = (None,) * len(output_values)
    else:
        gradient_values = _as_tuple(grad_outputs, name="grad_outputs")
        if len(gradient_values) != len(output_values):
            raise ValueError("grad_outputs must have the same length as outputs")
    initial = tuple(
        _initial_gradient(output, gradient)
        for output, gradient in zip(output_values, gradient_values, strict=True)
    )
    topo, gradients = _compute_gradients(output_values, initial)
    results: list[Tensor | None] = []
    for value in input_values:
        array = gradients.get(id(value))
        if array is None:
            if not allow_unused:
                if not retain_graph:
                    _release_graph(topo)
                raise RuntimeError(
                    "one of the differentiated Tensors was not used in the graph"
                )
            results.append(None)
        else:
            with cp.cuda.Device(value._device_index):
                results.append(Tensor._from_array(cp.array(array, copy=True)))
    if not retain_graph:
        _release_graph(topo)
    return tuple(results)
