"""Numerically stable, out-of-place activation functions."""

from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.special import erf, expit

from mytorch import _ops
from mytorch.tensor import Tensor


def relu(input: Tensor) -> Tensor:
    _ops.require_floating(input, "relu")
    return _ops.apply(
        lambda array: _same_dtype(cp.maximum(array, 0), array),
        input,
        backward=lambda grad, _result, arrays: (grad * (arrays[0] > 0),),
        name="relu",
    )


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    _ops.require_floating(input, "leaky_relu")
    if not isinstance(negative_slope, (int, float)) or isinstance(negative_slope, bool):
        raise TypeError("negative_slope must be a real number")
    return _ops.apply(
        lambda array: _same_dtype(
            cp.where(array >= 0, array, negative_slope * array), array
        ),
        input,
        backward=lambda grad, _result, arrays: (
            grad * cp.where(arrays[0] > 0, 1, negative_slope),
        ),
        name="leaky_relu",
    )


def sigmoid(input: Tensor) -> Tensor:
    _ops.require_floating(input, "sigmoid")
    return _ops.apply(
        lambda array: _same_dtype(expit(array), array),
        input,
        backward=lambda grad, result, _arrays: (grad * result * (1 - result),),
        name="sigmoid",
    )


def tanh(input: Tensor) -> Tensor:
    _ops.require_floating(input, "tanh")
    return _ops.apply(
        lambda array: _same_dtype(cp.tanh(array), array),
        input,
        backward=lambda grad, result, _arrays: (grad * (1 - result**2),),
        name="tanh",
    )


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    _ops.require_floating(input, "softmax")
    axis = _single_dim(dim, input, "softmax")

    def forward(array: cp.ndarray) -> cp.ndarray:
        shifted = array - cp.max(array, axis=axis, keepdims=True)
        exponentials = cp.exp(shifted)
        result = exponentials / cp.sum(exponentials, axis=axis, keepdims=True)
        return _same_dtype(result, array)

    return _ops.apply(
        forward,
        input,
        backward=lambda grad, result, _arrays: (
            result * (grad - cp.sum(grad * result, axis=axis, keepdims=True)),
        ),
        name="softmax",
    )


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    _ops.require_floating(input, "log_softmax")
    axis = _single_dim(dim, input, "log_softmax")

    def forward(array: cp.ndarray) -> cp.ndarray:
        shifted = array - cp.max(array, axis=axis, keepdims=True)
        normalizer = cp.log(cp.sum(cp.exp(shifted), axis=axis, keepdims=True))
        return _same_dtype(shifted - normalizer, array)

    return _ops.apply(
        forward,
        input,
        backward=lambda grad, result, _arrays: (
            grad - cp.exp(result) * cp.sum(grad, axis=axis, keepdims=True),
        ),
        name="log_softmax",
    )


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    _ops.require_floating(input, "gelu")
    if approximate == "none":

        def exact_backward(gradient, _result, arrays):
            source = arrays[0]
            cdf = 0.5 * (1.0 + erf(source / math.sqrt(2.0)))
            density = cp.exp(-0.5 * source**2) / math.sqrt(2.0 * math.pi)
            return (gradient * (cdf + source * density),)

        return _ops.apply(
            lambda array: _same_dtype(
                0.5 * array * (1.0 + erf(array / math.sqrt(2.0))), array
            ),
            input,
            backward=exact_backward,
            name="gelu",
        )
    if approximate == "tanh":
        coefficient = math.sqrt(2.0 / math.pi)

        def tanh_backward(gradient, _result, arrays):
            source = arrays[0]
            inner = coefficient * (source + 0.044715 * source**3)
            tanh_inner = cp.tanh(inner)
            derivative = 0.5 * (1 + tanh_inner) + 0.5 * source * (
                1 - tanh_inner**2
            ) * coefficient * (1 + 3 * 0.044715 * source**2)
            return (gradient * derivative,)

        return _ops.apply(
            lambda array: _same_dtype(
                0.5
                * array
                * (1.0 + cp.tanh(coefficient * (array + 0.044715 * array**3))),
                array,
            ),
            input,
            backward=tanh_backward,
            name="gelu_tanh",
        )
    raise ValueError("approximate must be either 'none' or 'tanh'")


def silu(input: Tensor) -> Tensor:
    _ops.require_floating(input, "silu")

    def backward(gradient, _result, arrays):
        source = arrays[0]
        probability = expit(source)
        return (gradient * probability * (1 + source * (1 - probability)),)

    return _ops.apply(
        lambda array: _same_dtype(array * expit(array), array),
        input,
        backward=backward,
        name="silu",
    )


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    _ops.require_floating(input, "softplus")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta <= 0:
        raise ValueError("beta must be a positive real number")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a real number")

    def forward(array: cp.ndarray) -> cp.ndarray:
        scaled = beta * array
        stable = cp.logaddexp(0, scaled) / beta
        return _same_dtype(cp.where(scaled > threshold, array, stable), array)

    def backward(gradient, _result, arrays):
        scaled = beta * arrays[0]
        derivative = cp.where(scaled > threshold, 1, expit(scaled))
        return (gradient * derivative,)

    return _ops.apply(forward, input, backward=backward, name="softplus")


def _single_dim(dim: int, input: Tensor, operation: str) -> int:
    normalized = _ops.normalize_dims(dim, input.ndim)
    if not isinstance(normalized, int):
        raise TypeError(f"{operation} dim must be a single integer")
    return normalized


def _same_dtype(result: cp.ndarray, input: cp.ndarray) -> cp.ndarray:
    return result.astype(input.dtype, copy=False)


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("linear expects Tensor input and weight")
    if weight.ndim != 2:
        raise ValueError(f"linear weight must be 2D, got shape {weight.shape}")
    if input.ndim < 1 or input.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"linear input shape {input.shape} is incompatible with weight "
            f"shape {weight.shape}"
        )
    result = input @ weight.T
    if bias is not None:
        if not isinstance(bias, Tensor) or bias.shape != (weight.shape[0],):
            raise ValueError(
                f"linear bias must have shape {(weight.shape[0],)}, "
                f"got {getattr(bias, 'shape', None)}"
            )
        result = result + bias
    return result


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError("mse_loss expects Tensor input and target")
    if input.shape != target.shape:
        raise ValueError(
            f"mse_loss requires matching shapes, got {input.shape} and {target.shape}"
        )
    _validate_reduction(reduction)
    squared = (input - target).square()
    if reduction == "none":
        return squared
    if reduction == "sum":
        return squared.sum()
    return squared.mean()


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError("cross_entropy expects Tensor input and target")
    _ops.require_floating(input, "cross_entropy")
    _validate_reduction(reduction)
    if target.dtype.kind not in {"i", "u"}:
        raise TypeError("cross_entropy target must have an integer dtype")
    if input.ndim == 1:
        if target.ndim != 0:
            raise ValueError("unbatched cross_entropy target must be scalar")
        batched = False
        classes = input.shape[0]
    elif input.ndim == 2:
        if target.shape != (input.shape[0],):
            raise ValueError(
                f"cross_entropy target must have shape {(input.shape[0],)}, "
                f"got {target.shape}"
            )
        batched = True
        classes = input.shape[1]
    else:
        raise ValueError("cross_entropy input must have shape (C,) or (N, C)")
    if bool(cp.any((target._array < 0) | (target._array >= classes))):
        raise ValueError("cross_entropy target contains an invalid class index")

    def forward(logits, labels):
        matrix = logits if batched else logits[None, :]
        indices = labels if batched else labels.reshape(1)
        shifted = matrix - cp.max(matrix, axis=1, keepdims=True)
        log_probs = shifted - cp.log(cp.sum(cp.exp(shifted), axis=1, keepdims=True))
        losses = -log_probs[cp.arange(matrix.shape[0]), indices]
        if reduction == "none":
            return losses if batched else losses.reshape(())
        if reduction == "sum":
            return losses.sum()
        return losses.mean()

    def backward(gradient, _result, arrays):
        logits, labels = arrays
        matrix = logits if batched else logits[None, :]
        indices = labels if batched else labels.reshape(1)
        shifted = matrix - cp.max(matrix, axis=1, keepdims=True)
        probabilities = cp.exp(shifted)
        probabilities /= cp.sum(probabilities, axis=1, keepdims=True)
        grad_logits = probabilities
        grad_logits[cp.arange(matrix.shape[0]), indices] -= 1
        if reduction == "mean":
            grad_logits /= matrix.shape[0]
        if reduction == "none" and batched:
            grad_logits *= gradient[:, None]
        else:
            grad_logits *= gradient
        if not batched:
            grad_logits = grad_logits.reshape(logits.shape)
        return grad_logits, None

    return _ops.apply(
        forward,
        input,
        target,
        backward=backward,
        name="cross_entropy",
    )
