"""Activation functions for :mod:`mytorch.nn.functional`."""

from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.special import erf, expit

from mytorch import _ops, _random
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


def threshold(input: Tensor, threshold: float, value: float) -> Tensor:
    _ops.require_floating(input, "threshold")
    _require_real("threshold", threshold)
    _require_real("value", value)
    return _ops.apply(
        lambda array: _same_dtype(cp.where(array > threshold, array, value), array),
        input,
        backward=lambda grad, _result, arrays: (grad * (arrays[0] > threshold),),
        name="threshold",
    )


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    _ops.require_floating(input, "hardtanh")
    _require_real("min_val", min_val)
    _require_real("max_val", max_val)
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    return _ops.apply(
        lambda array: _same_dtype(cp.clip(array, min_val, max_val), array),
        input,
        backward=lambda grad, _result, arrays: (
            grad * ((arrays[0] > min_val) & (arrays[0] < max_val)),
        ),
        name="hardtanh",
    )


def relu6(input: Tensor) -> Tensor:
    return hardtanh(input, 0.0, 6.0)


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    _ops.require_floating(input, "elu")
    _require_real("alpha", alpha)

    def forward(array):
        return _same_dtype(cp.where(array > 0, array, alpha * cp.expm1(array)), array)

    def backward(gradient, _result, arrays):
        source = arrays[0]
        return (gradient * cp.where(source > 0, 1, alpha * cp.exp(source)),)

    return _ops.apply(forward, input, backward=backward, name="elu")


def selu(input: Tensor) -> Tensor:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return elu(input, alpha=alpha) * scale


def celu(input: Tensor, alpha: float = 1.0) -> Tensor:
    _ops.require_floating(input, "celu")
    _require_real("alpha", alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    def forward(array):
        return _same_dtype(
            cp.where(array > 0, array, alpha * cp.expm1(array / alpha)), array
        )

    def backward(gradient, _result, arrays):
        source = arrays[0]
        return (gradient * cp.where(source > 0, 1, cp.exp(source / alpha)),)

    return _ops.apply(forward, input, backward=backward, name="celu")


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    _ops.require_floating(input, "prelu")
    _ops.require_floating(weight, "prelu")
    if weight.ndim != 1 or weight.numel() < 1:
        raise ValueError("prelu weight must be a non-empty 1D Tensor")
    if weight.numel() == 1:
        weight_shape = (1,) * input.ndim
        channelwise = False
    else:
        if input.ndim < 2 or input.shape[1] != weight.numel():
            raise ValueError(
                "multi-parameter prelu weight must match input channel dimension 1"
            )
        weight_shape = (1, weight.numel()) + (1,) * (input.ndim - 2)
        channelwise = True

    def forward(array, slopes):
        return _same_dtype(
            cp.where(array > 0, array, array * slopes.reshape(weight_shape)), array
        )

    def backward(gradient, _result, arrays):
        source, slopes = arrays
        shaped = slopes.reshape(weight_shape)
        input_gradient = gradient * cp.where(source > 0, 1, shaped)
        weight_gradient = gradient * cp.where(source > 0, 0, source)
        if channelwise:
            axes = (0,) + tuple(range(2, source.ndim))
            weight_gradient = weight_gradient.sum(axis=axes)
        else:
            weight_gradient = weight_gradient.sum().reshape(slopes.shape)
        return input_gradient, weight_gradient

    return _ops.apply(forward, input, weight, backward=backward, name="prelu")


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
) -> Tensor:
    _ops.require_floating(input, "rrelu")
    _require_real("lower", lower)
    _require_real("upper", upper)
    if lower > upper or lower < 0:
        raise ValueError("rrelu requires 0 <= lower <= upper")
    if not isinstance(training, bool):
        raise TypeError("training must be a bool")
    saved: dict[str, cp.ndarray] = {}

    def forward(array):
        slope = (
            _random.uniform(
                lower,
                upper,
                array.shape,
                device=input._device_index,
                dtype=array.dtype,
            )
            if training
            else cp.full(array.shape, (lower + upper) / 2, dtype=array.dtype)
        )
        saved["slope"] = slope
        return cp.where(array > 0, array, array * slope)

    return _ops.apply(
        forward,
        input,
        backward=lambda grad, _result, arrays: (
            grad * cp.where(arrays[0] > 0, 1, saved["slope"]),
        ),
        name="rrelu",
    )


def hardsigmoid(input: Tensor) -> Tensor:
    _ops.require_floating(input, "hardsigmoid")
    return _ops.apply(
        lambda array: _same_dtype(cp.clip(array / 6 + 0.5, 0, 1), array),
        input,
        backward=lambda grad, _result, arrays: (
            grad * ((arrays[0] > -3) & (arrays[0] < 3)) / 6,
        ),
        name="hardsigmoid",
    )


def hardswish(input: Tensor) -> Tensor:
    _ops.require_floating(input, "hardswish")

    def backward(gradient, _result, arrays):
        source = arrays[0]
        derivative = cp.where(
            source <= -3, 0, cp.where(source >= 3, 1, source / 3 + 0.5)
        )
        return (gradient * derivative,)

    return _ops.apply(
        lambda array: _same_dtype(array * cp.clip(array + 3, 0, 6) / 6, array),
        input,
        backward=backward,
        name="hardswish",
    )


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    _ops.require_floating(input, "hardshrink")
    _require_real("lambd", lambd)
    if lambd < 0:
        raise ValueError("lambd must be non-negative")
    return _ops.apply(
        lambda array: cp.where((array > lambd) | (array < -lambd), array, 0),
        input,
        backward=lambda grad, _result, arrays: (
            grad * ((arrays[0] > lambd) | (arrays[0] < -lambd)),
        ),
        name="hardshrink",
    )


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    _ops.require_floating(input, "softshrink")
    _require_real("lambd", lambd)
    if lambd < 0:
        raise ValueError("lambd must be non-negative")

    def forward(array):
        return cp.where(
            array > lambd, array - lambd, cp.where(array < -lambd, array + lambd, 0)
        )

    return _ops.apply(
        forward,
        input,
        backward=lambda grad, _result, arrays: (
            grad * ((arrays[0] > lambd) | (arrays[0] < -lambd)),
        ),
        name="softshrink",
    )


def tanhshrink(input: Tensor) -> Tensor:
    return input - tanh(input)


def logsigmoid(input: Tensor) -> Tensor:
    return -softplus(-input)


def softsign(input: Tensor) -> Tensor:
    return input / (1 + abs(input))


def softmin(input: Tensor, dim: int = -1) -> Tensor:
    return softmax(-input, dim=dim)


def mish(input: Tensor) -> Tensor:
    return input * tanh(softplus(input))


def _gated(input: Tensor, dim: int, gate: str) -> Tensor:
    axis = _single_dim(dim, input, gate)
    if input.shape[axis] % 2:
        raise ValueError(f"{gate} requires an even size along dim {dim}")
    first, second = input.chunk(2, dim=axis)
    gates = {
        "glu": sigmoid,
        "reglu": relu,
        "geglu": gelu,
        "swiglu": silu,
    }
    return first * gates[gate](second)


def glu(input: Tensor, dim: int = -1) -> Tensor:
    return _gated(input, dim, "glu")


def reglu(input: Tensor, dim: int = -1) -> Tensor:
    return _gated(input, dim, "reglu")


def geglu(input: Tensor, dim: int = -1) -> Tensor:
    return _gated(input, dim, "geglu")


def swiglu(input: Tensor, dim: int = -1) -> Tensor:
    return _gated(input, dim, "swiglu")


def _require_real(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    return float(value)


def _single_dim(dim: int, input: Tensor, operation: str) -> int:
    normalized = _ops.normalize_dims(dim, input.ndim)
    if not isinstance(normalized, int):
        raise TypeError(f"{operation} dim must be a single integer")
    return normalized


def _same_dtype(result: cp.ndarray, input: cp.ndarray) -> cp.ndarray:
    return result.astype(input.dtype, copy=False)
