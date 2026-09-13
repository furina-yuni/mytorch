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
            cp.random.uniform(lower, upper, size=array.shape).astype(array.dtype)
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


def bilinear(
    input1: Tensor,
    input2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    if input1.shape[:-1] != input2.shape[:-1]:
        raise ValueError("bilinear inputs must have matching batch dimensions")
    if weight.ndim != 3:
        raise ValueError("bilinear weight must have shape (out, in1, in2)")
    if input1.shape[-1] != weight.shape[1] or input2.shape[-1] != weight.shape[2]:
        raise ValueError("bilinear input features do not match weight")

    def backward(gradient, _result, arrays):
        left, right, matrix = arrays
        batch = math.prod(left.shape[:-1])
        return (
            cp.einsum("...o,oij,...j->...i", gradient, matrix, right),
            cp.einsum("...o,oij,...i->...j", gradient, matrix, left),
            cp.einsum(
                "bo,bi,bj->oij",
                gradient.reshape(batch, gradient.shape[-1]),
                left.reshape(batch, left.shape[-1]),
                right.reshape(batch, right.shape[-1]),
            ),
        )

    result = _ops.apply(
        lambda left, right, matrix: cp.einsum(
            "...i,oij,...j->...o", left, matrix, right
        ),
        input1,
        input2,
        weight,
        backward=backward,
        name="bilinear",
    )
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError("bilinear bias has the wrong shape")
        result = result + bias
    return result


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    from mytorch.tensor import tensor

    if not 0 <= p <= 1:
        raise ValueError("dropout probability must be between 0 and 1")
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0
    with cp.cuda.Device(input._device_index):
        mask = tensor(cp.random.random(input.shape) >= p, device=input.device)
    return input * mask / (1 - p)


def feature_dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    from mytorch.tensor import tensor

    if not 0 <= p <= 1:
        raise ValueError("dropout probability must be between 0 and 1")
    if input.ndim < 2:
        raise ValueError("feature dropout input must have at least two dimensions")
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0
    shape = input.shape[:2] + (1,) * (input.ndim - 2)
    with cp.cuda.Device(input._device_index):
        mask = tensor(cp.random.random(shape) >= p, device=input.device)
    return input * mask / (1 - p)


dropout1d = feature_dropout
dropout2d = feature_dropout
dropout3d = feature_dropout


def stochastic_depth(
    input: Tensor, p: float, mode: str = "row", training: bool = True
) -> Tensor:
    from mytorch.tensor import tensor

    if not 0 <= p <= 1 or mode not in {"row", "batch"}:
        raise ValueError("invalid stochastic_depth probability or mode")
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0
    shape = (
        (1,) * input.ndim
        if mode == "batch"
        else (input.shape[0],) + (1,) * (input.ndim - 1)
    )
    with cp.cuda.Device(input._device_index):
        mask = tensor(cp.random.random(shape) >= p, device=input.device)
    return input * mask / (1 - p)


def embedding(input: Tensor, weight: Tensor, padding_idx: int | None = None) -> Tensor:
    if input.dtype.kind not in "iu" or weight.ndim != 2:
        raise ValueError("embedding expects integer indices and a 2D weight")
    if padding_idx is not None:
        padding_idx = padding_idx + weight.shape[0] if padding_idx < 0 else padding_idx
        if padding_idx < 0 or padding_idx >= weight.shape[0]:
            raise ValueError("padding_idx is out of range")

    def backward(gradient, _result, arrays):
        matrix, indices = arrays
        grad_weight = cp.zeros_like(matrix)
        if padding_idx is None:
            cp.add.at(grad_weight, indices, gradient)
        else:
            selected = indices != padding_idx
            cp.add.at(grad_weight, indices[selected], gradient[selected])
        return grad_weight, None

    return _ops.apply(
        lambda matrix, indices: matrix[indices],
        weight,
        input,
        backward=backward,
        name="embedding",
    )


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Tensor,
    mode: str = "mean",
) -> Tensor:
    from mytorch.tensor import arange, full, maximum, stack, where, zeros

    if mode not in {"sum", "mean", "max"}:
        raise ValueError("embedding_bag mode must be 'sum', 'mean', or 'max'")
    if input.ndim != 1 or offsets.ndim != 1:
        raise ValueError("embedding_bag expects 1D input and offsets")
    values = embedding(input, weight)
    positions = arange(input.shape[0], dtype=cp.int64, device=input.device)
    outputs = []
    for bag in range(offsets.shape[0]):
        end = (
            offsets[bag + 1]
            if bag + 1 < offsets.shape[0]
            else full((), input.shape[0], dtype=cp.int64, device=input.device)
        )
        mask = (positions >= offsets[bag]) * (positions < end)
        count = mask.sum()
        if mode == "sum":
            outputs.append((values * mask.unsqueeze(1)).sum(0))
        elif mode == "mean":
            outputs.append((values * mask.unsqueeze(1)).sum(0) / maximum(count, 1))
        else:
            outputs.append(
                where(
                    count > 0,
                    where(mask.unsqueeze(1), values, -cp.inf).max(0),
                    zeros(weight.shape[1], device=input.device, dtype=weight.dtype),
                )
            )
    return stack(outputs)


def layer_norm(
    input: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if input.shape[-len(normalized_shape) :] != normalized_shape:
        raise ValueError("input trailing dimensions do not match normalized_shape")
    dims = tuple(range(input.ndim - len(normalized_shape), input.ndim))
    mean_value = input.mean(dims, keepdim=True)
    variance = ((input - mean_value) ** 2).mean(dims, keepdim=True)
    result = (input - mean_value) * (variance + eps).rsqrt()
    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias
    return result


def rms_norm(
    input: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if input.shape[-len(normalized_shape) :] != normalized_shape:
        raise ValueError("input trailing dimensions do not match normalized_shape")
    dims = tuple(range(input.ndim - len(normalized_shape), input.ndim))
    result = input * ((input * input).mean(dims, keepdim=True) + eps).rsqrt()
    return result if weight is None else result * weight


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if input.ndim < 2 or input.shape[1] % num_groups:
        raise ValueError("channels must be divisible by num_groups")
    grouped = input.reshape(
        input.shape[0], num_groups, input.shape[1] // num_groups, *input.shape[2:]
    )
    dims = tuple(range(2, grouped.ndim))
    mean_value = grouped.mean(dims, keepdim=True)
    variance = ((grouped - mean_value) ** 2).mean(dims, keepdim=True)
    result = ((grouped - mean_value) * (variance + eps).rsqrt()).reshape(input.shape)
    affine_shape = (1, input.shape[1]) + (1,) * (input.ndim - 2)
    if weight is not None:
        result = result * weight.reshape(affine_shape)
    if bias is not None:
        result = result + bias.reshape(affine_shape)
    return result


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if input.ndim < 2:
        raise ValueError("batch_norm expects at least 2D input")
    axes = (0,) + tuple(range(2, input.ndim))
    shape = (1, input.shape[1]) + (1,) * (input.ndim - 2)
    if training:
        mean_value = input.mean(axes, keepdim=True)
        variance = ((input - mean_value) ** 2).mean(axes, keepdim=True)
        if running_mean is not None:
            running_mean._copy_from(
                (1 - momentum) * running_mean._array
                + momentum * mean_value._array.reshape(-1)
            )
        if running_var is not None:
            count = math.prod(input.shape[axis] for axis in axes)
            corrected = variance._array.reshape(-1) * count / max(count - 1, 1)
            running_var._copy_from(
                (1 - momentum) * running_var._array + momentum * corrected
            )
    else:
        if running_mean is None or running_var is None:
            raise ValueError("evaluation batch_norm requires running statistics")
        mean_value, variance = running_mean.reshape(shape), running_var.reshape(shape)
    result = (input - mean_value) * (variance + eps).rsqrt()
    if weight is not None:
        result = result * weight.reshape(shape)
    if bias is not None:
        result = result + bias.reshape(shape)
    return result


def instance_norm(
    input: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if input.ndim < 3:
        raise ValueError("instance_norm expects at least 3D input")
    dims = tuple(range(2, input.ndim))
    shape = (1, input.shape[1]) + (1,) * (input.ndim - 2)
    if use_input_stats:
        mean_value = input.mean(dims, keepdim=True)
        variance = ((input - mean_value) ** 2).mean(dims, keepdim=True)
        if running_mean is not None:
            batch_mean = mean_value.mean(0).reshape(-1)
            running_mean._copy_from(
                (1 - momentum) * running_mean._array + momentum * batch_mean._array
            )
        if running_var is not None:
            batch_var = variance.mean(0).reshape(-1)
            running_var._copy_from(
                (1 - momentum) * running_var._array + momentum * batch_var._array
            )
    else:
        if running_mean is None or running_var is None:
            raise ValueError("instance_norm requires running statistics")
        mean_value, variance = running_mean.reshape(shape), running_var.reshape(shape)
    result = (input - mean_value) * (variance + eps).rsqrt()
    if weight is not None:
        result = result * weight.reshape(shape)
    if bias is not None:
        result = result + bias.reshape(shape)
    return result


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .conv import convolution

    return convolution(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .conv import convolution

    return convolution(input, weight, bias, stride, padding, dilation, groups)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .conv import convolution

    return convolution(input, weight, bias, stride, padding, dilation, groups)


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    from .conv import convolution_transpose

    return convolution_transpose(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )


conv_transpose2d = conv_transpose1d
conv_transpose3d = conv_transpose1d


def max_pool1d(input, kernel_size, stride=None, padding=0):
    from .conv import pool_nd

    return pool_nd(input, kernel_size, stride, padding, "max")


max_pool2d = max_pool1d
max_pool3d = max_pool1d


def avg_pool1d(input, kernel_size, stride=None, padding=0):
    from .conv import pool_nd

    return pool_nd(input, kernel_size, stride, padding, "avg")


avg_pool2d = avg_pool1d
avg_pool3d = avg_pool1d


def adaptive_max_pool1d(input, output_size):
    from .conv import adaptive_pool_nd

    return adaptive_pool_nd(input, output_size, "max")


adaptive_max_pool2d = adaptive_max_pool1d
adaptive_max_pool3d = adaptive_max_pool1d


def adaptive_avg_pool1d(input, output_size):
    from .conv import adaptive_pool_nd

    return adaptive_pool_nd(input, output_size, "avg")


adaptive_avg_pool2d = adaptive_avg_pool1d
adaptive_avg_pool3d = adaptive_avg_pool1d


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    from .attention import scaled_dot_product_attention as implementation

    return implementation(query, key, value, attn_mask, dropout_p, is_causal, scale)


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


def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    _validate_classification(input, target, "cross_entropy")
    _validate_reduction(reduction)
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")
    _require_real("label_smoothing", label_smoothing)
    if not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0, 1]")
    classes, output_shape, unbatched = _classification_shape(input, target)
    weights = _validate_class_weight(weight, classes, input, "cross_entropy")
    labels = target._array.reshape(-1)
    valid = labels != ignore_index
    if bool(cp.any(valid & ((labels < 0) | (labels >= classes)))):
        raise ValueError("cross_entropy target contains an invalid class index")

    def matrix_view(array):
        if unbatched:
            return array.reshape(1, classes)
        return cp.moveaxis(array, 1, -1).reshape(-1, classes)

    def forward(logits, label_array, weight_array):
        matrix = matrix_view(logits)
        flat_labels = label_array.reshape(-1)
        valid_mask = flat_labels != ignore_index
        safe_labels = cp.where(valid_mask, flat_labels, 0)
        shifted = matrix - cp.max(matrix, axis=1, keepdims=True)
        log_probs = shifted - cp.log(cp.sum(cp.exp(shifted), axis=1, keepdims=True))
        class_weights = (
            cp.ones(classes, dtype=matrix.dtype)
            if weight_array is None
            else weight_array
        )
        target_weight = class_weights[safe_labels]
        hard = -log_probs[cp.arange(matrix.shape[0]), safe_labels] * target_weight
        smooth = -(log_probs * class_weights[None, :]).sum(axis=1) / classes
        losses = ((1 - label_smoothing) * hard + label_smoothing * smooth) * valid_mask
        if reduction == "none":
            return losses.reshape(output_shape)
        total = losses.sum()
        if reduction == "sum":
            return total
        denominator = cp.sum(target_weight * valid_mask)
        return cp.where(denominator > 0, total / denominator, 0)

    def backward(gradient, _result, arrays):
        logits, label_array, weight_array = arrays
        matrix = matrix_view(logits)
        flat_labels = label_array.reshape(-1)
        valid_mask = flat_labels != ignore_index
        safe_labels = cp.where(valid_mask, flat_labels, 0)
        shifted = matrix - cp.max(matrix, axis=1, keepdims=True)
        probabilities = cp.exp(shifted)
        probabilities /= cp.sum(probabilities, axis=1, keepdims=True)
        class_weights = (
            cp.ones(classes, dtype=matrix.dtype)
            if weight_array is None
            else weight_array
        )
        target_weight = class_weights[safe_labels]
        coefficients = cp.broadcast_to(
            label_smoothing * class_weights[None, :] / classes,
            matrix.shape,
        ).copy()
        coefficients[cp.arange(matrix.shape[0]), safe_labels] += (
            1 - label_smoothing
        ) * target_weight
        coefficients *= valid_mask[:, None]
        grad_logits = (
            probabilities * coefficients.sum(axis=1, keepdims=True) - coefficients
        )
        if reduction == "mean":
            denominator = cp.sum(target_weight * valid_mask)
            grad_logits = cp.where(denominator > 0, grad_logits / denominator, 0)
        if reduction == "none":
            grad_logits *= gradient.reshape(-1, 1)
        else:
            grad_logits *= gradient
        if unbatched:
            grad_logits = grad_logits.reshape(logits.shape)
        else:
            moved_shape = (input.shape[0],) + input.shape[2:] + (classes,)
            grad_logits = cp.moveaxis(grad_logits.reshape(moved_shape), -1, 1)
        return grad_logits, None, None

    return _ops.apply(
        forward,
        input,
        target,
        weights,
        backward=backward,
        name="cross_entropy",
    )


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    _validate_pair(input, target, "l1_loss")
    return _reduce_loss(abs(input - target), reduction)


def smooth_l1_loss(
    input: Tensor, target: Tensor, reduction: str = "mean", beta: float = 1.0
) -> Tensor:
    _validate_pair(input, target, "smooth_l1_loss")
    _require_real("beta", beta)
    if beta < 0:
        raise ValueError("beta must be non-negative")
    difference = abs(input - target)
    if beta == 0:
        return _reduce_loss(difference, reduction)
    from mytorch.tensor import where

    loss = where(
        difference < beta, 0.5 * difference.square() / beta, difference - 0.5 * beta
    )
    return _reduce_loss(loss, reduction)


def huber_loss(
    input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0
) -> Tensor:
    _validate_pair(input, target, "huber_loss")
    _require_real("delta", delta)
    if delta <= 0:
        raise ValueError("delta must be positive")
    difference = abs(input - target)
    from mytorch.tensor import where

    loss = where(
        difference < delta,
        0.5 * difference.square(),
        delta * (difference - 0.5 * delta),
    )
    return _reduce_loss(loss, reduction)


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    weight: Tensor | None = None,
) -> Tensor:
    _validate_pair(input, target, "binary_cross_entropy")
    weights = _validate_optional_weight(weight, input, "binary_cross_entropy")
    if bool(cp.any((input._array < 0) | (input._array > 1))):
        raise ValueError("binary_cross_entropy input must be in [0, 1]")
    epsilon = 1e-12

    def forward(probabilities, targets, weight_array):
        clipped = cp.clip(probabilities, epsilon, 1 - epsilon)
        losses = -(targets * cp.log(clipped) + (1 - targets) * cp.log1p(-clipped))
        return losses if weight_array is None else losses * weight_array

    def backward(gradient, _result, arrays):
        probabilities, targets, weight_array = arrays
        clipped = cp.clip(probabilities, epsilon, 1 - epsilon)
        grad_input = (clipped - targets) / (clipped * (1 - clipped))
        if weight_array is not None:
            grad_input *= weight_array
        return gradient * grad_input, None, None

    loss = _ops.apply(
        forward,
        input,
        target,
        weights,
        backward=backward,
        name="binary_cross_entropy",
    )
    return _reduce_loss(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
) -> Tensor:
    _validate_pair(input, target, "binary_cross_entropy_with_logits")
    weights = _validate_optional_weight(
        weight, input, "binary_cross_entropy_with_logits"
    )
    positive_weights = _validate_optional_weight(
        pos_weight, input, "binary_cross_entropy_with_logits"
    )

    def forward(logits, targets, weight_array, positive_array):
        factor = 1 if positive_array is None else 1 + (positive_array - 1) * targets
        losses = (1 - targets) * logits + factor * cp.logaddexp(0, -logits)
        return losses if weight_array is None else losses * weight_array

    def backward(gradient, _result, arrays):
        logits, targets, weight_array, positive_array = arrays
        factor = 1 if positive_array is None else 1 + (positive_array - 1) * targets
        grad_input = (1 - targets) - factor * expit(-logits)
        if weight_array is not None:
            grad_input *= weight_array
        return gradient * grad_input, None, None, None

    loss = _ops.apply(
        forward,
        input,
        target,
        weights,
        positive_weights,
        backward=backward,
        name="binary_cross_entropy_with_logits",
    )
    return _reduce_loss(loss, reduction)


def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    _validate_classification(input, target, "nll_loss")
    classes, output_shape, unbatched = _classification_shape(input, target)
    weights = _validate_class_weight(weight, classes, input, "nll_loss")
    _validate_reduction(reduction)

    def matrix_view(array):
        return (
            array.reshape(1, classes)
            if unbatched
            else cp.moveaxis(array, 1, -1).reshape(-1, classes)
        )

    labels = target._array.reshape(-1)
    valid = labels != ignore_index
    if bool(cp.any(valid & ((labels < 0) | (labels >= classes)))):
        raise ValueError("nll_loss target contains an invalid class index")

    def forward(log_probs, label_array, weight_array):
        matrix = matrix_view(log_probs)
        labels = label_array.reshape(-1)
        valid = labels != ignore_index
        safe = cp.where(valid, labels, 0)
        class_weights = (
            cp.ones(classes, dtype=matrix.dtype)
            if weight_array is None
            else weight_array
        )
        selected_weights = class_weights[safe]
        losses = -matrix[cp.arange(matrix.shape[0]), safe] * selected_weights * valid
        if reduction == "none":
            return losses.reshape(output_shape)
        if reduction == "sum":
            return losses.sum()
        denominator = cp.sum(selected_weights * valid)
        return cp.where(denominator > 0, losses.sum() / denominator, 0)

    def backward(gradient, _result, arrays):
        log_probs, label_array, weight_array = arrays
        matrix = matrix_view(log_probs)
        labels = label_array.reshape(-1)
        valid = labels != ignore_index
        safe = cp.where(valid, labels, 0)
        class_weights = (
            cp.ones(classes, dtype=matrix.dtype)
            if weight_array is None
            else weight_array
        )
        selected_weights = class_weights[safe]
        grad_matrix = cp.zeros_like(matrix)
        scale = selected_weights * valid
        if reduction == "mean":
            denominator = scale.sum()
            scale = cp.where(denominator > 0, scale / denominator, 0)
        if reduction == "none":
            scale *= gradient.reshape(-1)
        else:
            scale *= gradient
        grad_matrix[cp.arange(matrix.shape[0]), safe] = -scale
        if unbatched:
            grad_input = grad_matrix.reshape(log_probs.shape)
        else:
            moved_shape = (input.shape[0],) + input.shape[2:] + (classes,)
            grad_input = cp.moveaxis(grad_matrix.reshape(moved_shape), -1, 1)
        return grad_input, None, None

    return _ops.apply(
        forward, input, target, weights, backward=backward, name="nll_loss"
    )


def kl_div(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    log_target: bool = False,
) -> Tensor:
    _validate_pair(input, target, "kl_div")
    if reduction not in {"none", "batchmean", "sum", "mean"}:
        raise ValueError("kl_div reduction must be none, batchmean, sum, or mean")
    if not isinstance(log_target, bool):
        raise TypeError("log_target must be a bool")
    if log_target:
        loss = target.exp() * (target - input)
    else:
        from mytorch.tensor import where

        positive = target > 0
        safe_target = where(positive, target, 1.0)
        loss = where(positive, target * (safe_target.log() - input), 0.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "batchmean":
        batch = input.shape[0] if input.ndim > 0 else 1
        return loss.sum() / batch
    return loss.mean()


def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    _validate_pair(input, target, "poisson_nll_loss")
    if not isinstance(log_input, bool) or not isinstance(full, bool):
        raise TypeError("log_input and full must be bools")
    _require_real("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be positive")
    loss = (
        input.exp() - target * input
        if log_input
        else input - target * (input + eps).log()
    )
    if full:
        from mytorch.tensor import where

        stirling = target * target.log() - target + 0.5 * (2 * math.pi * target).log()
        loss = loss + where(target > 1, stirling, 0.0)
    return _reduce_loss(loss, reduction)


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    variance: Tensor,
    reduction: str = "mean",
    *,
    full: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    _validate_pair(input, target, "gaussian_nll_loss")
    if not isinstance(variance, Tensor):
        raise TypeError("gaussian_nll_loss variance must be a Tensor")
    if variance.shape not in {input.shape, input.shape[:-1] + (1,)}:
        raise ValueError(
            "variance must match input shape or have a final singleton dim"
        )
    if bool(cp.any(variance._array < 0)):
        raise ValueError("variance must be non-negative")
    from mytorch.tensor import maximum

    safe_variance = maximum(variance, eps)
    loss = 0.5 * (safe_variance.log() + (input - target).square() / safe_variance)
    if full:
        loss = loss + 0.5 * math.log(2 * math.pi)
    return _reduce_loss(loss, reduction)


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    _validate_pair(input, target, "hinge_embedding_loss")
    _require_real("margin", margin)
    from mytorch.tensor import maximum, where

    loss = where(target == 1, input, maximum(margin - input, 0.0))
    return _reduce_loss(loss, reduction)


def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    _validate_pair(input1, input2, "margin_ranking_loss")
    _validate_pair(input1, target, "margin_ranking_loss")
    from mytorch.tensor import maximum

    return _reduce_loss(maximum(-target * (input1 - input2) + margin, 0.0), reduction)


def soft_margin_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    _validate_pair(input, target, "soft_margin_loss")
    return _reduce_loss(softplus(-target * input), reduction)


def multilabel_soft_margin_loss(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    _validate_pair(input, target, "multilabel_soft_margin_loss")
    if input.ndim < 1:
        raise ValueError("multilabel_soft_margin_loss expects at least 1D input")
    loss = binary_cross_entropy_with_logits(input, target, reduction="none").mean(-1)
    return _reduce_loss(loss, reduction)


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    _validate_pair(input1, input2, "cosine_embedding_loss")
    if input1.ndim not in {1, 2}:
        raise ValueError("cosine_embedding_loss expects (D,) or (N, D) inputs")
    expected = () if input1.ndim == 1 else (input1.shape[0],)
    if target.shape != expected:
        raise ValueError(f"cosine_embedding_loss target must have shape {expected}")
    _require_real("margin", margin)
    from mytorch.tensor import maximum

    cosine = (input1 * input2).sum(-1) / (
        input1.norm(dim=-1) * input2.norm(dim=-1) + 1e-12
    )
    from mytorch.tensor import where

    loss = where(target == 1, 1 - cosine, maximum(cosine - margin, 0.0))
    return _reduce_loss(loss, reduction)


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    _validate_pair(anchor, positive, "triplet_margin_loss")
    _validate_pair(anchor, negative, "triplet_margin_loss")
    if anchor.ndim not in {1, 2}:
        raise ValueError("triplet_margin_loss expects (D,) or (N, D) inputs")
    _require_real("margin", margin)
    if margin <= 0:
        raise ValueError("margin must be positive")
    from mytorch.tensor import maximum, minimum

    positive_distance = (abs(anchor - positive) + eps).norm(p=p, dim=-1)
    negative_distance = (abs(anchor - negative) + eps).norm(p=p, dim=-1)
    if swap:
        negative_distance = minimum(
            negative_distance, (abs(positive - negative) + eps).norm(p=p, dim=-1)
        )
    return _reduce_loss(
        maximum(positive_distance - negative_distance + margin, 0.0), reduction
    )


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    reduction: str = "mean",
    *,
    weight: Tensor | None = None,
) -> Tensor:
    _validate_classification(input, target, "multi_margin_loss")
    if input.ndim not in {1, 2}:
        raise ValueError("multi_margin_loss expects (C,) or (N, C) input")
    if p not in {1, 2}:
        raise ValueError("multi_margin_loss p must be 1 or 2")
    _require_real("margin", margin)
    classes = input.shape[-1]
    weights = _validate_class_weight(weight, classes, input, "multi_margin_loss")
    unbatched = input.ndim == 1
    if bool(cp.any((target._array < 0) | (target._array >= classes))):
        raise ValueError("multi_margin_loss target contains an invalid class index")

    def forward(scores, labels, weight_array):
        matrix = scores.reshape(1, -1) if unbatched else scores
        indices = labels.reshape(-1)
        correct = matrix[cp.arange(matrix.shape[0]), indices]
        terms = cp.maximum(0, margin - correct[:, None] + matrix) ** p
        terms[cp.arange(matrix.shape[0]), indices] = 0
        if weight_array is not None:
            terms *= weight_array[indices, None]
        losses = terms.sum(axis=1) / classes
        if reduction == "none":
            return losses.reshape(()) if unbatched else losses
        return losses.sum() if reduction == "sum" else losses.mean()

    def backward(gradient, _result, arrays):
        scores, labels, weight_array = arrays
        matrix = scores.reshape(1, -1) if unbatched else scores
        indices = labels.reshape(-1)
        correct = matrix[cp.arange(matrix.shape[0]), indices]
        raw = margin - correct[:, None] + matrix
        derivative = (raw > 0) * (1 if p == 1 else 2 * cp.maximum(raw, 0))
        derivative[cp.arange(matrix.shape[0]), indices] = 0
        if weight_array is not None:
            derivative *= weight_array[indices, None]
        derivative /= classes
        derivative[cp.arange(matrix.shape[0]), indices] = -derivative.sum(axis=1)
        if reduction == "mean":
            derivative /= matrix.shape[0]
        derivative *= (
            gradient.reshape(-1, 1)
            if reduction == "none" and not unbatched
            else gradient
        )
        return derivative.reshape(scores.shape), None, None

    return _ops.apply(
        forward, input, target, weights, backward=backward, name="multi_margin_loss"
    )


def sigmoid_focal_loss(
    input: Tensor,
    target: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    _validate_pair(input, target, "sigmoid_focal_loss")
    _require_real("alpha", alpha)
    _require_real("gamma", gamma)
    if not 0 <= alpha <= 1 or gamma < 0:
        raise ValueError("focal loss requires alpha in [0, 1] and gamma >= 0")
    probabilities = sigmoid(input)
    base = binary_cross_entropy_with_logits(input, target, reduction="none")
    probability_target = probabilities * target + (1 - probabilities) * (1 - target)
    alpha_target = alpha * target + (1 - alpha) * (1 - target)
    return _reduce_loss(
        alpha_target * (1 - probability_target) ** gamma * base, reduction
    )


def dice_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> Tensor:
    _validate_pair(input, target, "dice_loss")
    if input.ndim < 1:
        raise ValueError("dice_loss expects at least 1D input")
    if not isinstance(from_logits, bool):
        raise TypeError("from_logits must be a bool")
    _require_real("smooth", smooth)
    _require_real("eps", eps)
    if smooth < 0 or eps <= 0:
        raise ValueError("smooth must be non-negative and eps must be positive")
    probabilities = sigmoid(input) if from_logits else input
    axes = tuple(range(1, input.ndim)) if input.ndim > 1 else (0,)
    intersection = (probabilities * target).sum(axes)
    denominator = probabilities.square().sum(axes) + target.square().sum(axes)
    loss = 1 - (2 * intersection + smooth) / (denominator + smooth + eps)
    return _reduce_loss(loss, reduction)


def contrastive_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    _validate_pair(input1, input2, "contrastive_loss")
    if input1.ndim not in {1, 2}:
        raise ValueError("contrastive_loss expects (D,) or (N, D) inputs")
    expected = () if input1.ndim == 1 else (input1.shape[0],)
    if target.shape != expected:
        raise ValueError(f"contrastive_loss target must have shape {expected}")
    _require_real("margin", margin)
    if margin <= 0:
        raise ValueError("margin must be positive")
    from mytorch.tensor import maximum

    distance = (input1 - input2).norm(dim=-1)
    loss = 0.5 * (
        (1 - target) * distance.square()
        + target * maximum(margin - distance, 0.0).square()
    )
    return _reduce_loss(loss, reduction)


def _validate_pair(input: Tensor, target: Tensor, operation: str) -> None:
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError(f"{operation} expects Tensor inputs")
    _ops.require_floating(input, operation)
    _ops.require_floating(target, operation)
    if input.shape != target.shape:
        raise ValueError(
            f"{operation} requires matching shapes, got "
            f"{input.shape} and {target.shape}"
        )
    if input._device_index != target._device_index:
        raise ValueError(f"{operation} inputs must be on the same CUDA device")


def _reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    _validate_reduction(reduction)
    if reduction == "none":
        return loss
    return loss.sum() if reduction == "sum" else loss.mean()


def _validate_classification(input: Tensor, target: Tensor, operation: str) -> None:
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError(f"{operation} expects Tensor input and target")
    _ops.require_floating(input, operation)
    if target.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{operation} target must have an integer dtype")
    if input._device_index != target._device_index:
        raise ValueError(f"{operation} inputs must be on the same CUDA device")


def _classification_shape(
    input: Tensor, target: Tensor
) -> tuple[int, tuple[int, ...], bool]:
    if input.ndim == 1:
        if target.ndim != 0:
            raise ValueError("unbatched classification target must be scalar")
        return input.shape[0], (), True
    if input.ndim < 2:
        raise ValueError("classification input must have a class dimension")
    expected = (input.shape[0],) + input.shape[2:]
    if target.shape != expected:
        raise ValueError(f"classification target must have shape {expected}")
    return input.shape[1], expected, False


def _validate_class_weight(
    weight: Tensor | None, classes: int, input: Tensor, operation: str
) -> Tensor | None:
    if weight is None:
        return None
    if not isinstance(weight, Tensor) or weight.shape != (classes,):
        raise ValueError(f"{operation} weight must have shape {(classes,)}")
    if weight.requires_grad:
        raise ValueError(f"{operation} weight cannot require gradients")
    if weight._device_index != input._device_index:
        raise ValueError(f"{operation} weight must be on the input device")
    return weight


def _validate_optional_weight(
    weight: Tensor | None, input: Tensor, operation: str
) -> Tensor | None:
    if weight is None:
        return None
    if not isinstance(weight, Tensor):
        raise TypeError(f"{operation} weight must be a Tensor")
    if weight.requires_grad:
        raise ValueError(f"{operation} weight cannot require gradients")
    if weight._device_index != input._device_index:
        raise ValueError(f"{operation} weight must be on the input device")
    try:
        cp.broadcast_shapes(input.shape, weight.shape)
    except ValueError as exc:
        raise ValueError(f"{operation} weight is not broadcastable to input") from exc
    return weight
