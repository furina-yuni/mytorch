"""Layer operations for :mod:`mytorch.nn.functional`."""

from __future__ import annotations

import math

import cupy as cp

from mytorch import _ops, _random
from mytorch.tensor import Tensor


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
        result = result + bias.to(dtype=result.dtype)
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
        result = result + bias.to(dtype=result.dtype)
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
        mask = tensor(
            _random.random(input.shape, device=input._device_index) >= p,
            device=input.device,
        )
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
        mask = tensor(
            _random.random(shape, device=input._device_index) >= p,
            device=input.device,
        )
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
        mask = tensor(
            _random.random(shape, device=input._device_index) >= p,
            device=input.device,
        )
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
    from ..modules.convolution import convolution

    return convolution(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from ..modules.convolution import convolution

    return convolution(input, weight, bias, stride, padding, dilation, groups)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from ..modules.convolution import convolution

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
    from ..modules.convolution import convolution_transpose

    return convolution_transpose(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )


conv_transpose2d = conv_transpose1d
conv_transpose3d = conv_transpose1d


def max_pool1d(input, kernel_size, stride=None, padding=0):
    from ..modules.pooling import pool_nd

    return pool_nd(input, kernel_size, stride, padding, "max")


max_pool2d = max_pool1d
max_pool3d = max_pool1d


def avg_pool1d(input, kernel_size, stride=None, padding=0):
    from ..modules.pooling import pool_nd

    return pool_nd(input, kernel_size, stride, padding, "avg")


avg_pool2d = avg_pool1d
avg_pool3d = avg_pool1d


def adaptive_max_pool1d(input, output_size):
    from ..modules.pooling import adaptive_pool_nd

    return adaptive_pool_nd(input, output_size, "max")


adaptive_max_pool2d = adaptive_max_pool1d
adaptive_max_pool3d = adaptive_max_pool1d


def adaptive_avg_pool1d(input, output_size):
    from ..modules.pooling import adaptive_pool_nd

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
    from ..modules.attention import scaled_dot_product_attention as implementation

    return implementation(query, key, value, attn_mask, dropout_p, is_causal, scale)
