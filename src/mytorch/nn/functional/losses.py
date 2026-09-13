"""Loss functions for :mod:`mytorch.nn.functional`."""

from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.special import expit

from mytorch import _ops
from mytorch.tensor import Tensor

from .activations import _require_real, sigmoid, softplus


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
