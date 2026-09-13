"""Loss Module wrappers."""

from __future__ import annotations

from mytorch.tensor import Tensor

from ..functional import losses as F
from .base import Module


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
