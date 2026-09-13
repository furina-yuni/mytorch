"""Compatibility facade for foundational neural-network layers."""

from ._modules import embedding as _embedding
from ._modules import linear_extra as _linear_extra
from ._modules import normalization as _normalization
from ._modules import regularization as _regularization

Bilinear = _linear_extra.Bilinear
LazyLinear = _linear_extra.LazyLinear

Embedding = _embedding.Embedding
EmbeddingBag = _embedding.EmbeddingBag

Dropout = _regularization.Dropout
Dropout1d = _regularization.Dropout1d
Dropout2d = _regularization.Dropout2d
Dropout3d = _regularization.Dropout3d
StochasticDepth = _regularization.StochasticDepth

LayerNorm = _normalization.LayerNorm
RMSNorm = _normalization.RMSNorm
GroupNorm = _normalization.GroupNorm
BatchNorm1d = _normalization.BatchNorm1d
BatchNorm2d = _normalization.BatchNorm2d
BatchNorm3d = _normalization.BatchNorm3d
InstanceNorm1d = _normalization.InstanceNorm1d
InstanceNorm2d = _normalization.InstanceNorm2d
InstanceNorm3d = _normalization.InstanceNorm3d

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LazyLinear",
    "RMSNorm",
    "StochasticDepth",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
