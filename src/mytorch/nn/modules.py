"""Compatibility facade for neural-network building blocks."""

from ._modules import activations as _activations
from ._modules import base as _base
from ._modules import containers as _containers
from ._modules import linear as _linear
from ._modules import losses as _losses

Parameter = _base.Parameter
Module = _base.Module

Linear = _linear.Linear

Sequential = _containers.Sequential
ModuleList = _containers.ModuleList
ParameterList = _containers.ParameterList
Identity = _containers.Identity
Flatten = _containers.Flatten

ReLU = _activations.ReLU
Sigmoid = _activations.Sigmoid
Tanh = _activations.Tanh
SiLU = _activations.SiLU
LeakyReLU = _activations.LeakyReLU
Softmax = _activations.Softmax
LogSoftmax = _activations.LogSoftmax
GELU = _activations.GELU
Softplus = _activations.Softplus
Threshold = _activations.Threshold
Hardtanh = _activations.Hardtanh
ReLU6 = _activations.ReLU6
ELU = _activations.ELU
SELU = _activations.SELU
CELU = _activations.CELU
PReLU = _activations.PReLU
RReLU = _activations.RReLU
Hardsigmoid = _activations.Hardsigmoid
Hardswish = _activations.Hardswish
Hardshrink = _activations.Hardshrink
Softshrink = _activations.Softshrink
Tanhshrink = _activations.Tanhshrink
LogSigmoid = _activations.LogSigmoid
Softsign = _activations.Softsign
Mish = _activations.Mish
Softmin = _activations.Softmin
GLU = _activations.GLU
ReGLU = _activations.ReGLU
GEGLU = _activations.GEGLU
SwiGLU = _activations.SwiGLU

MSELoss = _losses.MSELoss
CrossEntropyLoss = _losses.CrossEntropyLoss
L1Loss = _losses.L1Loss
SmoothL1Loss = _losses.SmoothL1Loss
HuberLoss = _losses.HuberLoss
BCELoss = _losses.BCELoss
BCEWithLogitsLoss = _losses.BCEWithLogitsLoss
NLLLoss = _losses.NLLLoss
KLDivLoss = _losses.KLDivLoss
PoissonNLLLoss = _losses.PoissonNLLLoss
GaussianNLLLoss = _losses.GaussianNLLLoss
HingeEmbeddingLoss = _losses.HingeEmbeddingLoss
MarginRankingLoss = _losses.MarginRankingLoss
SoftMarginLoss = _losses.SoftMarginLoss
MultiLabelSoftMarginLoss = _losses.MultiLabelSoftMarginLoss
CosineEmbeddingLoss = _losses.CosineEmbeddingLoss
TripletMarginLoss = _losses.TripletMarginLoss
MultiMarginLoss = _losses.MultiMarginLoss
FocalLoss = _losses.FocalLoss
DiceLoss = _losses.DiceLoss
ContrastiveLoss = _losses.ContrastiveLoss

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CELU",
    "ContrastiveLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "DiceLoss",
    "ELU",
    "Flatten",
    "FocalLoss",
    "GEGLU",
    "GELU",
    "GLU",
    "GaussianNLLLoss",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "KLDivLoss",
    "L1Loss",
    "LeakyReLU",
    "Linear",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "Mish",
    "Module",
    "ModuleList",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "PReLU",
    "Parameter",
    "ParameterList",
    "PoissonNLLLoss",
    "RReLU",
    "ReGLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SwiGLU",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "TripletMarginLoss",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
