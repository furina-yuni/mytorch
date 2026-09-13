# ruff: noqa: F401
"""Stateful neural-network layers grouped by architecture."""

from .activations import (
    CELU,
    ELU,
    GEGLU,
    GELU,
    GLU,
    SELU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReGLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    SiLU,
    Softmax,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    SwiGLU,
    Tanh,
    Tanhshrink,
    Threshold,
)
from .attention import MultiheadAttention
from .base import Module, Parameter
from .containers import Flatten, Identity, ModuleList, ParameterList, Sequential
from .convolution import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .embedding import Embedding, EmbeddingBag
from .linear import Linear
from .linear_extra import Bilinear, LazyLinear
from .losses import (
    BCELoss,
    BCEWithLogitsLoss,
    ContrastiveLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
)
from .low_rank import LoRALinear, LowRankLinear, SwiGLUFeedForward
from .moe import SparseMoE, TopKRouter
from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    RMSNorm,
)
from .pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from .quantized import BitLinear, Int4WeightOnlyLinear, Int8Linear, PackedBitLinear
from .regularization import Dropout, Dropout1d, Dropout2d, Dropout3d, StochasticDepth
from .rotary import RotaryEmbedding, apply_rotary_pos_emb
from .transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = [name for name in globals() if not name.startswith("_")]
