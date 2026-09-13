"""Compatibility facade for efficient neural-network layers."""

from ._modules import low_rank as _low_rank
from ._modules import moe as _moe
from ._modules import quantized as _quantized

LowRankLinear = _low_rank.LowRankLinear
LoRALinear = _low_rank.LoRALinear
SwiGLUFeedForward = _low_rank.SwiGLUFeedForward

Int8Linear = _quantized.Int8Linear
Int4WeightOnlyLinear = _quantized.Int4WeightOnlyLinear
BitLinear = _quantized.BitLinear
PackedBitLinear = _quantized.PackedBitLinear

TopKRouter = _moe.TopKRouter
SparseMoE = _moe.SparseMoE

__all__ = [
    "BitLinear",
    "Int4WeightOnlyLinear",
    "Int8Linear",
    "LoRALinear",
    "LowRankLinear",
    "PackedBitLinear",
    "SparseMoE",
    "SwiGLUFeedForward",
    "TopKRouter",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
