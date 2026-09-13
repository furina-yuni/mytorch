"""Compatibility facade for attention and Transformer modules."""

from ._modules import attention as _attention
from ._modules import rotary as _rotary
from ._modules import transformer as _transformer

scaled_dot_product_attention = _attention.scaled_dot_product_attention
MultiheadAttention = _attention.MultiheadAttention

TransformerEncoderLayer = _transformer.TransformerEncoderLayer
TransformerEncoder = _transformer.TransformerEncoder

RotaryEmbedding = _rotary.RotaryEmbedding
apply_rotary_pos_emb = _rotary.apply_rotary_pos_emb
stack_rotary = _rotary.stack_rotary

__all__ = [
    "MultiheadAttention",
    "RotaryEmbedding",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "apply_rotary_pos_emb",
    "scaled_dot_product_attention",
    "stack_rotary",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
