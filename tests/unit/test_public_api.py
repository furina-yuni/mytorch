"""Compatibility checks for the documented public surface."""

from __future__ import annotations

import inspect

import mytorch as mt
from mytorch.nn import functional as F


def test_all_exports_are_available() -> None:
    assert all(hasattr(mt, name) for name in mt.__all__)
    assert all(hasattr(mt.nn, name) for name in mt.nn.__all__)
    assert all(hasattr(F, name) for name in F.__all__)
    assert all(hasattr(mt.optim, name) for name in mt.optim.__all__)


def test_compatibility_modules_reexport_canonical_objects() -> None:
    from mytorch.nn.attention import MultiheadAttention
    from mytorch.nn.basic import LayerNorm
    from mytorch.nn.conv import Conv2d
    from mytorch.nn.efficient import BitLinear
    from mytorch.nn.modules import Linear
    from mytorch.optim.other import Adafactor
    from mytorch.tensor import tensor

    assert MultiheadAttention is mt.nn.MultiheadAttention
    assert LayerNorm is mt.nn.LayerNorm
    assert Conv2d is mt.nn.Conv2d
    assert BitLinear is mt.nn.BitLinear
    assert Linear is mt.nn.Linear
    assert Adafactor is mt.optim.Adafactor
    assert tensor is mt.tensor
    assert Linear.__module__ == "mytorch.nn.modules"
    assert Conv2d.__module__ == "mytorch.nn.conv"
    assert MultiheadAttention.__module__ == "mytorch.nn.attention"
    assert Adafactor.__module__ == "mytorch.optim.other"
    assert tensor.__module__ == "mytorch.tensor"


def test_representative_signatures_remain_stable() -> None:
    expected = {
        mt.Tensor: ["data", "dtype", "device", "requires_grad"],
        F.relu: ["input"],
        F.cross_entropy: [
            "input",
            "target",
            "reduction",
            "weight",
            "ignore_index",
            "label_smoothing",
        ],
        mt.nn.Linear: ["in_features", "out_features", "bias", "device", "dtype"],
        mt.nn.Conv2d: [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
            "device",
            "dtype",
        ],
        mt.nn.MultiheadAttention: [
            "embed_dim",
            "num_heads",
            "dropout",
            "bias",
            "kdim",
            "vdim",
            "batch_first",
            "device",
            "dtype",
        ],
        mt.optim.SGD: [
            "params",
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "maximize",
        ],
        mt.optim.AdamW: [
            "params",
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
        ],
    }
    for public_object, names in expected.items():
        assert list(inspect.signature(public_object).parameters) == names
