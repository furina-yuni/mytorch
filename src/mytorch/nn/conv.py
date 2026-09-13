"""Compatibility facade for convolution and pooling layers."""

from ._modules import convolution as _convolution
from ._modules import pooling as _pooling

convolution = _convolution.convolution
convolution_transpose = _convolution.convolution_transpose
Conv1d = _convolution.Conv1d
Conv2d = _convolution.Conv2d
Conv3d = _convolution.Conv3d
ConvTranspose1d = _convolution.ConvTranspose1d
ConvTranspose2d = _convolution.ConvTranspose2d
ConvTranspose3d = _convolution.ConvTranspose3d

pool_nd = _pooling.pool_nd
adaptive_pool_nd = _pooling.adaptive_pool_nd
MaxPool1d = _pooling.MaxPool1d
MaxPool2d = _pooling.MaxPool2d
MaxPool3d = _pooling.MaxPool3d
AvgPool1d = _pooling.AvgPool1d
AvgPool2d = _pooling.AvgPool2d
AvgPool3d = _pooling.AvgPool3d
AdaptiveMaxPool1d = _pooling.AdaptiveMaxPool1d
AdaptiveMaxPool2d = _pooling.AdaptiveMaxPool2d
AdaptiveMaxPool3d = _pooling.AdaptiveMaxPool3d
AdaptiveAvgPool1d = _pooling.AdaptiveAvgPool1d
AdaptiveAvgPool2d = _pooling.AdaptiveAvgPool2d
AdaptiveAvgPool3d = _pooling.AdaptiveAvgPool3d

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "adaptive_pool_nd",
    "convolution",
    "convolution_transpose",
    "pool_nd",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
