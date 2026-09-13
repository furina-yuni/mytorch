"""Public functional neural-network API."""

from ._functional import activations as _activations
from ._functional import layers as _layers
from ._functional import losses as _losses

relu = _activations.relu
leaky_relu = _activations.leaky_relu
sigmoid = _activations.sigmoid
tanh = _activations.tanh
softmax = _activations.softmax
log_softmax = _activations.log_softmax
gelu = _activations.gelu
silu = _activations.silu
softplus = _activations.softplus
threshold = _activations.threshold
hardtanh = _activations.hardtanh
relu6 = _activations.relu6
elu = _activations.elu
selu = _activations.selu
celu = _activations.celu
prelu = _activations.prelu
rrelu = _activations.rrelu
hardsigmoid = _activations.hardsigmoid
hardswish = _activations.hardswish
hardshrink = _activations.hardshrink
softshrink = _activations.softshrink
tanhshrink = _activations.tanhshrink
logsigmoid = _activations.logsigmoid
softsign = _activations.softsign
softmin = _activations.softmin
mish = _activations.mish
glu = _activations.glu
reglu = _activations.reglu
geglu = _activations.geglu
swiglu = _activations.swiglu

linear = _layers.linear
bilinear = _layers.bilinear
dropout = _layers.dropout
feature_dropout = _layers.feature_dropout
dropout1d = _layers.dropout1d
dropout2d = _layers.dropout2d
dropout3d = _layers.dropout3d
stochastic_depth = _layers.stochastic_depth
embedding = _layers.embedding
embedding_bag = _layers.embedding_bag
layer_norm = _layers.layer_norm
rms_norm = _layers.rms_norm
group_norm = _layers.group_norm
batch_norm = _layers.batch_norm
instance_norm = _layers.instance_norm
conv1d = _layers.conv1d
conv2d = _layers.conv2d
conv3d = _layers.conv3d
conv_transpose1d = _layers.conv_transpose1d
conv_transpose2d = _layers.conv_transpose2d
conv_transpose3d = _layers.conv_transpose3d
max_pool1d = _layers.max_pool1d
max_pool2d = _layers.max_pool2d
max_pool3d = _layers.max_pool3d
avg_pool1d = _layers.avg_pool1d
avg_pool2d = _layers.avg_pool2d
avg_pool3d = _layers.avg_pool3d
adaptive_max_pool1d = _layers.adaptive_max_pool1d
adaptive_max_pool2d = _layers.adaptive_max_pool2d
adaptive_max_pool3d = _layers.adaptive_max_pool3d
adaptive_avg_pool1d = _layers.adaptive_avg_pool1d
adaptive_avg_pool2d = _layers.adaptive_avg_pool2d
adaptive_avg_pool3d = _layers.adaptive_avg_pool3d
scaled_dot_product_attention = _layers.scaled_dot_product_attention

mse_loss = _losses.mse_loss
cross_entropy = _losses.cross_entropy
l1_loss = _losses.l1_loss
smooth_l1_loss = _losses.smooth_l1_loss
huber_loss = _losses.huber_loss
binary_cross_entropy = _losses.binary_cross_entropy
binary_cross_entropy_with_logits = _losses.binary_cross_entropy_with_logits
nll_loss = _losses.nll_loss
kl_div = _losses.kl_div
poisson_nll_loss = _losses.poisson_nll_loss
gaussian_nll_loss = _losses.gaussian_nll_loss
hinge_embedding_loss = _losses.hinge_embedding_loss
margin_ranking_loss = _losses.margin_ranking_loss
soft_margin_loss = _losses.soft_margin_loss
multilabel_soft_margin_loss = _losses.multilabel_soft_margin_loss
cosine_embedding_loss = _losses.cosine_embedding_loss
triplet_margin_loss = _losses.triplet_margin_loss
multi_margin_loss = _losses.multi_margin_loss
sigmoid_focal_loss = _losses.sigmoid_focal_loss
dice_loss = _losses.dice_loss
contrastive_loss = _losses.contrastive_loss

__all__ = [
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "batch_norm",
    "bilinear",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "celu",
    "contrastive_loss",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "cosine_embedding_loss",
    "cross_entropy",
    "dice_loss",
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "elu",
    "embedding",
    "embedding_bag",
    "feature_dropout",
    "gaussian_nll_loss",
    "geglu",
    "gelu",
    "glu",
    "group_norm",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "instance_norm",
    "kl_div",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "logsigmoid",
    "margin_ranking_loss",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "mish",
    "mse_loss",
    "multi_margin_loss",
    "multilabel_soft_margin_loss",
    "nll_loss",
    "poisson_nll_loss",
    "prelu",
    "reglu",
    "relu",
    "relu6",
    "rms_norm",
    "rrelu",
    "scaled_dot_product_attention",
    "selu",
    "sigmoid",
    "sigmoid_focal_loss",
    "silu",
    "smooth_l1_loss",
    "soft_margin_loss",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "softsign",
    "stochastic_depth",
    "swiglu",
    "tanh",
    "tanhshrink",
    "threshold",
    "triplet_margin_loss",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
