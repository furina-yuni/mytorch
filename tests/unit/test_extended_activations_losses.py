from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt
from mytorch.nn import functional as F
from tests.helpers import finite_difference

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "function",
    [
        F.elu,
        F.selu,
        F.celu,
        F.hardsigmoid,
        F.hardswish,
        F.tanhshrink,
        F.logsigmoid,
        F.softsign,
        F.mish,
    ],
)
def test_smooth_activation_gradients(function) -> None:
    source = np.array([-1.3, -0.2, 0.7, 2.0], dtype=np.float64)
    value = mt.tensor(source, requires_grad=True)
    function(value).sum().backward()
    expected = finite_difference(function, source)
    np.testing.assert_allclose(value.grad.numpy(), expected, rtol=2e-4, atol=2e-5)


def test_piecewise_activations_and_extreme_stability() -> None:
    value = mt.tensor([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    functions = [
        F.threshold,
        F.hardtanh,
        F.relu6,
        F.hardshrink,
        F.softshrink,
        F.logsigmoid,
        F.mish,
    ]
    results = [
        functions[0](value, 0.0, -1.0),
        *(function(value) for function in functions[1:]),
    ]
    assert all(np.isfinite(result.numpy()).all() for result in results)
    np.testing.assert_allclose(F.relu6(value).numpy(), [0, 0, 0, 2, 6])


def test_prelu_rrelu_and_gated_modules() -> None:
    prelu = mt.nn.PReLU(2, init=0.25, dtype=mt.float64)
    value = mt.tensor([[[-2.0], [3.0]]], dtype=mt.float64, requires_grad=True)
    prelu(value).sum().backward()
    np.testing.assert_allclose(value.grad.numpy(), [[[0.25], [1.0]]])
    np.testing.assert_allclose(prelu.weight.grad.numpy(), [-2.0, 0.0])

    mt.manual_seed(42)
    layer = mt.nn.RReLU(0.1, 0.3)
    training = layer(mt.tensor([-2.0, -1.0])).numpy()
    mt.manual_seed(42)
    np.testing.assert_allclose(training, layer(mt.tensor([-2.0, -1.0])).numpy())
    layer.eval()
    np.testing.assert_allclose(layer(mt.tensor([-2.0, -1.0])).numpy(), [-0.4, -0.2])

    gated = mt.tensor([[1.0, 2.0, 0.0, 1.0]])
    assert F.glu(gated).shape == (1, 2)
    assert F.reglu(gated).shape == (1, 2)
    assert F.geglu(gated).shape == (1, 2)
    assert F.swiglu(gated).shape == (1, 2)
    with pytest.raises(ValueError, match="even"):
        F.glu(mt.ones(2, 3))


def test_regression_and_probability_losses() -> None:
    prediction = mt.tensor([1.0, 3.0], dtype=mt.float64, requires_grad=True)
    target = mt.tensor([2.0, 1.0], dtype=mt.float64)
    assert F.l1_loss(prediction, target).item() == pytest.approx(1.5)
    assert F.smooth_l1_loss(prediction, target).item() == pytest.approx(1.0)
    assert F.huber_loss(prediction, target).item() == pytest.approx(1.0)

    logits = mt.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    labels = mt.tensor([0.0, 1.0, 1.0])
    stable = F.binary_cross_entropy_with_logits(logits, labels)
    focal = F.sigmoid_focal_loss(logits, labels)
    (stable + focal.mean()).backward()
    assert np.isfinite(stable.item()) and np.isfinite(focal.numpy()).all()
    assert np.isfinite(logits.grad.numpy()).all()

    log_prob = F.log_softmax(mt.tensor([[2.0, 1.0]], requires_grad=True), dim=1)
    class_target = mt.tensor(np.array([0], dtype=np.int64))
    assert F.nll_loss(log_prob, class_target).item() > 0


def test_extended_cross_entropy_nd_weight_ignore_and_smoothing() -> None:
    logits = mt.tensor(
        np.array([[[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]], dtype=np.float64),
        requires_grad=True,
    )
    target = mt.tensor(np.array([[0, -100]], dtype=np.int64))
    weight = mt.tensor([1.0, 2.0, 3.0], dtype=mt.float64)
    loss = F.cross_entropy(
        logits, target, weight=weight, ignore_index=-100, label_smoothing=0.1
    )
    loss.backward()
    assert np.isfinite(loss.item())
    assert logits.grad.shape == logits.shape
    np.testing.assert_allclose(logits.grad.numpy()[:, :, 1], 0.0)


def test_distance_segmentation_and_margin_losses() -> None:
    left = mt.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    right = mt.tensor([[1.0, 0.0], [1.0, 0.0]])
    similarity = mt.tensor([1.0, -1.0])
    losses = [
        F.cosine_embedding_loss(left, right, similarity),
        F.triplet_margin_loss(left, right, mt.tensor([[0.0, 1.0], [-1.0, 0.0]])),
        F.contrastive_loss(left, right, mt.tensor([0.0, 1.0])),
        F.dice_loss(mt.tensor([[-1000.0, 1000.0]]), mt.tensor([[0.0, 1.0]])),
        F.margin_ranking_loss(mt.tensor([2.0]), mt.tensor([1.0]), mt.tensor([1.0])),
        F.hinge_embedding_loss(mt.tensor([0.5]), mt.tensor([1.0])),
        F.soft_margin_loss(mt.tensor([1.0]), mt.tensor([1.0])),
        F.multilabel_soft_margin_loss(
            mt.tensor([[1.0, -1.0]]), mt.tensor([[1.0, 0.0]])
        ),
    ]
    assert all(np.isfinite(loss.item()) for loss in losses)
    sum(losses[:3]).backward()
    assert np.isfinite(left.grad.numpy()).all()


def test_distribution_losses_and_loss_modules() -> None:
    input = mt.tensor([0.2, 0.8], requires_grad=True)
    target = mt.tensor([0.0, 1.0])
    variance = mt.tensor([1.0, 2.0])
    values = [
        F.poisson_nll_loss(input, target, log_input=False),
        F.gaussian_nll_loss(input, target, variance),
        F.kl_div(F.log_softmax(input, dim=0), F.softmax(target, dim=0)),
        mt.nn.BCEWithLogitsLoss()(input, target),
        mt.nn.DiceLoss()(input, target),
        mt.nn.FocalLoss()(input, target),
    ]
    assert all(np.isfinite(value.numpy()).all() for value in values)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_common_loss_reductions_and_multi_margin(reduction: str) -> None:
    input = mt.tensor([0.2, 0.8], requires_grad=True)
    target = mt.tensor([0.0, 1.0])
    losses = [
        F.l1_loss(input, target, reduction),
        F.smooth_l1_loss(input, target, reduction),
        F.huber_loss(input, target, reduction),
        F.binary_cross_entropy(input, target, reduction),
        F.binary_cross_entropy_with_logits(input, target, reduction),
        F.soft_margin_loss(input, target * 2 - 1, reduction),
    ]
    expected_shape = input.shape if reduction == "none" else ()
    assert all(loss.shape == expected_shape for loss in losses)
    sum(loss.sum() for loss in losses).backward()
    assert np.isfinite(input.grad.numpy()).all()

    scores = mt.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    labels = mt.tensor(np.array([0], dtype=np.int64))
    margin = F.multi_margin_loss(scores, labels, p=2, reduction=reduction)
    assert margin.shape == ((1,) if reduction == "none" else ())
    margin.sum().backward()
    assert np.isfinite(scores.grad.numpy()).all()
