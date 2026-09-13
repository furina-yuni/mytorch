from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt
from mytorch.nn import functional as F

pytestmark = pytest.mark.gpu


def test_module_registration_linear_and_sequential() -> None:
    model = mt.nn.Sequential(
        mt.nn.Linear(3, 4),
        mt.nn.ReLU(),
        mt.nn.Linear(4, 2, bias=False),
    )
    names = [name for name, _ in model.named_parameters()]

    assert names == ["0.weight", "0.bias", "2.weight"]
    assert len(model) == 3
    assert isinstance(model[1], mt.nn.ReLU)
    assert model(mt.ones(5, 3)).shape == (5, 2)
    model.eval()
    assert not model.training and all(not child.training for child in model.children())


def test_mse_and_cross_entropy_forward_backward() -> None:
    prediction = mt.tensor([[1.0], [4.0]], requires_grad=True)
    target = mt.tensor([[2.0], [2.0]])
    loss = F.mse_loss(prediction, target)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    np.testing.assert_allclose(prediction.grad.numpy(), [[-1.0], [2.0]])

    logits = mt.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], requires_grad=True)
    labels = mt.tensor(np.array([0, 2], dtype=np.int64))
    cross_entropy = F.cross_entropy(logits, labels)
    cross_entropy.backward()
    expected_probabilities = np.exp(logits.numpy())
    expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
    expected_probabilities[np.arange(2), [0, 2]] -= 1
    np.testing.assert_allclose(
        logits.grad.numpy(), expected_probabilities / 2, rtol=1e-6, atol=1e-6
    )


def test_sgd_weight_decay_momentum_and_zero_grad() -> None:
    parameter = mt.nn.Parameter([2.0])
    optimizer = mt.optim.SGD([parameter], lr=0.1, momentum=0.5, weight_decay=0.1)

    parameter.square().sum().backward()
    optimizer.step()
    np.testing.assert_allclose(parameter.numpy(), [1.58])
    optimizer.zero_grad(set_to_none=False)
    np.testing.assert_allclose(parameter.grad.numpy(), [0.0])

    parameter.square().sum().backward()
    optimizer.step()
    np.testing.assert_allclose(parameter.numpy(), [1.0382], rtol=1e-6)
    optimizer.zero_grad()
    assert parameter.grad is None


def test_gpu_linear_regression_training_converges() -> None:
    mt.manual_seed(7)
    inputs = mt.linspace(-1, 1, 64).reshape(32, 2)
    targets = inputs @ mt.tensor([[3.0], [-2.0]]) + 0.5
    model = mt.nn.Linear(2, 1)
    optimizer = mt.optim.SGD(model.parameters(), lr=0.15)
    initial = F.mse_loss(model(inputs), targets).item()

    for _ in range(150):
        optimizer.zero_grad()
        loss = F.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    final = F.mse_loss(model(inputs), targets).item()
    assert final < 1e-4
    assert final < initial * 0.01


def test_gpu_linear_classifier_training_reaches_high_accuracy() -> None:
    mt.manual_seed(11)
    source = np.array(
        [[-2, -1], [-1, -2], [-2, -2], [-1, -1], [1, 1], [2, 1], [1, 2], [2, 2]],
        dtype=np.float32,
    )
    inputs = mt.tensor(source)
    labels = mt.tensor(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64))
    model = mt.nn.Linear(2, 2)
    optimizer = mt.optim.SGD(model.parameters(), lr=0.2)
    initial = F.cross_entropy(model(inputs), labels).item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()

    logits = model(inputs)
    final = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1).numpy() == labels.numpy()).mean()
    assert final < initial
    assert accuracy >= 0.95
