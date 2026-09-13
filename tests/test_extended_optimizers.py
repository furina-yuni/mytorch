from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


OPTIMIZERS = [
    mt.optim.SGD,
    mt.optim.Adagrad,
    mt.optim.RMSprop,
    mt.optim.Adadelta,
    mt.optim.Adam,
    mt.optim.AdamW,
    mt.optim.Adamax,
    mt.optim.NAdam,
    mt.optim.RAdam,
    mt.optim.ASGD,
    mt.optim.Rprop,
    mt.optim.Adafactor,
    mt.optim.Lion,
]


def _apply_gradients(parameter, optimizer, gradients) -> None:
    for gradient in gradients:
        optimizer.zero_grad()
        (parameter * mt.tensor(gradient, dtype=mt.float64)).sum().backward()
        optimizer.step()


def _adam_reference(gradients, *, lr, weight_decay=0.0, decoupled=False):
    value = np.array([1.0, -2.0])
    first = np.zeros(2)
    second = np.zeros(2)
    for step, raw in enumerate(gradients, 1):
        gradient = raw.copy()
        if weight_decay and not decoupled:
            gradient += weight_decay * value
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        update = (
            -lr
            * (first / (1 - 0.9**step))
            / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        )
        if decoupled:
            update -= lr * weight_decay * value
        value += update
    return value


@pytest.mark.parametrize("optimizer_class", OPTIMIZERS)
def test_every_optimizer_updates_gpu_parameter_and_state(optimizer_class) -> None:
    parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    optimizer = optimizer_class([parameter])
    before = parameter.numpy()
    parameter.square().sum().backward()
    optimizer.step()
    assert not np.array_equal(parameter.numpy(), before)
    assert parameter._version == 1
    for value in optimizer.state.get(id(parameter), {}).values():
        if hasattr(value, "device"):
            assert int(value.device.id) == 0


def test_optimizer_parameter_groups_validation_and_zero_grad() -> None:
    first = mt.nn.Parameter([1.0])
    second = mt.nn.Parameter([2.0])
    optimizer = mt.optim.Adam(
        [{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.01}]
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.01)
    (first.square() + second.square()).sum().backward()
    optimizer.zero_grad(set_to_none=False)
    np.testing.assert_allclose(first.grad.numpy(), [0.0])

    with pytest.raises(ValueError, match="more than one"):
        mt.optim.SGD([first, first])
    with pytest.raises(ValueError, match="no parameters"):
        mt.optim.SGD([])
    with pytest.raises(ValueError):
        mt.optim.Adam([first], lr=0)
    with pytest.raises(ValueError, match="unknown optimizer option"):
        mt.optim.Adam([{"params": [first], "invented": 1}])


def test_none_gradient_skips_but_zero_gradient_advances_optimizer_state() -> None:
    parameter = mt.nn.Parameter([1.0])
    optimizer = mt.optim.Adam([parameter])
    optimizer.step()
    assert id(parameter) not in optimizer.state
    (parameter * 0.0).sum().backward()
    optimizer.step()
    assert optimizer.state[id(parameter)]["step"] == 1


def test_five_step_sgd_and_adam_family_match_numpy_references() -> None:
    gradients = [
        np.array([0.2, -0.1]),
        np.array([-0.3, 0.4]),
        np.array([0.1, 0.2]),
        np.array([-0.2, -0.3]),
        np.array([0.05, -0.05]),
    ]

    parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    optimizer = mt.optim.SGD([parameter], lr=0.1, momentum=0.5)
    expected = np.array([1.0, -2.0])
    momentum = None
    for gradient in gradients:
        momentum = gradient.copy() if momentum is None else 0.5 * momentum + gradient
        expected -= 0.1 * momentum
    _apply_gradients(parameter, optimizer, gradients)
    np.testing.assert_allclose(parameter.numpy(), expected, rtol=1e-12, atol=1e-12)

    for optimizer_class, options, reference in [
        (
            mt.optim.Adam,
            {"lr": 0.01, "weight_decay": 0.1},
            _adam_reference(gradients, lr=0.01, weight_decay=0.1),
        ),
        (
            mt.optim.AdamW,
            {"lr": 0.01, "weight_decay": 0.1},
            _adam_reference(gradients, lr=0.01, weight_decay=0.1, decoupled=True),
        ),
    ]:
        parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
        _apply_gradients(parameter, optimizer_class([parameter], **options), gradients)
        np.testing.assert_allclose(parameter.numpy(), reference, rtol=1e-12, atol=1e-12)


def test_five_step_adagrad_rmsprop_and_lion_match_numpy_references() -> None:
    gradients = [
        np.array([0.2, -0.1]),
        np.array([-0.3, 0.4]),
        np.array([0.1, 0.2]),
        np.array([-0.2, -0.3]),
        np.array([0.05, -0.05]),
    ]

    parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    accumulator = np.zeros(2)
    expected = np.array([1.0, -2.0])
    for gradient in gradients:
        accumulator += gradient**2
        expected -= 0.1 * gradient / (np.sqrt(accumulator) + 1e-10)
    _apply_gradients(parameter, mt.optim.Adagrad([parameter], lr=0.1), gradients)
    np.testing.assert_allclose(parameter.numpy(), expected, rtol=1e-12, atol=1e-12)

    parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    average = np.zeros(2)
    expected = np.array([1.0, -2.0])
    for gradient in gradients:
        average = 0.9 * average + 0.1 * gradient**2
        expected -= 0.01 * gradient / (np.sqrt(average) + 1e-8)
    _apply_gradients(
        parameter, mt.optim.RMSprop([parameter], lr=0.01, alpha=0.9), gradients
    )
    np.testing.assert_allclose(parameter.numpy(), expected, rtol=1e-12, atol=1e-12)

    parameter = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    momentum = np.zeros(2)
    expected = np.array([1.0, -2.0])
    for gradient in gradients:
        direction = np.sign(0.9 * momentum + 0.1 * gradient)
        expected -= 0.01 * direction + 0.01 * 0.1 * expected
        momentum = 0.99 * momentum + 0.01 * gradient
    _apply_gradients(
        parameter,
        mt.optim.Lion([parameter], lr=0.01, weight_decay=0.1),
        gradients,
    )
    np.testing.assert_allclose(parameter.numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [
        (mt.optim.AdamW, {"lr": 0.05}),
        (mt.optim.Lion, {"lr": 0.01}),
        (mt.optim.Adafactor, {"lr": 0.05}),
    ],
)
def test_modern_optimizers_reduce_small_regression(optimizer_class, kwargs) -> None:
    mt.manual_seed(123)
    inputs = mt.linspace(-1, 1, 32).reshape(16, 2)
    targets = inputs @ mt.tensor([[2.0], [-3.0]]) + 0.25
    model = mt.nn.Linear(2, 1)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    initial = mt.nn.functional.mse_loss(model(inputs), targets).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = mt.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    assert mt.nn.functional.mse_loss(model(inputs), targets).item() < initial
