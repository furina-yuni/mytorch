import math

import numpy as np
import pytest

import mytorch as mt


def _optimizer(lr=1.0):
    return mt.optim.SGD([mt.nn.Parameter([1.0])], lr=lr, momentum=0.9)


def test_closed_form_epoch_schedulers() -> None:
    optimizer = _optimizer()
    step = mt.optim.StepLR(optimizer, step_size=2, gamma=0.1)
    values = []
    for _ in range(5):
        step.step()
        values.append(optimizer.param_groups[0]["lr"])
    np.testing.assert_allclose(values, [1.0, 1.0, 0.1, 0.1, 0.01])

    optimizer = _optimizer()
    multi = mt.optim.MultiStepLR(optimizer, [1, 3], gamma=0.5)
    values = []
    for _ in range(4):
        multi.step()
        values.append(optimizer.param_groups[0]["lr"])
    np.testing.assert_allclose(values, [1.0, 0.5, 0.5, 0.25])

    optimizer = _optimizer()
    exponential = mt.optim.ExponentialLR(optimizer, gamma=0.5)
    exponential.step(3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.125)


def test_linear_cosine_warm_restart_and_sequential() -> None:
    optimizer = _optimizer()
    linear = mt.optim.LinearLR(optimizer, start_factor=0.2, total_iters=2)
    np.testing.assert_allclose(
        [linear.get_lr()[0], (linear.step(), linear.get_last_lr()[0])[1]],
        [0.2, 0.2],
    )
    linear.step()
    assert linear.get_last_lr()[0] == pytest.approx(0.6)

    optimizer = _optimizer()
    cosine = mt.optim.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.1)
    cosine.step(4)
    assert cosine.get_last_lr()[0] == pytest.approx(0.1)

    optimizer = _optimizer()
    restart = mt.optim.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    restart.step(2)
    assert restart.get_last_lr()[0] == pytest.approx(1.0)

    optimizer = _optimizer()
    warmup = mt.optim.LinearLR(optimizer, start_factor=0.5, total_iters=1)
    decay = mt.optim.ExponentialLR(optimizer, gamma=0.1)
    sequential = mt.optim.SequentialLR(optimizer, [warmup, decay], [2])
    sequential.step(0)
    assert sequential.get_last_lr()[0] == pytest.approx(0.5)
    sequential.step(2)
    assert sequential.get_last_lr()[0] == pytest.approx(1.0)
    sequential.step(3)
    assert sequential.get_last_lr()[0] == pytest.approx(0.1)


def test_plateau_one_cycle_and_scheduler_checkpoint(tmp_path) -> None:
    optimizer = _optimizer()
    plateau = mt.optim.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1, cooldown=1, min_lr=0.2
    )
    for metric in [1.0, 1.1, 1.2]:
        plateau.step(metric)
    assert plateau.get_last_lr() == [0.5]
    for metric in [1.3, 1.4, 1.5, 1.6]:
        plateau.step(metric)
    assert plateau.get_last_lr() == [0.25]

    optimizer = _optimizer(lr=0.9)
    cycle = mt.optim.OneCycleLR(optimizer, max_lr=1.0, total_steps=4)
    seen = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        cycle.step()
        seen.append(optimizer.param_groups[0]["lr"])
    assert max(seen) <= 1.0
    assert seen[-1] == pytest.approx(1.0 / 25 / 1e4)
    assert all(math.isfinite(value) and value >= 0 for value in seen)

    state_path = tmp_path / "scheduler.mtz"
    mt.save_checkpoint({"scheduler": cycle.state_dict()}, state_path)
    restored_optimizer = _optimizer(lr=0.123)
    restored = mt.optim.OneCycleLR(restored_optimizer, max_lr=1.0, total_steps=4)
    restored.load_state_dict(mt.load_checkpoint(state_path)["scheduler"])
    assert restored.last_epoch == cycle.last_epoch
    assert restored.get_last_lr() == cycle.get_last_lr()
    assert restored_optimizer.param_groups[0]["lr"] == seen[-1]


def test_scheduler_validation() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="positive"):
        mt.optim.StepLR(optimizer, 0)
    with pytest.raises(ValueError, match="sorted"):
        mt.optim.MultiStepLR(optimizer, [2, 1])
    with pytest.raises(ValueError, match="less than 1"):
        mt.optim.ReduceLROnPlateau(optimizer, factor=1)


def test_scheduler_state_validation_and_additional_modes() -> None:
    optimizer = _optimizer()
    scheduler = mt.optim.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=0.1
    )
    scheduler.step(4)
    assert 0.1 <= scheduler.get_last_lr()[0] <= 1.0
    state = scheduler.state_dict()

    restored_optimizer = _optimizer()
    restored = mt.optim.CosineAnnealingWarmRestarts(
        restored_optimizer, T_0=1, T_mult=2, eta_min=0.1
    )
    restored.load_state_dict(state)
    assert restored.get_last_lr() == scheduler.get_last_lr()
    with pytest.raises(ValueError, match="type"):
        mt.optim.StepLR(restored_optimizer, 1).load_state_dict(state)

    adam = mt.optim.Adam([mt.nn.Parameter([1.0])])
    cycle = mt.optim.OneCycleLR(
        adam,
        max_lr=0.1,
        total_steps=6,
        anneal_strategy="linear",
        three_phase=True,
    )
    for _ in range(6):
        cycle.step()
    assert adam.param_groups[0]["betas"][0] == pytest.approx(0.95)
    with pytest.raises(ValueError, match="more than"):
        cycle.step()

    maximum = mt.optim.ReduceLROnPlateau(
        restored_optimizer,
        mode="max",
        threshold_mode="abs",
        patience=0,
        cooldown=1,
    )
    maximum.step(0.5)
    maximum.step(0.4)
    maximum_state = maximum.state_dict()
    clone = mt.optim.ReduceLROnPlateau(
        restored_optimizer,
        mode="max",
        threshold_mode="abs",
        patience=0,
        cooldown=1,
    )
    clone.load_state_dict(maximum_state)
    assert clone.best == 0.5
