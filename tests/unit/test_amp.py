import numpy as np
import pytest

import mytorch as mt


def test_autocast_dtype_policy_and_nested_restore() -> None:
    left = mt.randn(4, 8)
    right = mt.randn(8, 3)
    assert not mt.amp.is_autocast_enabled()
    with mt.amp.autocast():
        assert mt.amp.is_autocast_enabled()
        product = left @ right
        stable = product.softmax(dim=-1)
        with mt.amp.autocast(enabled=False):
            full = left @ right
        assert mt.amp.is_autocast_enabled()
    assert not mt.amp.is_autocast_enabled()
    assert product.dtype == mt.float16
    assert stable.dtype == mt.float32
    assert full.dtype == mt.float32

    double = mt.randn(2, 2, dtype=mt.float64)
    with mt.amp.autocast():
        assert (double @ double).dtype == mt.float64


def test_amp_keeps_master_parameters_and_gradients_float32() -> None:
    mt.manual_seed(91)
    model = mt.nn.Linear(8, 2)
    optimizer = mt.optim.AdamW(model.parameters(), lr=0.01)
    scaler = mt.amp.GradScaler(init_scale=128, growth_interval=1)
    inputs = mt.randn(16, 8)
    targets = mt.randn(16, 2)

    with mt.amp.autocast():
        prediction = model(inputs)
        loss = mt.nn.functional.mse_loss(prediction, targets)
    assert prediction.dtype == mt.float16
    assert loss.dtype == mt.float32
    scaler.scale(loss).backward()
    assert all(parameter.grad.dtype == mt.float32 for parameter in model.parameters())
    assert scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == 256
    assert all(parameter.dtype == mt.float32 for parameter in model.parameters())


def test_grad_scaler_overflow_skips_update_and_restores_state(tmp_path) -> None:
    parameter = mt.nn.Parameter([2.0])
    optimizer = mt.optim.SGD([parameter], lr=1.0)
    scaler = mt.amp.GradScaler(init_scale=16, growth_interval=2)
    (parameter * float("inf") * scaler.get_scale()).sum().backward()
    before = parameter.numpy().copy()
    assert not scaler.step(optimizer)
    np.testing.assert_array_equal(parameter.numpy(), before)
    scaler.update()
    assert scaler.get_scale() == 8

    path = tmp_path / "scaler.mtz"
    mt.save_checkpoint({"scaler": scaler.state_dict()}, path)
    restored = mt.amp.GradScaler()
    restored.load_state_dict(mt.load_checkpoint(path)["scaler"])
    assert restored.get_scale() == 8

    with pytest.raises(RuntimeError, match="preceding step"):
        restored.update()


def test_grad_scaler_rejects_duplicate_step() -> None:
    parameter = mt.nn.Parameter([1.0])
    optimizer = mt.optim.SGD([parameter])
    scaler = mt.amp.GradScaler()
    scaler.scale(parameter.sum()).backward()
    assert scaler.step(optimizer)
    with pytest.raises(RuntimeError, match="already"):
        scaler.step(optimizer)


def test_grad_scaler_unscale_and_disabled_mode() -> None:
    parameter = mt.nn.Parameter([2.0])
    optimizer = mt.optim.SGD([parameter], lr=0.1)
    scaler = mt.amp.GradScaler(init_scale=4)
    scaler.scale(parameter.square().sum()).backward()
    scaler.unscale_(optimizer)
    np.testing.assert_allclose(parameter.grad.numpy(), [4.0])
    with pytest.raises(RuntimeError, match="only once"):
        scaler.unscale_(optimizer)
    assert scaler.step(optimizer)
    scaler.update(new_scale=32)
    assert scaler.get_scale() == 32

    disabled = mt.amp.GradScaler(enabled=False)
    optimizer.zero_grad()
    disabled.scale(parameter.sum()).backward()
    assert disabled.step(optimizer)
    disabled.update()
    with pytest.raises(TypeError, match="float16"):
        mt.amp.autocast(dtype=mt.float32)
