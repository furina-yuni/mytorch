from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_rng_state_replays_gpu_random_operations() -> None:
    mt.manual_seed(321)
    state = mt.get_rng_state()
    expected_uniform = mt.rand(3, 4)
    expected_normal = mt.randn(2, 5)
    mt.set_rng_state(state)
    np.testing.assert_array_equal(mt.rand(3, 4).numpy(), expected_uniform.numpy())
    np.testing.assert_array_equal(mt.randn(2, 5).numpy(), expected_normal.numpy())


def test_optimizer_state_uses_stable_indices_and_restores_gpu_arrays() -> None:
    first = mt.nn.Parameter([1.0, -2.0], dtype=mt.float64)
    second = mt.nn.Parameter([3.0], dtype=mt.float64)
    optimizer = mt.optim.Adam(
        [
            {"params": [first], "lr": 0.02},
            {"params": [second], "lr": 0.01},
        ]
    )
    (first.square().sum() + second.square().sum()).backward()
    optimizer.step()
    state = optimizer.state_dict()

    assert list(state["state"]) == [0, 1]
    assert state["param_groups"][0]["params"] == [0]
    assert isinstance(state["state"][0]["exp_avg"], mt.Tensor)

    restored_first = mt.nn.Parameter([0.0, 0.0], dtype=mt.float64)
    restored_second = mt.nn.Parameter([0.0], dtype=mt.float64)
    restored = mt.optim.Adam([restored_first, restored_second])
    with pytest.raises(ValueError, match="group count"):
        restored.load_state_dict(state)

    restored = mt.optim.Adam(
        [
            {"params": [restored_first], "lr": 1.0},
            {"params": [restored_second], "lr": 1.0},
        ]
    )
    restored.load_state_dict(state)
    assert restored.param_groups[0]["lr"] == pytest.approx(0.02)
    assert restored.state[id(restored_first)]["step"] == 1
    np.testing.assert_array_equal(
        restored.state[id(restored_first)]["exp_avg"].get(),
        state["state"][0]["exp_avg"].numpy(),
    )


def test_checkpoint_roundtrip_preserves_nested_safe_values(tmp_path: Path) -> None:
    path = tmp_path / "training.mtz"
    checkpoint = {
        "epoch": 7,
        "model": {"weight": mt.tensor([[1.0, 2.0]])},
        "metrics": [0.75, 0.5],
        "options": (True, None, "run-a"),
        "indexed": {3: "value"},
    }
    mt.save_checkpoint(checkpoint, path)
    loaded = mt.load_checkpoint(path)

    assert path.exists()
    assert loaded["epoch"] == 7
    assert loaded["metrics"] == [0.75, 0.5]
    assert loaded["options"] == (True, None, "run-a")
    assert loaded["indexed"] == {3: "value"}
    np.testing.assert_array_equal(loaded["model"]["weight"].numpy(), [[1.0, 2.0]])

    with pytest.raises(TypeError, match="cannot store"):
        mt.save_checkpoint({"unsafe": object()}, tmp_path / "unsafe.mtz")


def test_dataloader_state_restores_next_epoch_order() -> None:
    dataset = mt.data.TensorDataset(mt.arange(12, dtype=mt.int64))
    first = mt.data.DataLoader(dataset, batch_size=3, shuffle=True, seed=17)
    list(first)
    state = first.state_dict()
    expected = [batch[0].numpy() for batch in first]

    restored = mt.data.DataLoader(dataset, batch_size=3, shuffle=True, seed=999)
    restored.load_state_dict(state)
    actual = [batch[0].numpy() for batch in restored]
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(actual_batch, expected_batch)

    incompatible = mt.data.DataLoader(dataset, batch_size=4, shuffle=True, seed=17)
    with pytest.raises(ValueError, match="batch_size"):
        incompatible.load_state_dict(state)
