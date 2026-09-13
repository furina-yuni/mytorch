from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def _components(seed: int):
    mt.manual_seed(seed)
    inputs = mt.linspace(-1, 1, 32).reshape(16, 2)
    targets = inputs @ mt.tensor([[1.5], [-2.0]]) + 0.2
    dataset = mt.data.TensorDataset(inputs, targets)
    loader = mt.data.DataLoader(dataset, batch_size=4, shuffle=True, seed=77)
    model = mt.nn.Sequential(
        mt.nn.Linear(2, 8),
        mt.nn.ReLU(),
        mt.nn.Dropout(0.2),
        mt.nn.Linear(8, 1),
    )
    optimizer = mt.optim.Adam(model.parameters(), lr=0.03)
    return model, optimizer, loader


def _train_epochs(model, optimizer, loader, epochs: int) -> list[float]:
    losses = []
    for _ in range(epochs):
        total = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = mt.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total)
    return losses


def test_training_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    continuous_model, continuous_optimizer, continuous_loader = _components(123)
    continuous_losses = _train_epochs(
        continuous_model, continuous_optimizer, continuous_loader, 6
    )

    interrupted_model, interrupted_optimizer, interrupted_loader = _components(123)
    interrupted_losses = _train_epochs(
        interrupted_model, interrupted_optimizer, interrupted_loader, 3
    )
    checkpoint_path = tmp_path / "resume.mtz"
    mt.save_checkpoint(
        {
            "epoch": 3,
            "model": interrupted_model.state_dict(),
            "optimizer": interrupted_optimizer.state_dict(),
            "rng_state": mt.get_rng_state(),
            "dataloader": interrupted_loader.state_dict(),
        },
        checkpoint_path,
    )

    resumed_model, resumed_optimizer, resumed_loader = _components(999)
    checkpoint = mt.load_checkpoint(checkpoint_path)
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_loader.load_state_dict(checkpoint["dataloader"])
    mt.set_rng_state(checkpoint["rng_state"])
    resumed_losses = _train_epochs(resumed_model, resumed_optimizer, resumed_loader, 3)

    np.testing.assert_array_equal(
        interrupted_losses + resumed_losses, continuous_losses
    )
    for continuous, resumed in zip(
        continuous_model.parameters(), resumed_model.parameters(), strict=True
    ):
        np.testing.assert_array_equal(resumed.numpy(), continuous.numpy())
    continuous_state = continuous_optimizer.state_dict()
    resumed_state = resumed_optimizer.state_dict()
    for index in continuous_state["state"]:
        assert resumed_state["state"][index]["step"] == 24
        for name in ("exp_avg", "exp_avg_sq"):
            np.testing.assert_array_equal(
                resumed_state["state"][index][name].numpy(),
                continuous_state["state"][index][name].numpy(),
            )
