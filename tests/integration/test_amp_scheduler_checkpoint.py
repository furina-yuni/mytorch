from pathlib import Path

import numpy as np

import mytorch as mt


def _components(seed: int):
    mt.manual_seed(seed)
    inputs = mt.linspace(-1, 1, 64).reshape(32, 2)
    targets = inputs @ mt.tensor([[1.25], [-0.75]]) + 0.1
    loader = mt.data.DataLoader(
        mt.data.TensorDataset(inputs, targets),
        batch_size=8,
        shuffle=True,
        seed=19,
    )
    model = mt.nn.Sequential(
        mt.nn.Linear(2, 8),
        mt.nn.ReLU(),
        mt.nn.Dropout(0.1),
        mt.nn.Linear(8, 1),
    )
    optimizer = mt.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = mt.optim.CosineAnnealingLR(optimizer, T_max=16, eta_min=0.001)
    scaler = mt.amp.GradScaler(init_scale=128, growth_interval=5)
    return model, optimizer, scheduler, scaler, loader


def _train(model, optimizer, scheduler, scaler, loader, epochs):
    history = []
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            with mt.amp.autocast():
                loss = mt.nn.functional.mse_loss(model(inputs), targets)
            scaler.scale(loss).backward()
            updated = scaler.step(optimizer)
            scaler.update()
            if updated:
                scheduler.step()
            history.append(
                (loss.item(), scheduler.get_last_lr()[0], scaler.get_scale())
            )
    return history


def test_amp_scheduler_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    continuous = _components(44)
    continuous_history = _train(*continuous, 4)

    interrupted = _components(44)
    first_history = _train(*interrupted, 2)
    model, optimizer, scheduler, scaler, loader = interrupted
    path = tmp_path / "full-training.mtz"
    mt.save_checkpoint(
        {
            "epoch": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "dataloader": loader.state_dict(),
            "rng_state": mt.get_rng_state(),
        },
        path,
    )

    resumed = _components(999)
    model, optimizer, scheduler, scaler, loader = resumed
    state = mt.load_checkpoint(path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    loader.load_state_dict(state["dataloader"])
    mt.set_rng_state(state["rng_state"])
    resumed_history = _train(*resumed, 2)

    np.testing.assert_array_equal(first_history + resumed_history, continuous_history)
    for expected, actual in zip(
        continuous[0].parameters(), resumed[0].parameters(), strict=True
    ):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
