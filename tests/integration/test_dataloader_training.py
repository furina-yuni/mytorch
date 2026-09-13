import mytorch as mt


def test_dataloader_mini_batch_training_reduces_loss() -> None:
    mt.manual_seed(31)
    inputs = mt.linspace(-2, 2, 48).unsqueeze(1)
    targets = inputs * 2.5 - 0.75
    dataset = mt.data.TensorDataset(inputs, targets)
    loader = mt.data.DataLoader(dataset, batch_size=8, shuffle=True, seed=31)
    model = mt.nn.Linear(1, 1)
    optimizer = mt.optim.Adam(model.parameters(), lr=0.05)

    with mt.no_grad():
        initial = mt.nn.functional.mse_loss(model(inputs), targets).item()
    for _ in range(60):
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss = mt.nn.functional.mse_loss(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
    with mt.no_grad():
        final = mt.nn.functional.mse_loss(model(inputs), targets).item()

    assert final < initial * 1e-3
    assert final < 1e-4
