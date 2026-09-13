import mytorch as mt


def _train(model, input, target, steps=25, lr=0.03):
    optimizer = mt.optim.Adam(model.parameters(), lr=lr)
    initial = None
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = model(input)
        loss = mt.nn.functional.mse_loss(prediction, target)
        if initial is None:
            initial = loss.item()
        loss.backward()
        optimizer.step()
    return initial, loss.item()


def test_small_cnn_training_reduces_loss():
    mt.manual_seed(21)
    inputs = mt.randn(16, 1, 4, 4)
    targets = inputs.mean((1, 2, 3)).unsqueeze(1)
    model = mt.nn.Sequential(
        mt.nn.Conv2d(1, 2, 3, padding=1),
        mt.nn.ReLU(),
        mt.nn.Flatten(),
        mt.nn.Linear(32, 1),
    )
    initial, final = _train(model, inputs, targets)
    assert final < initial * 0.25


class SequenceRegressor(mt.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = mt.nn.GRU(3, 6, batch_first=True)
        self.output = mt.nn.Linear(6, 1)

    def forward(self, input):
        sequence, _ = self.gru(input)
        return self.output(sequence[:, -1])


def test_gru_training_reduces_loss():
    mt.manual_seed(22)
    inputs = mt.randn(12, 4, 3)
    targets = inputs.mean((1, 2)).unsqueeze(1)
    initial, final = _train(SequenceRegressor(), inputs, targets, steps=30)
    assert final < initial * 0.35


class TransformerRegressor(mt.nn.Module):
    def __init__(self):
        super().__init__()
        layer = mt.nn.TransformerEncoderLayer(
            8, 2, 16, dropout=0, batch_first=True, norm_first=True
        )
        self.encoder = mt.nn.TransformerEncoder(layer, 1)
        self.output = mt.nn.Linear(8, 1)

    def forward(self, input):
        return self.output(self.encoder(input).mean(1))


def test_transformer_training_reduces_loss():
    mt.manual_seed(23)
    inputs = mt.randn(10, 4, 8)
    targets = inputs.mean((1, 2)).unsqueeze(1)
    initial, final = _train(TransformerRegressor(), inputs, targets, steps=25, lr=0.02)
    assert final < initial * 0.3
