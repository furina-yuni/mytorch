"""Train and run inference with a small GPU-only MyTorch classifier."""

from __future__ import annotations

import numpy as np

import mytorch as mt
from mytorch.nn import functional as F


class SimpleClassifier(mt.nn.Module):
    """A two-layer MLP that classifies points using their x/y coordinates."""

    def __init__(self, hidden_features: int = 16) -> None:
        super().__init__()
        self.hidden = mt.nn.Linear(2, hidden_features)
        self.activation = mt.nn.ReLU()
        self.output = mt.nn.Linear(hidden_features, 3)

    def forward(self, inputs: mt.Tensor) -> mt.Tensor:
        features = self.activation(self.hidden(inputs))
        return self.output(features)


def make_training_data() -> tuple[mt.Tensor, mt.Tensor]:
    """Return three small 2D clusters already transferred to the GPU."""
    points = np.array(
        [
            [-2.4, -1.2],
            [-2.0, -0.8],
            [-1.8, -1.4],
            [-2.2, -0.5],
            [-1.5, -1.0],
            [-2.6, -0.7],
            [2.4, -1.2],
            [2.0, -0.8],
            [1.8, -1.4],
            [2.2, -0.5],
            [1.5, -1.0],
            [2.6, -0.7],
            [-0.4, 2.2],
            [0.0, 1.7],
            [0.5, 2.4],
            [-0.6, 1.5],
            [0.7, 1.8],
            [0.1, 2.7],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.arange(3, dtype=np.int64), 6)
    return mt.tensor(points), mt.tensor(labels)


def train(
    model: SimpleClassifier,
    inputs: mt.Tensor,
    labels: mt.Tensor,
    *,
    epochs: int = 250,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> list[float]:
    """Train with full-batch SGD and return the loss from every epoch."""
    model.train()
    optimizer = mt.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        history.append(loss_value)
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d}/{epochs}: loss={loss_value:.6f}")

    return history


def predict_probabilities(model: SimpleClassifier, inputs: mt.Tensor) -> mt.Tensor:
    """Return class probabilities without constructing an autograd graph."""
    model.eval()
    with mt.no_grad():
        return F.softmax(model(inputs), dim=1)


def predict(model: SimpleClassifier, inputs: mt.Tensor) -> mt.Tensor:
    """Return the most likely class index for each input on the GPU."""
    return predict_probabilities(model, inputs).argmax(dim=1)


def accuracy(model: SimpleClassifier, inputs: mt.Tensor, labels: mt.Tensor) -> float:
    """Compute accuracy after explicitly copying only final results to the CPU."""
    predicted = predict(model, inputs).numpy()
    expected = labels.numpy()
    return float((predicted == expected).mean())


def main() -> None:
    mt.manual_seed(2026)
    inputs, labels = make_training_data()
    model = SimpleClassifier()

    history = train(model, inputs, labels)
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")
    print(f"training accuracy: {accuracy(model, inputs, labels):.1%}")

    new_points = mt.tensor([[-2.1, -1.0], [2.1, -1.0], [0.0, 2.1]])
    probabilities = predict_probabilities(model, new_points)
    predictions = probabilities.argmax(dim=1)
    class_names = ("left", "right", "top")

    print("\nnew point probabilities:")
    print(probabilities.numpy())
    print("predictions:")
    for point, class_index in zip(new_points.numpy(), predictions.numpy(), strict=True):
        print(f"  {point.tolist()} -> {class_names[int(class_index)]}")


if __name__ == "__main__":
    main()
