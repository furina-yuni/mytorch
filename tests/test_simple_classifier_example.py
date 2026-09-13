from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

import mytorch as mt

_example = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "simple_classifier.py")
)
SimpleClassifier = _example["SimpleClassifier"]
accuracy = _example["accuracy"]
make_training_data = _example["make_training_data"]
predict = _example["predict"]
predict_probabilities = _example["predict_probabilities"]
train = _example["train"]

pytestmark = pytest.mark.gpu


def test_simple_classifier_can_train_and_predict() -> None:
    mt.manual_seed(2026)
    inputs, labels = make_training_data()
    model = SimpleClassifier(hidden_features=12)

    history = train(
        model,
        inputs,
        labels,
        epochs=150,
        learning_rate=0.1,
        verbose=False,
    )

    assert history[-1] < history[0] * 0.05
    assert history[-1] < 0.02
    assert accuracy(model, inputs, labels) == pytest.approx(1.0)

    new_points = mt.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    probabilities = predict_probabilities(model, new_points)
    predictions = predict(model, new_points)

    assert probabilities.device == "cuda:0"
    assert not probabilities.requires_grad
    np.testing.assert_allclose(
        probabilities.sum(dim=1).numpy(), np.ones(3), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(predictions.numpy(), [0, 1, 2])
