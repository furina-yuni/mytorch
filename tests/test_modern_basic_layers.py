import numpy as np
import pytest

import mytorch as mt


def test_bilinear_and_lazy_linear_backward():
    left = mt.randn(4, 3, requires_grad=True)
    right = mt.randn(4, 2, requires_grad=True)
    layer = mt.nn.Bilinear(3, 2, 5)
    output = layer(left, right)
    output.sum().backward()
    assert output.shape == (4, 5)
    assert layer.weight.grad.shape == layer.weight.shape
    assert left.grad.shape == left.shape
    assert right.grad.shape == right.shape

    lazy = mt.nn.LazyLinear(7)
    with pytest.raises(RuntimeError, match="before parameters"):
        list(lazy.parameters())
    assert lazy(mt.randn(2, 9)).shape == (2, 7)
    assert lazy.in_features == 9
    assert len(list(lazy.parameters())) == 2


def test_embedding_duplicate_gradient_and_embedding_bag():
    embedding = mt.nn.Embedding(6, 3, padding_idx=0)
    indices = mt.tensor([0, 1, 1, 2, 4], dtype=mt.int64)
    embedding(indices).sum().backward()
    np.testing.assert_array_equal(embedding.weight.grad.numpy()[1], [2, 2, 2])
    np.testing.assert_array_equal(embedding.weight.grad.numpy()[0], [0, 0, 0])
    np.testing.assert_array_equal(embedding.weight.numpy()[0], [0, 0, 0])

    bag = mt.nn.EmbeddingBag(6, 3, mode="sum")
    bag_indices = mt.tensor([1, 1, 2, 4], dtype=mt.int64)
    output = bag(bag_indices, mt.tensor([0, 2], dtype=mt.int64))
    expected = np.stack(
        [bag.weight.numpy()[[1, 1]].sum(0), bag.weight.numpy()[[2, 4]].sum(0)]
    )
    np.testing.assert_allclose(output.numpy(), expected)


def test_dropout_reproducibility_and_eval_identity():
    value = mt.ones(128)
    dropout = mt.nn.Dropout(0.5)
    mt.manual_seed(17)
    first = dropout(value)
    mt.manual_seed(17)
    second = dropout(value)
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    dropout.eval()
    assert dropout(value) is value


def test_normalization_layers_are_finite_and_differentiable():
    layers_and_inputs = [
        (mt.nn.LayerNorm(4), mt.randn(3, 4, requires_grad=True)),
        (mt.nn.RMSNorm(4), mt.randn(3, 4, requires_grad=True)),
        (mt.nn.GroupNorm(2, 4), mt.randn(2, 4, 3, 3, requires_grad=True)),
        (
            mt.nn.InstanceNorm2d(4, affine=True),
            mt.randn(2, 4, 3, 3, requires_grad=True),
        ),
        (mt.nn.BatchNorm2d(4), mt.randn(2, 4, 3, 3, requires_grad=True)),
    ]
    for layer, value in layers_and_inputs:
        output = layer(value)
        assert np.isfinite(output.numpy()).all()
        output.sum().backward()
        assert value.grad is not None


def test_batch_norm_train_eval_and_running_state():
    layer = mt.nn.BatchNorm1d(2, momentum=1.0)
    value = mt.tensor([[1.0, 2.0], [3.0, 6.0]])
    training = layer(value)
    np.testing.assert_allclose(training.numpy().mean(0), [0, 0], atol=1e-6)
    np.testing.assert_allclose(layer.running_mean.numpy(), [2, 4])
    layer.eval()
    evaluated = layer(value)
    assert not np.allclose(evaluated.numpy(), training.numpy())
