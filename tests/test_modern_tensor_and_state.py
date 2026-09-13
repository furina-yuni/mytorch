from pathlib import Path

import numpy as np
import pytest

import mytorch as mt


def test_expand_repeat_pad_and_contiguous_backward():
    source = mt.tensor([[1.0], [2.0]], requires_grad=True)
    output = source.expand(2, 3).repeat(2, 1)
    padded = mt.pad(output, (1, 2, 1, 0)).contiguous()
    padded.sum().backward()
    np.testing.assert_array_equal(source.grad.numpy(), [[6.0], [6.0]])
    assert padded.shape == (5, 6)


def test_gather_scatter_add_and_topk_gradients():
    source = mt.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], requires_grad=True)
    index = mt.tensor([[1, 0], [2, 0]], dtype=mt.int64)
    selected = source.gather(1, index)
    values, positions = source.topk(2, 1)
    (selected.sum() + values.sum()).backward()
    np.testing.assert_array_equal(positions.numpy(), [[1, 2], [2, 0]])
    np.testing.assert_array_equal(source.grad.numpy(), [[1, 2, 1], [2, 0, 2]])

    base = mt.zeros(2, 3)
    scattered = base.scatter_add(1, index, mt.ones(2, 2))
    np.testing.assert_array_equal(scattered.numpy(), [[1, 1, 0], [1, 0, 1]])


def test_round_rsqrt_and_reshape_inference_backward():
    source = mt.tensor([1.0, 4.0, 9.0], requires_grad=True)
    result = source.rsqrt().reshape(-1, 1)
    result.sum().backward()
    expected = -0.5 * np.asarray([1.0, 4.0, 9.0]) ** -1.5
    np.testing.assert_allclose(source.grad.numpy(), expected, rtol=1e-6)

    round_source = mt.tensor([1.2, 1.8], requires_grad=True)
    rounded = round_source.round()
    rounded.sum().backward()
    np.testing.assert_array_equal(round_source.grad.numpy(), [0, 0])


def test_module_buffers_state_and_safe_npz_roundtrip(tmp_path: Path):
    module = mt.nn.BatchNorm1d(3)
    module(mt.randn(4, 3))
    state = module.state_dict()
    assert set(state) == {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    path = tmp_path / "state.npz"
    mt.save(state, path)
    loaded = mt.load(path)
    clone = mt.nn.BatchNorm1d(3)
    clone.load_state_dict(loaded)
    module.eval()
    clone.eval()
    sample = mt.randn(2, 3)
    np.testing.assert_allclose(module(sample).numpy(), clone(sample).numpy())

    clone.to(dtype=mt.float64)
    assert clone.weight.dtype == mt.float64
    assert clone.num_batches_tracked.dtype == mt.int64


def test_module_list_parameter_list_and_nonpersistent_buffer():
    modules = mt.nn.ModuleList([mt.nn.Linear(2, 3), mt.nn.Linear(3, 1)])
    parameters = mt.nn.ParameterList([mt.nn.Parameter(mt.ones(2))])
    assert len(modules) == 2
    assert len(parameters) == 1
    modules.register_buffer("cache", mt.ones(1), persistent=False)
    assert "cache" not in modules.state_dict()


def test_initializers_increment_version_and_state_errors_are_explicit():
    parameter = mt.nn.Parameter(mt.zeros(4, 3))
    version = parameter._version
    mt.nn.init.xavier_uniform_(parameter)
    assert parameter._version == version + 1
    assert np.isfinite(parameter.numpy()).all()

    module = mt.nn.Linear(3, 2)
    incomplete = module.state_dict()
    incomplete.pop("bias")
    with pytest.raises(RuntimeError, match="missing keys"):
        module.load_state_dict(incomplete)
