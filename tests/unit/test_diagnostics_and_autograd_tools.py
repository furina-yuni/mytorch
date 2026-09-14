from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_inference_mode_is_nested_and_restores_grad_state() -> None:
    source = mt.tensor([2.0], requires_grad=True)
    assert not mt.is_inference_mode_enabled()
    with mt.inference_mode():
        assert mt.is_inference_mode_enabled()
        assert not mt.is_grad_enabled()
        result = source.square()
        assert result.is_inference()
        assert not result.requires_grad
        with mt.inference_mode(False):
            assert mt.is_grad_enabled()
            assert not mt.is_inference_mode_enabled()
    assert mt.is_grad_enabled()
    assert not mt.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="inference Tensor"):
        result.requires_grad_()


def test_autograd_grad_returns_vjp_without_accumulating() -> None:
    x = mt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = mt.tensor([4.0, 5.0, 6.0], requires_grad=True)
    output = (x * y).sum()

    grad_x, grad_y = mt.grad(output, (x, y))

    np.testing.assert_allclose(grad_x.numpy(), y.numpy())
    np.testing.assert_allclose(grad_y.numpy(), x.numpy())
    assert x.grad is None
    assert y.grad is None
    with pytest.raises(RuntimeError, match="freed"):
        output.backward()


def test_autograd_grad_multiple_outputs_seeds_and_unused() -> None:
    x = mt.tensor([2.0, 3.0], requires_grad=True)
    unused = mt.tensor([1.0], requires_grad=True)
    first = x.square()
    second = x.sum()
    grad_x, grad_unused = mt.grad(
        (first, second),
        (x, unused),
        (mt.tensor([2.0, 4.0]), None),
        allow_unused=True,
    )
    np.testing.assert_allclose(grad_x.numpy(), [9.0, 25.0])
    assert grad_unused is None


def test_detect_anomaly_reports_operation_and_nonfinite_gradient() -> None:
    with mt.detect_anomaly():
        value = mt.tensor([0.0], requires_grad=True)
        output = value.sqrt().sum()
        with pytest.raises(RuntimeError, match="sqrt backward"):
            output.backward()
    assert not mt.is_anomaly_enabled()


def test_gradient_norm_and_value_clipping() -> None:
    parameter = mt.nn.Parameter([3.0, 4.0])
    parameter.square().sum().backward()
    original = mt.nn.utils.clip_grad_norm_([parameter], 5.0)
    assert original.item() == pytest.approx(10.0)
    np.testing.assert_allclose(parameter.grad.numpy(), [3.0, 4.0], rtol=1e-5)

    mt.nn.utils.clip_grad_value_(parameter, 2.5)
    np.testing.assert_allclose(parameter.grad.numpy(), [2.5, 2.5])


def test_cuda_memory_statistics_are_consistent() -> None:
    mt.cuda.reset_peak_memory_stats()
    value = mt.ones(4096)
    assert value.device == "cuda:0"
    stats = mt.cuda.memory_stats()
    free, total = mt.cuda.mem_get_info()
    assert stats["allocated_bytes"] > 0
    assert stats["reserved_bytes"] >= stats["allocated_bytes"]
    assert stats["max_allocated_bytes"] >= stats["allocated_bytes"]
    assert total > free > 0
    assert "Peak allocated" in mt.cuda.memory_summary()
    mt.cuda.synchronize()


def test_profiler_records_gpu_operations_and_shapes() -> None:
    left = mt.ones(32, 32)
    right = mt.ones(32, 32)
    with mt.profiler.profile(record_shapes=True) as measured:
        result = (left @ right).relu().sum()
    assert result.item() == pytest.approx(32768.0)
    events = measured.events()
    assert [event.name for event in events] == ["matmul", "relu", "sum"]
    assert events[0].input_shapes == ((32, 32), (32, 32))
    assert all(event.cuda_time_ms >= 0 for event in events)
    assert "matmul" in measured.summary()


def test_collect_env_reports_mytorch_and_cuda() -> None:
    info = mt.utils.get_env_info()
    assert info["mytorch"] == mt.__version__
    assert info["device_count"] >= 1
    assert info["devices"]
    assert "MyTorch environment" in mt.utils.format_env_info(info)
