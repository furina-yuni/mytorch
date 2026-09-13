import numpy as np

import mytorch as mt


def test_scaled_attention_reference_mask_and_backward():
    query = mt.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    key = mt.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    value = mt.tensor([[[2.0, 1.0], [4.0, 3.0]]], requires_grad=True)
    output = mt.nn.functional.scaled_dot_product_attention(query, key, value)
    scores = np.asarray([[[1.0, 0.0], [0.0, 1.0]]]) / np.sqrt(2)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    np.testing.assert_allclose(output.numpy(), weights @ value.numpy(), rtol=1e-6)
    output.sum().backward()
    assert query.grad is not None and key.grad is not None and value.grad is not None

    blocked = mt.zeros(2, 2, dtype=mt.bool)
    safe = mt.nn.functional.scaled_dot_product_attention(
        query.detach(), key.detach(), value.detach(), blocked
    )
    np.testing.assert_array_equal(safe.numpy(), np.zeros_like(safe.numpy()))


def test_multihead_transformer_and_rotary_shapes():
    value = mt.randn(2, 5, 8, requires_grad=True)
    attention = mt.nn.MultiheadAttention(8, 2, batch_first=True)
    output, weights = attention(value, value, value)
    assert output.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 5)

    layer = mt.nn.TransformerEncoderLayer(
        8, 2, 16, dropout=0, batch_first=True, ffn="swiglu"
    )
    encoder = mt.nn.TransformerEncoder(layer, 2)
    encoded = encoder(value)
    encoded.sum().backward()
    assert encoded.shape == value.shape

    rotary = mt.nn.RotaryEmbedding(8, max_seq_len=5)
    cosine, sine = rotary(value)
    query, key = mt.nn.apply_rotary_pos_emb(
        value.detach(), value.detach(), cosine, sine
    )
    assert query.shape == key.shape == value.shape


def test_cross_attention_padding_mask_and_fully_masked_rows_are_finite():
    attention = mt.nn.MultiheadAttention(8, 2, kdim=6, vdim=4, batch_first=True)
    query = mt.randn(2, 3, 8, requires_grad=True)
    key = mt.randn(2, 5, 6, requires_grad=True)
    value = mt.randn(2, 5, 4, requires_grad=True)
    padding = mt.tensor(
        [[False, False, False, True, True], [True, True, True, True, True]],
        dtype=mt.bool,
    )
    output, weights = attention(
        query, key, value, key_padding_mask=padding, is_causal=True
    )
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(weights.numpy()).all()
    np.testing.assert_array_equal(weights.numpy()[1], np.zeros((3, 5)))
    output.sum().backward()
    assert query.grad is not None and key.grad is not None and value.grad is not None


def test_low_rank_lora_merge_and_state_roundtrip():
    value = mt.randn(4, 8, requires_grad=True)
    low_rank = mt.nn.LowRankLinear(8, 5, 3)
    low_rank(value).sum().backward()
    assert low_rank.left.grad is not None and low_rank.right.grad is not None

    lora = mt.nn.LoRALinear(8, 5, rank=2)
    lora.lora_b._copy_from(mt.randn(lora.lora_b.shape) * 0.1)
    expected = lora(value.detach()).numpy()
    lora.merge()
    np.testing.assert_allclose(lora(value.detach()).numpy(), expected, rtol=1e-5)
    clone = mt.nn.LoRALinear(8, 5, rank=2)
    clone.load_state_dict(lora.state_dict())
    assert clone.merged
    np.testing.assert_allclose(clone(value.detach()).numpy(), expected, rtol=1e-5)
    lora.unmerge()
    assert not lora.merged


def test_quantized_and_bitlinear_inference():
    mt.manual_seed(4)
    base = mt.nn.Linear(32, 7)
    value = mt.randn(3, 32)
    int8 = mt.nn.Int8Linear.from_float(base)
    int4 = mt.nn.Int4WeightOnlyLinear.from_float(base, group_size=16)
    np.testing.assert_allclose(int8(value).numpy(), base(value).numpy(), atol=0.03)
    np.testing.assert_allclose(int4(value).numpy(), base(value).numpy(), atol=0.2)

    bit = mt.nn.BitLinear(32, 7)
    training_input = mt.randn(3, 32, requires_grad=True)
    bit(training_input).sum().backward()
    assert bit.weight.grad is not None and training_input.grad is not None
    bit.eval()
    expected = bit(value).numpy()
    packed = bit.to_inference()
    actual = packed(value).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert packed.last_kernel_used
    assert packed.compression_ratio >= 16

    restored = bit.to_inference()
    restored.load_state_dict(packed.state_dict())
    np.testing.assert_allclose(restored(value).numpy(), actual, atol=1e-4, rtol=1e-4)

    half_bit = mt.nn.BitLinear(32, 7, dtype=mt.float16)
    half_bit.eval()
    half_output = half_bit.to_inference()(mt.randn(2, 32, dtype=mt.float16))
    assert half_output.dtype == mt.float16


def test_sparse_moe_routes_and_backpropagates():
    moe = mt.nn.SparseMoE(8, 16, num_experts=4, top_k=2)
    value = mt.randn(2, 3, 8, requires_grad=True)
    output, auxiliary = moe(value)
    assert output.shape == value.shape
    assert auxiliary.shape == ()
    (output.square().mean() + auxiliary).backward()
    assert value.grad is not None
    assert moe.router.projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for expert in moe.experts
        for parameter in expert.parameters()
    )
