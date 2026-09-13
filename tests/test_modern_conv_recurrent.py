import numpy as np

import mytorch as mt


def test_conv2d_known_forward_and_backward():
    layer = mt.nn.Conv2d(1, 1, 2, bias=False, dtype=mt.float64)
    layer.weight._copy_from(mt.ones(layer.weight.shape, dtype=mt.float64))
    value = mt.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=mt.float64,
        requires_grad=True,
    )
    output = layer(value)
    np.testing.assert_array_equal(output.numpy(), [[[[12, 16], [24, 28]]]])
    output.sum().backward()
    np.testing.assert_array_equal(
        value.grad.numpy(), [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]
    )
    np.testing.assert_array_equal(layer.weight.grad.numpy(), [[[[12, 16], [24, 28]]]])


def test_conv_and_transpose_all_dimensions_shapes():
    cases = [
        (mt.nn.Conv1d, mt.nn.ConvTranspose1d, (2, 2, 6)),
        (mt.nn.Conv2d, mt.nn.ConvTranspose2d, (2, 2, 5, 6)),
        (mt.nn.Conv3d, mt.nn.ConvTranspose3d, (1, 2, 4, 5, 6)),
    ]
    for conv_type, transpose_type, shape in cases:
        value = mt.randn(shape, requires_grad=True)
        conv = conv_type(2, 4, 3, padding=1, groups=2)
        output = conv(value)
        assert output.shape[1] == 4
        output.sum().backward()
        assert value.grad.shape == value.shape

        source = mt.randn(shape)
        transpose = transpose_type(2, 4, 3, stride=2, padding=1, output_padding=1)
        assert transpose(source).shape[2:] == tuple(size * 2 for size in shape[2:])


def test_pooling_forward_backward_and_adaptive_shapes():
    value = mt.tensor([[[1.0, 3.0, 2.0, 4.0]]], requires_grad=True)
    maximum = mt.nn.MaxPool1d(2)(value)
    np.testing.assert_array_equal(maximum.numpy(), [[[3, 4]]])
    maximum.sum().backward()
    np.testing.assert_array_equal(value.grad.numpy(), [[[0, 1, 0, 1]]])

    image = mt.randn(2, 3, 7, 9, requires_grad=True)
    average = mt.nn.AvgPool2d(3, stride=2, padding=1)(image)
    adaptive = mt.nn.AdaptiveAvgPool2d((2, 3))(image)
    assert average.shape == (2, 3, 4, 5)
    assert adaptive.shape == (2, 3, 2, 3)
    (average.sum() + adaptive.sum()).backward()
    assert image.grad.shape == image.shape


def test_recurrent_cells_and_multilayer_bidirectional_shapes():
    batch = mt.randn(3, 5, requires_grad=True)
    for cell in (mt.nn.RNNCell(5, 4), mt.nn.GRUCell(5, 4)):
        output = cell(batch)
        output.sum().backward()
        assert output.shape == (3, 4)

    lstm_cell = mt.nn.LSTMCell(5, 4)
    hidden, state = lstm_cell(batch.detach())
    assert hidden.shape == state.shape == (3, 4)

    sequence = mt.randn(2, 6, 5, requires_grad=True)
    gru = mt.nn.GRU(5, 4, 2, batch_first=True, bidirectional=True, dropout=0.2)
    output, hidden = gru(sequence)
    assert output.shape == (2, 6, 8)
    assert hidden.shape == (4, 2, 4)
    output.sum().backward()
    assert sequence.grad.shape == sequence.shape

    lstm = mt.nn.LSTM(5, 4, batch_first=True)
    output, (hidden, state) = lstm(sequence.detach())
    assert output.shape == (2, 6, 4)
    assert hidden.shape == state.shape == (1, 2, 4)
