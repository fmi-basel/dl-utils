from itertools import product

import numpy as np
import tensorflow as tf
import pytest

from dlutils.models.unet import GenericUnetBase
from dlutils.models.unet import _abbreviate
from dlutils.models.heads import add_fcn_output_layers


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''make sure models are gone after each test.
    '''
    from tensorflow.keras.backend import clear_session
    clear_session()


def test_abbreviate():
    '''check if the abbreviation convenience function yields the expected short.
    '''
    from tensorflow.keras.layers import BatchNormalization, LayerNormalization

    assert _abbreviate(BatchNormalization.__name__) == 'BN'
    assert _abbreviate(LayerNormalization.__name__) == 'LN'


@pytest.mark.parametrize(
    "input_shape,width,n_levels,with_bn",
    list(
        product(
            [(16, 19, 1), (16, 21, 16, 2)],  # input shapes
            [0.3, 1., 1.2],  # width
            [2, 4],  # n_levels
            [False, True]  # with_bn
        )))
def test_unet_setup(input_shape, width, n_levels, with_bn):
    '''test model constructor and forward pass.
    '''
    batch_size = 3

    model = GenericUnetBase(input_shape=input_shape,
                            width=width,
                            n_levels=n_levels,
                            with_bn=with_bn)
    model = add_fcn_output_layers(
        model,
        ['pred'],
        [1],
    )

    model.compile(optimizer='adam', loss={
        'pred': 'binary_crossentropy',
    })

    model.summary()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))
    assert np.all(0 <= pred) and np.all(pred <= 1.)


@pytest.mark.parametrize(
    "input_shape,n_levels,with_bn",
    list(
        product(
            [(17, 15, 1), (5, 8, 10, 2)],  # shapes
            [2, 3],  # levels
            [False, True]  # batch norm
        )))
def test_training_unet(input_shape, n_levels, with_bn):
    '''test if fit decreases loss.
    '''
    model = GenericUnetBase(input_shape=input_shape,
                            width=0.5,
                            n_levels=n_levels,
                            with_bn=with_bn)
    model = add_fcn_output_layers(
        model,
        ['pred'],
        [1],
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='bce')

    def generate_dummy_data(shape, batch_size, num_batches):
        np.random.seed(42)
        shape = (batch_size, ) + shape
        for _ in range(num_batches):
            inputs = np.random.randn(*shape)
            yield inputs, tf.reduce_max(inputs, axis=-1, keepdims=True) > 0

    ins, outs = zip(*list(generate_dummy_data(input_shape, 3, 10)))
    initial_loss = model.evaluate(ins, outs)

    # train model
    model.fit(ins, outs, epochs=10)
    trained_loss = model.evaluate(ins, outs, verbose=False)
    assert trained_loss + 0.1 < initial_loss
