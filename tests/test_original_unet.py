from itertools import product

import numpy as np
import tensorflow as tf
import pytest

from dlutils.models.unet import GenericUnetBase
from dlutils.models.unet import UnetBuilder
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


@pytest.mark.parametrize('base_features', [1, 2, 8, 32])
def test_unet_builder_nfeatures(base_features):

    builder = UnetBuilder(conv_layer=None,
                          upsampling_layer=None,
                          downsampling_layer=None,
                          n_levels=5,
                          n_blocks=2,
                          base_features=base_features)

    num_features = [builder.features_of_level(level) for level in range(5)]
    assert np.all(
        num_features ==
        [base_features * 2**level for level in range(len(num_features))])


@pytest.mark.parametrize('num_blocks, num_levels, expected_trace',
                         [(1, 5, 'BDBDBDBDBUCBUCBUCBUCB'),
                          (2, 5, 'BBDBBDBBDBBDBBUCBBUCBBUCBBUCBB'),
                          (1, 3, 'BDBDBUCBUCB'),
                          (2, 3, 'BBDBBDBBUCBBUCBB'),
                          (4, 3, 'BBBBDBBBBDBBBBUCBBBBUCBBBB'),
                          (5, 1, 'BBBBB'),
                          (1, 2, 'BDBUCB')])
def test_unet_builder_order(num_levels, num_blocks, expected_trace):
    '''checks correctness of builder call order.
    '''
    class InspectingUnetBuilder(UnetBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = ''

        def add_trace(self, char):
            self.trace += char

        def add_single_block(self, *args, **kwargs):
            self.add_trace('B')
            return super().add_single_block(*args, **kwargs)

        def add_downsampling(self, *args, **kwargs):
            self.add_trace('D')
            return super().add_downsampling(*args, **kwargs)

        def add_upsampling(self, *args, **kwargs):
            self.add_trace('U')
            return super().add_upsampling(*args, **kwargs)

        def add_combiner(self, *args, **kwargs):
            self.add_trace('C')
            return super().add_combiner(*args, **kwargs)

    builder = InspectingUnetBuilder(
        conv_layer=tf.keras.layers.Conv2D,
        upsampling_layer=tf.keras.layers.UpSampling2D,
        downsampling_layer=tf.keras.layers.MaxPooling2D,
        n_levels=num_levels,
        n_blocks=num_blocks,
        base_features=8)

    in_tensor = tf.keras.layers.Input(batch_shape=(None, 32, 32, 1))
    builder.build_unet_block(in_tensor)

    assert builder.trace == expected_trace


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
    assert trained_loss + 0.05 < initial_loss
