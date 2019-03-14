from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model
from keras import optimizers
from dlutils.models import load_model

from dlutils.layers.grouped_conv import GroupedConv2D, GroupedConv3D

from scipy.ndimage.filters import gaussian_filter

import numpy as np
import pytest
import os


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from keras.backend import clear_session
    clear_session()


@pytest.mark.parametrize("conv_fct, patch_size, in_features, out_features, cardinality",
                         [[GroupedConv2D, (100, 50), 10, 8, 2], 
                          [GroupedConv2D, (32, 37), 8, 4, 4],
                          [GroupedConv3D, (100, 50, 48), 10, 8, 2], 
                          [GroupedConv3D, (32, 37, 11), 8, 4, 4],])
def test_layer(conv_fct, patch_size, in_features, out_features, cardinality):
    '''
    '''
    input_shape = patch_size + (in_features, )

    input_layer = Input(input_shape)
    output = conv_fct(
        out_features, kernel_size=(3), cardinality=cardinality)(input_layer)
    model = Model(inputs=input_layer, outputs=output)

    optimizer = optimizers.get('sgd')
    optimizer.lr = 0.1
    model.compile(optimizer, loss='mae')
    model.summary()

    # prepare some dummy data.
    x = np.random.randn(100, *input_shape)
    y = np.asarray([gaussian_filter(xx, sigma=3) for xx in x])
    y = y[..., :out_features]

    # test training and inference.
    model.fit(x, y, verbose=1, epochs=3, batch_size=50)
    pred = model.predict(x)

    assert all(x == y for x, y in zip(pred.shape[:-1], x.shape[:-1]))
    assert pred.shape[-1] == out_features


@pytest.mark.parametrize("conv_fct, patch_size, cardinality, strides",
                         [[GroupedConv2D, (100, 50), 2, 2], 
                          [GroupedConv2D, (33, 39), 4, 3],
                          [GroupedConv3D, (100, 50, 48), 2, 2], 
                          [GroupedConv3D, (33, 39, 39), 4, 3],])
def test_layer_strided(conv_fct, patch_size, cardinality, strides):
    '''
    '''
    in_features = 8
    out_features = 4
    input_shape = patch_size + (in_features, )
    training_size = 10
                    
    target_shape = (training_size,) + tuple(s//strides for s in patch_size) + (out_features,)

    input_layer = Input(input_shape)
    output = conv_fct(
        out_features,
        kernel_size=(3),
        cardinality=cardinality,
        strides=strides,
        padding='same')(input_layer)
    model = Model(inputs=input_layer, outputs=output)

    optimizer = optimizers.get('sgd')
    optimizer.lr = 0.1
    model.compile(optimizer, loss='mae')
    model.summary()

    # prepare some dummy data.
    x = np.random.randn(training_size, *input_shape)
    y = np.asarray([
        gaussian_filter(yy, sigma=3) for yy in np.random.randn(*target_shape)
    ])

    # test training and inference.
    model.fit(x, y, verbose=1, epochs=3, batch_size=50)
    pred = model.predict(x)

    assert all(x == y for x, y in zip(pred.shape, target_shape))


@pytest.mark.parametrize("conv_fct, input_shape",
                         [[GroupedConv2D, (200, 200, 40)], 
                          [GroupedConv3D, (30, 200, 200, 40)]])
def test_layer_save(conv_fct, input_shape):
    '''
    '''
    cardinality = 10
    out_features = 20
    input_layer = Input(input_shape)
    output = conv_fct(
        out_features, kernel_size=(3), cardinality=cardinality)(input_layer)
    model = Model(inputs=input_layer, outputs=output)
    model.compile('sgd', loss='mae')

    out_path = os.path.join('/tmp', 'grouped_conv_model_UT.h5')
    model.save(out_path)

    new_model = load_model(out_path)
    model.summary()
    new_model.summary()


if __name__ == '__main__':
    test_layer((100, 100), in_features=8, out_features=4, cardinality=2)
    test_layer_save()
