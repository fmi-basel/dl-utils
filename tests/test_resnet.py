from dlutils.models.fcn_resnet import ResnetBase
from dlutils.models.heads import add_fcn_output_layers

from itertools import product

import numpy as np
import pytest


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from keras.backend import clear_session
    clear_session()


@pytest.mark.parametrize(
    "input_shape,n_levels,width,n_blocks",
    list(
        product(
            [(259, 297, 1), (302, 315, 3)],  # input shapes
            [2, 5],  # n_levels
            [0.3, 1.2],  # width
            [2, 3]  # n_blocks
        )))
def test_resnet_setup(input_shape, n_levels, width, n_blocks):
    '''
    '''
    batch_size = 3

    model = ResnetBase(
        input_shape=input_shape,
        with_bn=True,
        dropout=0.5,
        n_levels=n_levels,
        width=width,
        n_blocks=n_blocks)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    model.summary()
    print(model.name)

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


def test_preprocessed():

    from keras.layers import MaxPooling2D
    from keras.engine import Input
    from keras.models import Model
    from keras.layers import Concatenate
    from keras.layers import Lambda

    input_shape = (500, 500, 1)
    batch_size = 3
    input_tensor = Input(batch_shape=(None, ) + input_shape, name='input')
    x_max = MaxPooling2D(pool_size=2)(input_tensor)
    negative = Lambda(lambda x: -x)(input_tensor)
    x_min = MaxPooling2D(pool_size=2)(negative)

    x = Concatenate()([x_min, x_max])

    resnet = ResnetBase(input_tensor=x, with_bn=True, dropout=0.5)

    model = Model(inputs=input_tensor, outputs=resnet.outputs)
    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(optimizer='adam', loss='mean_absolute_error')

    model.summary()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        print(name, pred.shape)


if __name__ == '__main__':
    test_resnet_setup((300, 300, 1), 3, 0.2, 3)
