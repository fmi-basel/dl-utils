from dlutils.models.factory import construct_base_model
from dlutils.models.heads import add_fcn_output_layers

from itertools import product

import numpy as np
import pytest


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from tensorflow.keras.backend import clear_session
    clear_session()


@pytest.mark.parametrize(
    "name,input_shape,n_levels",
    list(
        product(
            ['unet', ],
            [(32, 31, 1), (33, 47, 2)],  # input shapes
            [2, 4],  # n_levels
        )))
def test_constructor(name, input_shape, n_levels):
    '''
    '''
    batch_size = 3

    model = construct_base_model(
        name=name, input_shape=input_shape, n_levels=n_levels)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    model.summary()

    assert name in model.name.lower()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


if __name__ == '__main__':
    test_constructor('resnet', (300, 300, 1), 3, 0.5)
