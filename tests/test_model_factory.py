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
    "name,input_shape,n_levels,dropout",
    list(
        product(
            ['unet', ],
            [(259, 297, 1), (300, 313, 2)],  # input shapes
            [3, 5],  # n_levels
            [0, 0.05],  # dropout
        )))
def test_constructor(name, input_shape, n_levels, dropout):
    '''
    '''
    batch_size = 3

    model = construct_base_model(
        name=name, input_shape=input_shape, n_levels=n_levels, dropout=dropout)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    model.summary()

    print(name)
    print(model.name)

    assert name in model.name.lower()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


if __name__ == '__main__':
    test_constructor('resnet', (300, 300, 1), 3, 0.5)
