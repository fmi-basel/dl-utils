from dlutils.models.unet import GenericUnetBase
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
    "input_shape,width,n_levels,dropout,with_bn",
    list(
        product(
            [
                (259, 297, 1),
            ],  # input shapes
            [0.3, 1, 2],  # width
            [
                2,
                5,
            ],  # n_levels
            [0, 0.5],  # dropout
            [False, True]  # with_bn
        )))
def test_unet_setup(input_shape, width, n_levels, dropout, with_bn):
    '''
    '''
    batch_size = 3

    model = GenericUnetBase(
        input_shape=input_shape,
        width=width,
        n_levels=n_levels,
        dropout=dropout,
        with_bn=with_bn)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    model.summary()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


if __name__ == '__main__':
    test_unet_setup((300, 300, 1), 0.5, 9, True, 0.5)
