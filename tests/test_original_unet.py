from itertools import product

import numpy as np
import pytest

from dlutils.models.unet import GenericUnetBase
from dlutils.models.heads import add_fcn_output_layers


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from tensorflow.keras.backend import clear_session
    clear_session()


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
    '''
    '''
    batch_size = 3

    model = GenericUnetBase(
        input_shape=input_shape,
        width=width,
        n_levels=n_levels,
        with_bn=with_bn)

    pred_names = [
        'pred',
    ]
    model = add_fcn_output_layers(
        model,
        pred_names,
        [
            1,
        ],
    )

    model.compile(
        optimizer='adam', loss={
            'pred': 'binary_crossentropy',
        })

    model.summary()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))
    assert np.all(0 <= pred) and np.all(pred <= 1.)


if __name__ == '__main__':
    test_unet_setup((300, 300, 1), 0.5, 9, True, 0.5)
