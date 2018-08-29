from dlutils.models.deeplab import Deeplab
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
    "input_shape,n_levels,width",
    list(
        product(
            [(259, 297, 1), (302, 315, 3)],  # input shapes
            [2, 3],  # n_levels
            [1.,],  # width
        )))
def test_resnext_setup(input_shape, n_levels, width):
    '''
    '''
    batch_size = 3

    model = Deeplab(
        input_shape=input_shape,
        rate_scale=2,
        dropout=0.05,
        n_levels=n_levels,
        width=width)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.summary()

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

    print(model.output_names)
    print(pred[0].shape)
    print(pred[1].shape)
    print(img.shape)
    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


if __name__ == '__main__':
    test_resnext_setup((500, 500, 1), 2, 1.)
