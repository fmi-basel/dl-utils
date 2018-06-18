from dlutils.models.resnext import ResneXtBase
from dlutils.models.utils import add_fcn_output_layers

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
            [2, 3] # n_blocks
        )))
def test_resnet_setup(input_shape, n_levels, width, n_blocks):
    '''
    '''
    batch_size = 3

    model = ResneXtBase(
        input_shape=input_shape,
        dropout=0.05,
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



if __name__ == '__main__':
    test_resnet_setup((303, 301, 1), 4, 1, 3)
