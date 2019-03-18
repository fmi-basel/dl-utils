from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dlutils.models.hourglass import GenericHourglassBase
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

params_isotropic = list(
        product(
            [
                (256, 256, 3),
                (310, 199, 1),
                (64,64,64, 1),
                (69,22,98, 1),
            ],  # input shapes
            [1, 0.5],  # width
            [1, 4],  # cardinality
            [1, 2], # n_stacks
            [1,4],  # n_levels
            [True, False],  # with_bn
            [False],
            [0.5, ] ,  # dropout
        ))
        
params_anisotropic = list(
        product(
            [
                (69,22,98, 1),
            ],  # input shapes
            [1, 0.5],  # width
            [1, 4],  # cardinality
            [1, 2], # n_stacks
            [1,4],  # n_levels
            [True, False],  # with_bn
            [True],
            [0.0, ] ,  # dropout
        ))


@pytest.mark.parametrize(
    "input_shape,width,cardinality,n_stacks,n_levels,with_bn,anisotropic,dropout",
    params_isotropic + params_anisotropic
    )
def test_setup(input_shape, width, cardinality, n_stacks, n_levels, with_bn, anisotropic, dropout):
    '''
    '''
    batch_size = 3

    model = GenericHourglassBase(
        input_shape=tuple(None for _ in range(len(input_shape)-1)) + (input_shape[-1],),
        width=width,
        cardinality=cardinality,
        n_stacks=n_stacks,
        n_levels=n_levels,
        with_bn=with_bn,
        anisotropic=anisotropic,
        dropout=dropout)

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
        print(img.shape, pred.shape)
        assert all(x == y for x, y in zip(pred.shape[:-1], img.shape))


if __name__ == '__main__':
    test_setup((64,64,64, 1), 1, 4, 1, 1, False, True, 0.5)
    
