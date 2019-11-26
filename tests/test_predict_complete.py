from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dlutils.prediction import predict_complete

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


@pytest.mark.parametrize("input_shape,image_shape,border",
                         list(
                             product([(224, 224, 1), (230, 230, 1)],
                                     [(300, 400, 1),
                                      (200, 300, 1)], [30, (30, 20)])))
def test_predict_complete(input_shape, image_shape, border):
    '''
    '''
    batch_size = 1
    model = GenericUnetBase(input_shape=input_shape, n_levels=2, n_blocks=1)
    model.summary(line_length=150)
    
    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    image = np.random.randn(*image_shape)

    prediction = predict_complete(
        model, image, batch_size=batch_size, border=border)

    for key, val in prediction.items():
        assert val.shape == image.shape, \
            'prediction[{}].shape does not match image.shape! {} != {}'.format(
                key, val.shape, image.shape)


if __name__ == '__main__':
    test_predict_complete((250, 250, 1), (300, 351, 1))
