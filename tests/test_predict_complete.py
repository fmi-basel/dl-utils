from dlutils.prediction import predict_complete

from dlutils.models.fcn_resnet import ResnetBase
from dlutils.models.utils import add_fcn_output_layers

from itertools import product
import numpy as np
import cv2

import pytest


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from keras.backend import clear_session
    clear_session()


@pytest.mark.parametrize("input_shape,image_shape",
                         list(
                             product([(224, 224, 1), (230, 230, 1)],
                                     [(300, 400, 1), (200, 300, 1)])))
def test_predict_complete(input_shape, image_shape):
    '''
    '''
    batch_size = 1
    model = ResnetBase(input_shape=input_shape)
    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    image = np.random.randn(*image_shape)

    prediction = predict_complete(
        model, image, batch_size=batch_size, border=30)

    for key, val in prediction.iteritems():
        assert val.shape == image.shape, \
            'prediction[{}].shape does not match image.shape! {} != {}'.format(
                key, val.shape, image.shape)


if __name__ == '__main__':
    test_predict_complete((250, 250, 1))
