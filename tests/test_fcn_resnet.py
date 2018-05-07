from detection.detectors.fcn_resnet import construct_resnet23
from detection.detectors.fcn_resnet import construct_resnet11
from detection.Detector import make_detector

from test_utils import get_test_image

from keras.backend import clear_session

import os
import pytest
import numpy as np


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    yield

    # make sure models are gone after each test.
    clear_session()


@pytest.mark.parametrize("constructor",
                         [construct_resnet11, construct_resnet23])
def test_original_compile(constructor, batch_size=1):
    '''
    '''
    patch_size = (224, 224, 3)
    model = constructor(
        batch_size=batch_size,
        input_shape=patch_size,
        no_classes=1,
        learning_rate=1e-3,
        loss='mae')

    img = np.zeros((batch_size, ) + patch_size)
    exp = np.ones((batch_size, ) + patch_size[:-1] + (1, ))
    scores = model.evaluate(img, exp)
    assert scores[0] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("batch_size", [1, 7])
def test_resnet23_without_compilation(batch_size):
    '''
    '''
    weight_file = os.path.abspath(
        '/home/markus/workspace/hm-cell-tracking/experiments/cv-training/resnet-23/split-0/model_checkpoints/model.hdf5'
    )

    assert os.path.exists(weight_file)

    patch_size = (224, 224, 3)
    model = construct_resnet23(
        batch_size=batch_size,
        input_shape=patch_size,
        no_classes=1,
        weight_file=weight_file)

    detector = make_detector(model, border_size=15)
    img = get_test_image(load=True)
    pred = detector.predict_complete(img)

    assert pred.shape == img.shape[:-1]
    assert pred.min() == pytest.approx(0.)
    assert pred.max() >= 0.9
    assert pred.max() <= 1.0


if __name__ == '__main__':
    test_original_compile()
