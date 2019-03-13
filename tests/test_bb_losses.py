from dlutils.losses.bb_losses import masked_smooth_l1_loss, focal_loss

from keras import backend as K
from keras import layers

from itertools import product
import numpy as np

import pytest


@pytest.mark.parametrize("sigma", [0.01, 0.25, 0.5, 1, 2, 3])
def test_smooth_l1(sigma):
    '''TODO make this a proper test.
    '''
    x = layers.Input(shape=(None, ))
    y = layers.Input(shape=(None, ))
    loss_func = K.Function([x, y], [masked_smooth_l1_loss(0, sigma)(x, y)])

    xx = np.linspace(-1, 1, 10)
    xx[xx < 0.5] = 0
    dxx = np.linspace(-10, 10, 1000)
    loss = np.asarray([loss_func([[xx], [xx - dx]]) for dx in dxx]).flatten()
    print(loss)
    assertion = [0 < val < np.abs(dx) for val, dx in zip(loss, dxx)]
    print(assertion)
    assert all(assertion)


@pytest.mark.parametrize("alpha,gamma",
                         list(
                             product([0.1, 0.25, 0.5, 0.75, 0.9],
                                     [0.5, 1.0, 2.0, 3.])))
def test_focal(alpha, gamma):
    '''TODO make this a proper test.
    '''

    x = layers.Input(shape=(None, ))
    y = layers.Input(shape=(None, ))
    loss_func = K.Function([x, y], [focal_loss(alpha, gamma)(x, y)])

    xx = np.linspace(-1, 1, 10) <= 0
    pxx = np.linspace(0.01, 0.99, 1000)
    loss = np.asarray(
        [loss_func([[xx], [px * np.ones_like(xx)]]) for px in pxx]).flatten()
    assert all((0 < val < 5 for val in loss))


if __name__ == '__main__':
    test_focal(0.1, 0.5)
    test_smooth_l1(0.5)
