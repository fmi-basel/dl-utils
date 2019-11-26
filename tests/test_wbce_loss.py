from tensorflow.keras import backend as K
from tensorflow.keras import layers

import numpy as np

import pytest

from dlutils.losses.weighted_bce import WeightedBinaryCrossEntropy
from tensorflow.keras.losses import binary_crossentropy


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_unweighted(ndim):
    '''make sure result is identical to bce when pos_weight=1.

    '''
    pos_weight = 1

    loss_func = WeightedBinaryCrossEntropy(pos_weight)
    regular_bce = binary_crossentropy

    reshape = (10, ) * ndim + (1, )
    px = np.linspace(0, 1, np.prod(reshape)).reshape(reshape)
    dxx = np.linspace(-0.5, 0.5, 10)
    labels = px > 0.5

    loss = np.asarray(
        [loss_func(labels, px - dx) for dx in dxx]).flatten()
    regular_loss = np.asarray(
        [regular_bce(labels, px - dx) for dx in dxx]).flatten()

    assert all(pytest.approx(x - y, 0.) for x, y in zip(loss, regular_loss))


@pytest.mark.parametrize("pos_weight", [0.5, 2, 5])
def test_loss(pos_weight):
    '''test scaling with pos_weight.

    '''
    loss_func = WeightedBinaryCrossEntropy(pos_weight)
    regular_bce = binary_crossentropy

    px = np.linspace(0, 1, 100)
    dxx = np.linspace(-0.5, 0.5, 10)
    labels = px > 0.5

    loss = np.asarray([loss_func(labels, px - dx) for dx in dxx]).flatten()
    regular_loss = np.asarray(
        [regular_bce(labels, px - dx) for dx in dxx]).flatten()

    assert pytest.approx(loss[0] - regular_loss[0], 0.)
    assert pytest.approx(loss[-1] - regular_loss[-1] * pos_weight, 0.)


if __name__ == '__main__':
    test_unweighted(2)
    #test_loss(2)
