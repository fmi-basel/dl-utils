from keras import backend as K
from keras import layers

import numpy as np

import pytest

from dlutils.losses.weighted_bce import weighted_binary_crossentropy
from keras.losses import binary_crossentropy


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_unweighted(ndim):
    '''make sure result is identical to bce when pos_weight=1.

    '''
    pos_weight = 1

    shape = (None, ) * ndim + (1, )
    x = layers.Input(shape=shape)
    y = layers.Input(shape=shape)
    weighted_bce = K.Function([x, y],
                              [weighted_binary_crossentropy(pos_weight)(x, y)])

    regular_bce = K.function([x, y], [binary_crossentropy(x, y)])

    reshape = (10, ) * ndim + (1, )
    px = np.linspace(0, 1, np.prod(reshape)).reshape(reshape)
    dxx = np.linspace(-0.5, 0.5, 10)
    labels = px > 0.5

    loss = np.asarray(
        [weighted_bce([[labels], [px - dx]]) for dx in dxx]).flatten()
    regular_loss = np.asarray(
        [regular_bce([[labels], [px - dx]]) for dx in dxx]).flatten()

    # import matplotlib.pyplot as plt
    # plt.plot(loss, marker='x')
    # plt.plot(regular_loss, marker='o')
    # plt.show()

    assert all(pytest.approx(x - y, 0.) for x, y in zip(loss, regular_loss))


@pytest.mark.parametrize("pos_weight", [0.5, 2, 5])
def test_loss(pos_weight):
    '''test scaling with pos_weight.

    '''
    x = layers.Input(shape=(None, ))
    y = layers.Input(shape=(None, ))
    loss_func = K.Function([x, y],
                           [weighted_binary_crossentropy(pos_weight)(x, y)])

    regular_bce = K.function([x, y], [binary_crossentropy(x, y)])

    px = np.linspace(0, 1, 100)
    dxx = np.linspace(-0.5, 0.5, 10)
    labels = px > 0.5

    loss = np.asarray(
        [loss_func([[labels], [px - dx]]) for dx in dxx]).flatten()
    regular_loss = np.asarray(
        [regular_bce([[labels], [px - dx]]) for dx in dxx]).flatten()

    # import matplotlib.pyplot as plt
    # plt.plot(loss, marker='x')
    # plt.plot(regular_loss, marker='o')
    # plt.show()

    assert pytest.approx(loss[0] - regular_loss[0], 0.)
    assert pytest.approx(loss[-1] - regular_loss[-1] * pos_weight, 0.)


if __name__ == '__main__':
    test_unweighted(2)
    #test_loss(2)
