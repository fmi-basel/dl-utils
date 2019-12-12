import tensorflow as tf
import pytest
import numpy as np
from itertools import product

from dlutils.losses.recursive_loss import recursive_loss


@pytest.mark.parametrize("n_steps, n_channels, alpha",
                         list(product([1, 3, 7], [1, 3], [0.4, 1., 1.5])))
def test_recursive_loss(n_steps, n_channels, alpha):

    loss_fun = tf.keras.losses.MeanSquaredError()

    y_preds = tf.random.normal((n_steps, 4, 122, 128, n_channels))
    y_true = tf.random.normal((4, 122, 128, n_channels))

    # TODO better way that rewrite tensorflow op in python?
    manual_loss = sum([
        alpha**(n_steps - (idx + 1)) * loss_fun(y_true, yp)
        for idx, yp in enumerate(y_preds)
    ])
    wg_scaling = sum(
        [alpha**(n_steps - (idx + 1)) for idx in range(len(y_preds))])
    manual_loss /= wg_scaling

    r_loss = recursive_loss(loss_fun, alpha)(y_true, y_preds)
    np.testing.assert_almost_equal(manual_loss.numpy(),
                                   r_loss.numpy(),
                                   decimal=5)

    # simulate y_reds rank change (squeeze) by keras model.fit
    if n_channels == 1:
        r_loss = recursive_loss(loss_fun, alpha)(y_true,
                                                 tf.squeeze(y_preds, axis=-1))
        np.testing.assert_almost_equal(manual_loss.numpy(),
                                       r_loss.numpy(),
                                       decimal=5)
