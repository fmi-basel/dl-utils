import tensorflow as tf
import pytest
import numpy as np
from itertools import product

from dlutils.losses.recursive_loss import RecursiveLoss


@pytest.mark.parametrize("n_steps, n_channels, alpha",
                         list(product([1, 3, 7], [1, 3], [0.4, 1., 1.5])))
def test_RecursiveLoss(n_steps, n_channels, alpha):

    loss_fun = tf.keras.losses.MeanSquaredError()

    y_preds = tf.random.normal((n_steps, 4, 122, 128, n_channels))
    y_true = tf.random.normal((4, 122, 128, n_channels))

    weights = np.asarray(
        [alpha**(n_steps - (idx + 1)) for idx in range(len(y_preds))])
    weights /= weights.sum()
    manual_loss = sum(weight * loss_fun(y_true, yp)
                      for weight, yp in zip(weights, y_preds))

    r_loss = RecursiveLoss(loss_fun, alpha)(y_true, y_preds)
    np.testing.assert_almost_equal(manual_loss.numpy(),
                                   r_loss.numpy(),
                                   decimal=5)

    # simulate y_reds rank change (squeeze) by keras model.fit
    if n_channels == 1:
        r_loss = RecursiveLoss(loss_fun, alpha)(y_true,
                                                tf.squeeze(y_preds, axis=-1))
        np.testing.assert_almost_equal(manual_loss.numpy(),
                                       r_loss.numpy(),
                                       decimal=5)
