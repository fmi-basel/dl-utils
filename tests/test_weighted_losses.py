import pytest
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

from dlutils.losses.weighted_losses import weighted_l1_loss, weighted_l2_loss, weighted_binary_crossentropy


@pytest.mark.xfail
def test_weighted_l1_loss():
    '''
    '''
    loss_func = weighted_l1_loss()

    yt_val = np.random.rand(10, 5)
    yt_weight = np.zeros((10, 1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = np.concatenate([yt_val, yt_weight], axis=-1)

    yp = np.zeros_like(yt_val)

    loss_tf = loss_func(yt[None, ...], yp[None, ...])

    loss_np = (np.abs(yt_val - yp) * yt_weight).sum() / 5

    np.testing.assert_almost_equal(loss_tf, loss_np)


@pytest.mark.xfail
def test_weighted_l2_loss():
    '''
    '''
    loss_func = weighted_l2_loss()

    yt_val = np.random.rand(10, 5)
    yt_weight = np.zeros((10, 1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = np.concatenate([yt_val, yt_weight], axis=-1)

    yp = np.zeros_like(yt_val)

    loss_tf = loss_func(yt[None, ...], yp[None, ...])

    loss_np = (np.square(yt_val - yp) * yt_weight).sum() / 5

    np.testing.assert_almost_equal(loss_tf, loss_np)


@pytest.mark.xfail
def test_weighted_binary_crossentropy_loss():
    '''
    '''
    loss_func = weighted_binary_crossentropy(from_logits=False)

    yt_val = np.random.rand(10, 5) > 0.5
    yt_weight = np.zeros((10, 1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = tf.convert_to_tensor(
        np.concatenate([yt_val, yt_weight], axis=-1)[None].astype(np.float32))
    yp = tf.convert_to_tensor(np.random.rand(10, 5)[None].astype(np.float32))

    loss_tf = loss_func(yt, yp)

    loss_np = log_loss(
        yt_val.flat,
        yp.numpy().flat,
        sample_weight=np.broadcast_to(yt_weight, yp.shape).flat)

    np.testing.assert_almost_equal(loss_tf.numpy(), loss_np)
