import pytest
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

from dlutils.losses.weighted_losses import PixelWeightedL1Loss, PixelWeightedMSE, PixelWeightedBinaryCrossentropy


def dummy_data(bool_groundtruth=False):

    np.random.seed(8)

    yt_val = np.random.rand(10, 10)
    if bool_groundtruth:
        yt_val = yt_val > 0.5

    yt_weight = np.zeros((10, 10))
    yt_weight[0:5, 0:5] = 3
    yt_weight[5:, 5:] = 1.2
    yt_weight = np.ones_like(yt_val)
    yt_weight = yt_weight / yt_weight.sum()

    # add batch dim in front
    y_true = np.stack([yt_val, yt_weight], axis=-1)[None]
    y_pred = np.random.rand(10, 10, 1)[None]

    return y_true.astype(np.float32), y_pred.astype(np.float32)


def broadcast_to_batch_and_evaluate(loss_func, batch_size, yt, yp):

    yt = np.broadcast_to(yt, (batch_size, ) + yt.shape[1:])
    yp = np.broadcast_to(yp, (batch_size, ) + yp.shape[1:])

    return loss_func(tf.convert_to_tensor(yt), tf.convert_to_tensor(yp))


def test_PixelWeightedL1Loss():
    '''
    '''
    loss_func = PixelWeightedL1Loss()
    yt, yp = dummy_data()

    loss_np = (np.abs(yt[..., 0:1] - yp) * yt[..., 1:]).sum()

    for batch_size in [1, 3, 10]:
        loss_tf = broadcast_to_batch_and_evaluate(loss_func, batch_size, yt,
                                                  yp)
        np.testing.assert_almost_equal(loss_tf, loss_np, decimal=5)


def test_PixelWeightedMSE():
    '''
    '''
    loss_func = PixelWeightedMSE()
    yt, yp = dummy_data()

    loss_np = (np.square(yt[..., 0:1] - yp) * yt[..., 1:]).sum()

    for batch_size in [1, 3, 10]:
        loss_tf = broadcast_to_batch_and_evaluate(loss_func, batch_size, yt,
                                                  yp)
        np.testing.assert_almost_equal(loss_tf, loss_np, decimal=5)


def test_PixelWeightedBinaryCrossentropy():
    '''
    '''
    loss_func = PixelWeightedBinaryCrossentropy(from_logits=False)
    yt, yp = dummy_data(bool_groundtruth=True)

    loss_np = log_loss(yt[..., 0].flat,
                       yp.flat,
                       sample_weight=yt[..., 1:].flat)

    for batch_size in [1, 3, 10]:
        loss_tf = broadcast_to_batch_and_evaluate(loss_func, batch_size, yt,
                                                  yp)
        np.testing.assert_almost_equal(loss_tf, loss_np, decimal=5)
