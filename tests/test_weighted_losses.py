from dlutils.losses.weighted_losses import weighted_l1_loss, weighted_l2_loss, weighted_binary_crossentropy

from keras import backend as K
from keras import layers

import numpy as np
from sklearn.metrics import log_loss

import pytest


def test_weighted_l1_loss():
    '''
    '''
    y_true = layers.Input(shape=(None,6,))
    y_pred = layers.Input(shape=(None,5,))
    loss_func = K.Function([y_true, y_pred], [weighted_l1_loss()(y_true, y_pred)])

    yt_val = np.random.rand(10,5)
    yt_weight = np.zeros((10,1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = np.concatenate([yt_val, yt_weight], axis=-1)
    
    yp = np.zeros_like(yt_val)

    loss_tf = loss_func([yt[None, ...], yp[None, ...]])
    
    loss_np = (np.abs(yt_val-yp) * yt_weight).sum() / 5
    
    np.testing.assert_almost_equal(loss_tf, loss_np)
    
def test_weighted_l2_loss():
    '''
    '''
    y_true = layers.Input(shape=(None,6,))
    y_pred = layers.Input(shape=(None,5,))
    loss_func = K.Function([y_true, y_pred], [weighted_l2_loss()(y_true, y_pred)])

    yt_val = np.random.rand(10,5)
    yt_weight = np.zeros((10,1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = np.concatenate([yt_val, yt_weight], axis=-1)
    
    yp = np.zeros_like(yt_val)

    loss_tf = loss_func([yt[None, ...], yp[None, ...]])
    
    loss_np = (np.square(yt_val-yp) * yt_weight).sum() / 5
    
    np.testing.assert_almost_equal(loss_tf, loss_np)


def test_weighted_binary_crossentropy_loss():
    '''
    '''
    y_true = layers.Input(shape=(None,6,))
    y_pred = layers.Input(shape=(None,5,))
    loss_func = K.Function([y_true, y_pred], [weighted_binary_crossentropy(from_logits=False)(y_true, y_pred)])

    yt_val = np.random.rand(10,5)>0.5
    yt_weight = np.zeros((10,1))
    yt_weight[0] = 0.75
    yt_weight[7] = 0.25
    yt = np.concatenate([yt_val, yt_weight], axis=-1)
    
    yp = np.random.rand(10,5)

    loss_tf = loss_func([yt[None, ...], yp[None, ...]])
    
    loss_np = log_loss(yt_val.flat, yp.flat, sample_weight=np.broadcast_to(yt_weight, yp.shape).flat)
    
    np.testing.assert_almost_equal(loss_tf, loss_np)
