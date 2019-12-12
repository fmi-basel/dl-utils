from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def weighted_l1_loss():
    '''Computes L1 loss, using the last channel of y_true as 
    normalization weights

    Notes:
    ------
    - sum of weights == 1 is expected
    - implemented as channels_last

    '''
    def loss(y_true, y_pred):
        '''
        '''
        # extract pre-computed normalization channel
        weights = y_true[..., -1:]
        y_true = y_true[..., 0:-1]

        loss = K.abs(y_pred - y_true)
        # sum over spatial dims, mean over batch and channel dims
        loss = tf.math.reduce_sum(loss * weights,
                                  axis=tuple(range(1, len(y_pred.shape))))
        return tf.math.reduce_mean(loss)

    return loss


def weighted_l2_loss():
    '''Computes L1 loss, using the last channel of y_true as 
    normalization weights

    Notes:
    ------
    - sum of weights == 1 is expected
    - implemented as channels_last

    '''
    def loss(y_true, y_pred):
        '''
        '''
        # extract pre-computed normalization channel
        weights = y_true[..., -1:]
        y_true = y_true[..., 0:-1]

        loss = K.square(y_pred - y_true)
        # sum over spatial dims, mean over batch and channel dims
        loss = tf.math.reduce_sum(loss * weights,
                                  axis=tuple(range(1, len(y_pred.shape))))
        return tf.math.reduce_mean(loss)

    return loss


def weighted_binary_crossentropy(from_logits=False):
    '''Computes binary cross-entropy loss, using the last channel of 
    y_true  as normalization weights

    Notes:
    ------
    - sum of weights == 1 is expected
    - implemented as channels_last

    '''
    def loss(y_true, y_pred):
        '''
        '''

        # extract pre-computed normalization channel
        weights = y_true[..., -1:]
        y_true = y_true[..., 0:-1]

        loss = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
        # sum over spatial dims, mean over batch and channel dims
        loss = tf.math.reduce_sum(loss * weights,
                                  axis=tuple(range(1, len(y_pred.shape))))
        return tf.math.reduce_mean(loss)

    return loss
