from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def CosineAnnealingSchedule(lr_max, lr_min, epoch_max, reset_decay=1):
    '''create a learning rate scheduler that follows the approach from

    SGDR: Stochastic gradient descent with warm restarts,
    Loshchilov & Hutter, ICLR 2017

    TODO implement increasing epoch_max multiplier.
    TODO implement checkpointing the model each time a reset is done.

    '''
    if lr_max <= lr_min:
        raise ValueError(
            'lr_max has to be larger than lr_min! {} !> {}'.format(
                lr_max, lr_min))

    def schedule(epoch, current_lr=None):
        '''schedule function to be passed to LearningRateScheduler.

        '''
        current_lr_max = reset_decay**-int(epoch / epoch_max) * lr_max
        cosine_factor = (
            1 + np.cos(float(epoch % epoch_max) / epoch_max * np.pi))
        return lr_min + 0.5 * (current_lr_max - lr_min) * cosine_factor

    return schedule


def get_lr_metric(optimizer):
    '''adds learning rate as a pseudo metric in order to log it.

    '''

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
