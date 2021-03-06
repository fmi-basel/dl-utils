from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def CosineAnnealingSchedule(lr_max, lr_min, epoch_max, epoch_max_growth=1., reset_decay=1.):
    '''create a learning rate scheduler that follows the approach from

    SGDR: Stochastic gradient descent with warm restarts,
    Loshchilov & Hutter, ICLR 2017

    TODO implement checkpointing the model each time a reset is done.

    '''
    if lr_max <= lr_min:
        raise ValueError(
            'lr_max has to be larger than lr_min! {} !> {}'.format(
                lr_max, lr_min))

    def schedule(epoch, current_lr=None):
        '''schedule function to be passed to LearningRateScheduler.

        '''
        n_reset = 0
        epoch_since_reset = epoch
        while epoch_since_reset >= 0:
            epoch_since_reset -= epoch_max_growth**n_reset * epoch_max
            n_reset += 1
        
        n_reset -= 1
        epoch_since_reset += epoch_max_growth**n_reset * epoch_max
        
        current_lr_max = reset_decay**n_reset * lr_max
        current_epoch_max = epoch_max_growth**n_reset * epoch_max
        
        cosine_factor = (
            1 + np.cos(float(epoch_since_reset) / current_epoch_max * np.pi))
        return lr_min + 0.5 * (current_lr_max - lr_min) * cosine_factor

    return schedule
    

def stepLRSchedule(step_size=100, gamma=0.1, initial_lr=0.001):
    '''Creates a learning rate scheduler that reduces the learning rate 
    by a factor 'gamma' every 'step_size' epochs.
    '''
    
    def schedule(epoch, current_lr=None):
        '''schedule function to be passed to LearningRateScheduler.

        '''
        
        return initial_lr * gamma**(epoch//step_size)

    return schedule


def get_lr_metric(optimizer):
    '''adds learning rate as a pseudo metric in order to log it.

    '''

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
