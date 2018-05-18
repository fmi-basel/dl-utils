import numpy as np


def CosineAnnealingSchedule(lr_max, lr_min, epoch_max):

    def schedule(epoch, current_lr):
        '''
        '''
        cosine_factor = (
            1 + np.cos(float(epoch % epoch_max) / epoch_max * np.pi))
        return lr_min + 0.5 * (lr_max - lr_min) * cosine_factor

    return schedule
