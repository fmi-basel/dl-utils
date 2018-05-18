import numpy as np


class CosineAnnealingSchedule(object):
    def __init__(self, lr_max, lr_min, epoch_max):
        '''
        '''
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epoch_max = epoch_max

    def schedule(self, epoch, current_lr):
        '''
        '''
        cosine_factor = (
            1 + np.cos(float(epoch % self.epoch_max) / self.epoch_max * np.pi))
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * cosine_factor
