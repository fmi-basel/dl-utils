import numpy as np


def flip_axis(x, axis):
    '''flip axis of image for data augmentation.
    '''
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
