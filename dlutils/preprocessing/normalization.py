from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np


def normalize(img, offset=0, scale=1, min_std=0.):
    '''normalize intensities according to Hampel estimator.

    NOTE lambda = 0.05 is experimental
    '''
    std = img.std()
    if std < min_std:
        std = min_std

    mean = img.mean()
    return (np.tanh((img - mean) / std)) * scale + offset
    
    
def min_max_scaling(img, min_val=0, max_val=1, eps=1e-5):
    '''Re-scale input to be within [min_val, max_val]
    
    '''
    
    img -= img.min()
    img = img / (img.max()+eps) * (max_val-min_val)

    return img + min_val


def standardize(img, min_scale=0.):
    '''normalize intensities according to whitening transform.
    '''
    mean = img.mean()
    scale = max(min_scale, img.std())
    return (img - mean) / float(scale)
