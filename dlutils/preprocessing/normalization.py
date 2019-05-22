from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import warnings
warnings.filterwarnings('once', category=UserWarning, module=__name__)


def normalize(img, offset=0, scale=1, min_std=0.):
    '''normalize intensities according to Hampel estimator.

    NOTE lambda = 0.05 is experimental
    '''
    std = img.std()
    if std < min_std:
        std = min_std

    mean = img.mean()
    return (np.tanh((img - mean) / std)) * scale + offset


def min_max_scaling(img,
                    min_val=0,
                    max_val=1,
                    eps=1e-5,
                    saturation=0.0,
                    separate_channels=False):
    '''Re-scale input to be within [min_val, max_val]
    
    '''
    if separate_channels:
        if img.shape[-1] > 3:
            warnings.warn(
                'Normalizing channels separately, n channels = {}'.format(
                    img.shape[-1]), UserWarning)

        axis = tuple(range(img.ndim - 1))
        keepdims = True
    else:
        axis = None
        keepdims = False

    img = img.astype(dtype=np.float, copy=True)
    img -= np.quantile(img, saturation, axis=axis, keepdims=keepdims)  # min
    img = img / (np.quantile(img, 1 - saturation, axis=axis, keepdims=keepdims)
                 + eps) * (max_val - min_val)
    img = img + min_val
    img = np.clip(img, min_val, max_val)

    return img


def standardize(img, min_scale=0., separate_channels=False):
    '''normalize intensities according to whitening transform.
    '''

    if separate_channels:
        if img.shape[-1] > 3:
            warnings.warn(
                'Normalizing channels separately, n channels = {}'.format(
                    img.shape[-1]), UserWarning)

        axis = tuple(range(img.ndim - 1))
        keepdims = True
        min_scale = np.broadcast_to(np.asarray(min_scale), img.shape[-1])
    else:
        axis = None
        keepdims = False
        min_scale = np.asarray(min_scale)

    mean = img.mean(axis=axis, keepdims=keepdims)
    scale = np.maximum(min_scale, img.std(axis=axis, keepdims=keepdims))
    return (img - mean) / scale
