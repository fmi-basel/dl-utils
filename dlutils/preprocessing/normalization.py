import numpy as np


def normalize(img, offset=0, scale=1, min_std=0.):
    '''normalize intensities according to Hampel estimator.

    NOTE lambda = 0.05 is experimental
    '''
    std = img.std()
    if std < min_std:
        std = min_std

    mean = img.mean()
    return (np.tanh(0.05 *
                    (img - mean) / std)) * scale + offset
