import numpy as np


def normalize(img, offset=0, scale=1):
    '''normalize intensities according to Hampel estimator.

    NOTE lambda = 0.05 is experimental
    '''
    return (np.tanh(0.05 *
                    (img - img.mean()) / float(img.std()))) * scale + offset
