from scipy.ndimage import find_objects
from math import ceil
import numpy as np


def crop_object(images, labels, margins=0, min_shape=0):
    '''
    Crop all images based on object found in labels.

    To crop labels, pass it in the images list as well.
    '''

    margins = np.broadcast_to(np.asarray(margins), labels.ndim).copy()
    min_shape = np.broadcast_to(np.asarray(min_shape), labels.ndim)

    loc = find_objects(labels >= 1)[0]

    # if needed, increase margins to output object at least min_shape
    object_shape = np.asarray(tuple(sli.stop - sli.start for sli in loc))
    margins = np.maximum(margins, np.ceil((min_shape - object_shape) / 2))

    loc = tuple(
        slice(max(sli.start - margin, 0), sli.stop + margin)
        for sli, margin in zip(loc, margins))

    return [image[loc] for image in images]
