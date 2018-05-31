from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label

import numpy as np


def segment_nuclei(cell_pred, border_pred, threshold, upper_threshold,
                   smoothness, obj_size):
    '''watershed based segmentation of nuclei.

    '''
    combined = cell_pred * (1 - border_pred)
    gaussian_filter(combined, sigma=smoothness, output=combined)

    # maxima are going to be the seeds.
    maxima = maximum_filter(combined, size=obj_size) == combined

    # Filter out maxima
    maxima[combined < threshold] = False
    markers = label(np.logical_or(combined > upper_threshold, maxima))[0]
    markers[np.logical_not(maxima)] = 0

    # TODO consider removing small objects or seeds from small objects.

    segmentation = watershed(
        -combined,
        markers=markers,
        mask=cell_pred > threshold,
        watershed_line=True)
    return segmentation
