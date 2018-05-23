from scipy.ndimage.morphology import grey_closing
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries

from numpy import logical_not
from numpy import logical_and
from numpy import float32
from numpy import exp


def generate_border_map(segmentation, border_width=1, decay=10):
    '''calculate border target map from instance segmentation.

    Notes
    -----
    Border map is a detection heatmap calculated as

     f(x) = exp( - dt(x) / decay )

    where dt(..) is the distance transform from the segmentation
    border pixels. If segmentation is an instance segmentation,
    i.e. invidual labels for each instance, then the border will
    outline different instances.

    '''
    border_width = max(border_width - 2, 0)

    boundary = find_boundaries(
        segmentation, connectivity=2, mode='thick', background=0)
    if border_width > 0:
        grey_dilation(boundary, border_width, output=boundary)
    boundary = logical_not(boundary)
    boundary = boundary.astype(float32)
    boundary = distance_transform_edt(boundary)
    boundary = exp(-boundary / decay)
    return boundary


def generate_separator_map(segmentation, border_width=4, decay=10, reach=25):
    '''calculate borders between foreground instances.

    Notes
    -----
    Border map is a detection heatmap calculated as

     f(x) = exp( - dt(x) / decay )

    where dt(..) is the distance transform from the segmentation
    border pixels. If segmentation is an instance segmentation,
    i.e. invidual labels for each instance, then the border will
    outline different instances.

    '''
    border_width = max(border_width - 2, 0)

    dist, indices = distance_transform_edt(
        segmentation == 0, return_indices=True, return_distances=True)
    closest = segmentation[indices.tolist()]

    boundary = find_boundaries(closest, connectivity=2, mode='thick')

    if border_width > 0:
        grey_dilation(boundary, border_width, output=boundary)

    # limit separators to areas close to cells.
    boundary = logical_and(boundary, dist <= reach)

    # turn binary separator map into heatmap
    boundary = logical_not(boundary)
    boundary = boundary.astype(float32)
    boundary = distance_transform_edt(boundary)
    boundary = exp(-boundary / decay)
    return boundary


def close_segmentation(segmentation, size, **kwargs):
    '''close holes in segmentation maps for training.

    '''
    return grey_closing(segmentation, size=size, **kwargs)
