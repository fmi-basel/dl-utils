from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import find_objects
from scipy.ndimage.filters import gaussian_filter
from skimage.util import pad, crop

from dlutils.preprocessing.normalization import min_max_scaling
import numpy as np


def generate_distance_transform(segmentation, sampling=1.0, sigma=0.5):
    '''calculate the distance transform separately for each labeled
    cluster and normalizes it.

    '''
    if not isinstance(segmentation, np.ndarray) or segmentation.dtype != int:
        raise ValueError(
            'Expected an integer numpy.ndarray as segmentation labels, got: {}, {}'.format(
                type(segmentation), segmentation.dtype))
    if sampling is None:
        sampling = 1.0

    transform = np.zeros_like(segmentation, dtype=np.float32)
    for label in filter(None, np.unique(segmentation)):
        loc = find_objects(segmentation == label)[0]
        mask = pad(segmentation[loc] == label, 1, 'constant')
        dist = distance_transform_edt(mask, sampling=sampling)
        transform[loc] += min_max_scaling(crop(dist, 1))

    if sigma > 0:
        transform = gaussian_filter(
            transform, sigma=sigma / np.asarray(sampling))
        transform = min_max_scaling(transform)

    return transform


def shrink_labels(segmentation, sampling=1.0, distance_thresh=0.5, val=0):
    '''Thresholds normalized distance transform to shrink labeled instances
    '''
    shrunk_segmentation = segmentation.copy()
    dist = generate_distance_transform(segmentation, sampling=sampling)
    shrunk_segmentation[ (dist<distance_thresh) & (segmentation!=0)] = val

    return shrunk_segmentation


def label_boundaries(segmentation, sampling=1.0, inner_expansion=0.2, outer_expansion=0.2, val=-1):
    '''Define boundaries relative to each instance size
    '''
    
    sampling = np.broadcast_to(np.asarray(sampling), segmentation.ndim)
    boundary_segmentation = segmentation.copy()
    
    for label in filter(None, np.unique(segmentation)):
        loc = find_objects(segmentation == label)[0]
        sizes = [s.stop-s.start for s in loc]
        
        # expand the roi to surrounding region
        loc = tuple(slice(max(0,int(sl.start-si*outer_expansion)), sl.stop+int(si*outer_expansion)) for sl, si in zip(loc,sizes))
        
        # instance
        mask = segmentation[loc] == label
        dist_transform = distance_transform_edt(mask, sampling=sampling)
        dist_threshold_inner = dist_transform.max()*inner_expansion
        dist_threshold_outer = dist_transform.max()*outer_expansion
        boundary_segmentation[loc][(dist_transform<dist_threshold_inner) & mask] = val
        
        # background (with threshold relative to isntance size)
        mask = np.invert(mask)
        dist_transform = distance_transform_edt(mask, sampling=sampling)
        boundary_segmentation[loc][(dist_transform<dist_threshold_outer) & mask] = val
        
    return boundary_segmentation
        
        
