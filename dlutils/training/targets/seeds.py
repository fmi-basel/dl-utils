from scipy.ndimage.filters import gaussian_filter
from scipy.signal import gaussian
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist
from skimage.morphology import thin, skeletonize, skeletonize_3d

import numpy as np

# TODO optimize speed (place vs filter)

def paste_slices(tup):
  pos, w, max_w = tup
  wall_min = max(pos, 0)
  wall_max = min(pos+w, max_w)
  block_min = -min(pos, 0)
  block_max = max_w-max(pos+w, max_w)
  block_max = block_max if block_max != 0 else None
  return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
  loc_zip = zip(loc, block.shape, wall.shape)
  wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
  wall[wall_slices] += block[block_slices]

def generate_seed_map(segmentation, sampling=1.0, sigma=1.5):
    '''Places a gaussian kernel on each instance's center of mass
    
    Notes
    -----
    point on skeleton closest to center mass = center
    '''
    if sampling is None:
        sampling = 1.0
    
    # TODO split help functions (gkernel, get_center, place kernel)
    
    # TODO anisotropic kernel based on sampling
    # ~ kernel_size = int(sigma*4) 
    # ~ kernel_size +=  1 - kernel_size%1 # odd size  kernel
    # ~ gkern1d = gaussian(kernel_size, std=sigma)
    # ~ gkern = np.matmul(gkern1d.reshape(-1,1), gkern1d.reshape(1,-1))
    # ~ if segmentation.ndim == 3:
        # ~ gkern = np.matmul(gkern.reshape(kernel_size,kernel_size,1), gkern1d.reshape(1,1,-1))
        
    seed_map = np.zeros_like(segmentation, dtype=np.float32)
    for label in np.unique(segmentation):
        if label > 0:
            loc = find_objects(segmentation == label)[0]
            offset = np.asarray([sli.start for sli in loc])              
            # ~ #  skeleton = thin(segmentation[loc]==label)
            skeleton = skeletonize_3d(segmentation[loc]==label)
            skeleton = np.argwhere(skeleton)+offset
            center_mass = center_of_mass(segmentation[loc]==label)
            center_mass = np.asarray(center_mass) + offset
            if skeleton.shape[0] > 0: # skeletonize_3d sometimes return empty image??
                dist_to_center_mass = cdist([center_mass], skeleton, 'euclidean')
                center = skeleton[ np.argmin(dist_to_center_mass) ]#.reshape(2,1)
            else:
                center = center_mass.astype(int)
                
            # ~ paste(seed_map, gkern, tuple(center))
            seed_map[tuple(center)] = 1.
    
    seed_map = gaussian_filter(seed_map, sigma=sigma/np.asarray(sampling))
    seed_map = transform = min_max_scaling(seed_map)
        
    return seed_map
