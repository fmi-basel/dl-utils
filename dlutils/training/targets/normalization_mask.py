import numpy as np

def generate_normalization_mask(segmentation, include_background=False):
    ''' Computes an instance wise weighting mask.
    
    Notes:
    ------
    Background expected to be labeled as 0
    '''

    normalization = np.zeros_like(segmentation, dtype=np.float32)
    n_labels = 0
    for label in np.unique(segmentation):
        if include_background or label > 0:
            instance_mask = segmentation==label
            area = instance_mask.sum()    
            normalization[instance_mask] = 1./area
            n_labels += 1
    
    if n_labels > 1:        
        normalization = normalization / n_labels
    
    return normalization
    
