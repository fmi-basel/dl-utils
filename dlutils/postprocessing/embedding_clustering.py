import hdbscan
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.ndimage.morphology import grey_opening, grey_closing
from scipy.ndimage import label as ndimage_label
from scipy.ndimage.morphology import binary_fill_holes


def add_pixel_coordinates(embeddings, weight=0.001, sampling=1.0):
    '''
    stack weighted pixel coordinates with embeddings
    '''
    
    ndim = len(embeddings.shape[:-1])
    if isinstance(sampling, (list,tuple)):
        if len(sampling) == 1:
            sampling = sampling*ndim
        elif len(sampling) != ndim:
            raise ValueError('wrong dimension of sampling argument, expected 1 or {}, got: {}'.format(ndim, len(sampling)) )
    else:
        sampling = (sampling,)*ndim
    sampling = np.asarray(sampling)
    sampling = sampling/sampling.min() # only keep ratio, work in pixel coordinates
    
    
    coords_list = []
    for dim in range(ndim):
        c = np.fromfunction(lambda *p: p[dim]/sampling[dim] , embeddings.shape[:-1], dtype=embeddings.dtype)
        coords_list.append(c)
         
    coords = np.stack(coords_list, axis=-1) * weight    
    return np.concatenate([embeddings, coords], axis=-1)

def masked_HDBSCAN(embeddings, fg_mask, min_cluster_size=100, min_samples=10):
    '''Applies HDBSCAN only to foreground objects, also computes mean embeddings for each class
    
    Notes:
    ------
    only emeddings in foreground are clustered. HDBSCAN outliers are added to background.
    Mean embeddings does not include background label==0
    '''
    
    n_features = embeddings.shape[-1]
    embeddings_foreground = embeddings[fg_mask]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples, 
                                metric='l2',
                                )
    labels_foreground = clusterer.fit_predict(embeddings_foreground)
    labels_foreground += 1 # outliers==1 --> background==0
    
    unique_labels = np.unique(labels_foreground)
    if len(unique_labels) > 1 and unique_labels.max() > 0:
        clf = NearestCentroid(metric='euclidean')
        clf.fit(embeddings_foreground, labels_foreground)
        centers = clf.centroids_
        if np.unique(labels_foreground)[0] == 0:
            centers = centers[1:] # discard centroid for label 0 (outliers/background)
    else:
        centers = np.empty((0,n_features))
    
    labels = np.zeros(embeddings.shape[:-1])
    labels[fg_mask] = labels_foreground
    
    # labels = grey_opening(labels, size=(3,3))
    # labels = grey_closing(labels, size=(3,3))
    labels = labels_fill_holes(labels)
    
    return labels, centers

def labels_fill_holes(labels):
    '''Binary fill holes applied to each label.
    
    Notes:
    ------
    Does not handle label hierarchy. (assumes labels are not nested)
    '''
        
    for l in np.unique(labels):
        if l>0:
            l_filled = binary_fill_holes(labels==l)
            labels[l_filled] = l 
    return labels

def split_labels(labels):
    '''
    split in disconnected components
    '''
    
    labels_split = np.zeros_like(labels)
    total_count = 0
    for l in np.unique(labels):
        if l > 0:
            l_split, n_split = ndimage_label(labels==l)
            l_split[l_split>0] += total_count
            labels_split += l_split 
            total_count += n_split
            
    return labels_split
            
def renumber_label(labels):
    '''Renumber labels from 1 to #labels
    '''
    shape = labels.shape
    u, indices = np.unique(labels, return_inverse=True)
    lut = np.asarray([i for i in range(len(u))])
    labels = lut[indices]
    labels = np.reshape(labels, shape)
    
    return labels

def remove_small_labels(labels, threshold=1000):
    '''remove labels with pixel count smaller than threshold
    '''

    unique_labels, unique_labels_count = np.unique(labels, return_counts=True)
    small_labels_id = unique_labels[unique_labels_count<1000]
    mask = np.isin(labels,small_labels_id)
    labels[mask] = 0
    labels = renumber_label(labels)
    
    return labels

def cluster_embeddings_3D(embeddings, fg_mask, coordinate_weight=0.001, sampling=1.0):
    '''Clusters embeddings with HDBSCAN by processing each slice separately and merging centroids on z axis.
    '''

    embeddings = add_pixel_coordinates(embeddings, weight=coordinate_weight, sampling=sampling)
    labels = np.zeros(shape=embeddings.shape[:-1])
    label_count = 0
    all_centers = np.empty((0,embeddings.shape[-1]))

    # label slice independtly
    for z in range(embeddings.shape[0]):
    
        if fg_mask[z].sum() > 0:
            # ~ print('processing slice: ', z)
            l,c = masked_HDBSCAN(embeddings[z], fg_mask[z])
            unique_l = np.unique(l)[1:]
            if len(unique_l) > 0:
                l[l>0] += label_count
                label_count += len(unique_l)
            
                all_centers = np.concatenate([all_centers, c], axis=0)
                labels[z] = l

    # min cluster size == min n_slices a cell must span (not enforced, centroids still merged laterally)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='l2')
    z_clusters_lut = clusterer.fit_predict(all_centers)
    z_clusters_lut += 1 # outliers merged with background

#     print('number of z outliers', (z_clusters==0).sum())
    fg_labels = labels>0
    labels[fg_labels] = z_clusters_lut[ (labels[fg_labels]).astype(np.int)-1 ]+1
    
    return labels

def embeddings_to_labels(embeddings, fg_mask, coordinate_weight=0.001, sampling=1.0, size_threshold=1000):
    ''' Convert embeddings to actual labels.
    '''
    
    labels = cluster_embeddings_3D(embeddings, fg_mask, coordinate_weight=coordinate_weight, sampling=sampling)
    labels = split_labels(labels)
    labels = remove_small_labels(labels, threshold=size_threshold)
    
    return labels
