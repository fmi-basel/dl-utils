import hdbscan
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.ndimage.morphology import grey_opening, grey_closing
from scipy.ndimage import label as ndimage_label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import relabel_sequential

import warnings
warnings.filterwarnings('once', category=UserWarning, module=__name__)


def add_pixel_coordinates(embeddings, weight=0.001, sampling=1.0):
    '''
    stack weighted pixel coordinates with embeddings
    '''

    ndim = len(embeddings.shape[:-1])
    sampling = np.broadcast_to(np.asarray(sampling), ndim)
    # only keep ratio, work in pixel coordinates
    sampling = sampling / sampling.min()

    coords_list = []
    for dim in range(ndim):
        c = np.fromfunction(
            lambda *p: p[dim] * sampling[dim], embeddings.shape[:-1], dtype=embeddings.dtype)
        coords_list.append(c)

    coords = np.stack(coords_list, axis=-1) * weight
    return np.concatenate([embeddings, coords], axis=-1)


def masked_HDBSCAN(embeddings, fg_mask, min_cluster_size=100, min_samples=10):
    '''Applies HDBSCAN only to foreground objects, also computes mean
    embeddings for each class

    Notes
    -----
    only emeddings in foreground are clustered. HDBSCAN outliers are
    added to background.
    Mean embeddings does not include background label==0
    '''

    n_features = embeddings.shape[-1]
    embeddings_foreground = embeddings[fg_mask]
    if len(embeddings_foreground) > 500000:
        raise RuntimeError('Trying to cluster too many points: {}'.format(
                len(embeddings_foreground)))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='l2',
                                )
    labels_foreground = clusterer.fit_predict(embeddings_foreground)
    labels_foreground += 1  # outliers==1 --> background==0

    unique_labels = np.unique(labels_foreground)
    if len(unique_labels) > 1 and unique_labels.max() > 0:
        clf = NearestCentroid(metric='euclidean')
        clf.fit(embeddings_foreground, labels_foreground)
        centers = clf.centroids_
        if np.unique(labels_foreground)[0] == 0:
            # discard centroid for label 0 (outliers/background)
            centers = centers[1:]
    else:
        centers = np.empty((0, n_features))

    labels = np.zeros(embeddings.shape[:-1])
    labels[fg_mask] = labels_foreground

    return labels, centers


def labels_fill_holes(labels):
    '''Binary fill holes applied to each label.

    Notes:
    ------
    Does not handle label hierarchy. (assumes labels are not nested)
    '''

    for l in np.unique(labels):
        if l > 0:
            l_filled = binary_fill_holes(labels == l)
            labels[l_filled] = l
    return labels


def labels_fill_holes_sliced(labels):
    '''Applies labels_fill_holes one slice at a time

    Notes:
    ------
    Does not handle label hierarchy. (assumes labels are not nested)
    '''

    for z in range(labels.shape[0]):
        labels[z] = labels_fill_holes(labels[z])

    return labels


def split_labels(labels):
    '''
    split in disconnected components
    '''

    labels_split = np.zeros_like(labels)
    total_count = 0
    for l in np.unique(labels):
        if l > 0:
            l_split, n_split = ndimage_label(labels == l)
            l_split[l_split > 0] += total_count
            labels_split += l_split
            total_count += n_split

    return labels_split


def remove_small_labels(labels, threshold=100):
    '''remove labels with pixel count smaller than threshold
    '''

    unique_labels, unique_labels_count = np.unique(labels, return_counts=True)
    small_labels_id = unique_labels[unique_labels_count < threshold]
    mask = np.isin(labels, small_labels_id)
    labels[mask] = 0

    return labels


def cluster_embeddings_3D_sliced(embeddings, fg_mask):
    '''Clusters embeddings with HDBSCAN by processing each slice
    separately and merging centroids on z axis.
    '''

    warnings.warn(
        'step 2: clustering along z axis is not strictly enforced, \
        centroids can still be merged laterally',
        UserWarning)

    labels = np.zeros(shape=embeddings.shape[:-1])
    label_count = 0
    all_centers = np.empty((0, embeddings.shape[-1]))

    # label slices independently
    for z in range(embeddings.shape[0]):

        if fg_mask[z].sum() > 100:
            l, c = masked_HDBSCAN(embeddings[z], fg_mask[z])
            l = labels_fill_holes(l)
            unique_l = np.unique(l)[1:]
            if len(unique_l) > 0:
                l[l > 0] += label_count
                label_count += len(unique_l)

                all_centers = np.concatenate([all_centers, c], axis=0)
                labels[z] = l

    # merge clusters along z
    # TODO: restrict to merging only along z axis
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='l2')
    z_clusters_lut = clusterer.fit_predict(all_centers)
    z_clusters_lut += 1  # outliers merged with background

    fg_labels = labels > 0
    labels[fg_labels] = z_clusters_lut[(
        labels[fg_labels]).astype(np.int) - 1] + 1

    return labels


def embeddings_to_labels(
        embeddings,
        fg_mask,
        coordinate_weight=0.001,
        sampling=1.0,
        size_threshold=300,
        min_samples=100,
        sliced=False):
    ''' Convert embeddings to actual labels.
    '''

    if sliced:
        embeddings = add_pixel_coordinates(
            embeddings, weight=coordinate_weight, sampling=sampling)
        labels = cluster_embeddings_3D_sliced(embeddings, fg_mask)
        labels = split_labels(labels)
        labels = remove_small_labels(labels, threshold=size_threshold)
        labels = renumber_label(labels)

    else:
        embeddings = add_pixel_coordinates(
            embeddings, weight=coordinate_weight, sampling=sampling)
        labels, _ = masked_HDBSCAN(
            embeddings, fg_mask, min_cluster_size=size_threshold, min_samples=min_samples)
        labels = labels_fill_holes_sliced(labels)
        labels = split_labels(labels)
        labels = remove_small_labels(labels, threshold=size_threshold)

        # directly use foreground prediction as label if no cluster found with
        # HDBSCAN (likely a single instance)
        if len(np.unique(labels)) <= 1 and fg_mask.sum() > size_threshold:
            labels = fg_mask.astype(labels.dtype)

        labels, _, _ = relabel_sequential(labels)

    return labels
