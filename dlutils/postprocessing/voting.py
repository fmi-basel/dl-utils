import tensorflow as tf
import numpy as np
from dlutils.improc import local_max

from tensorflow.python.ops.gen_clustering_ops import nearest_neighbors


def count_votes(fg_embeddings, spatial_shape, spacing=1):
    '''Computes 2D/3D voting histogram of embeddings in px coordinates
    
    Args:
        fg_embeddings: foreground semi-conv embeddings of shape [n_foreground_px, rank]
        spatial_shape: spatial shape of the original image
        spacing: pixel/voxel size
    '''

    rank = fg_embeddings.shape[1]
    n_pixels = tf.reduce_prod(spatial_shape)
    spacing = np.broadcast_to(np.asarray(spacing), rank)[None]
    fg_embeddings_px = tf.cast(tf.round(fg_embeddings / spacing), tf.int32)

    # flatten the spatial domain to a single dimension with linear index
    # --> use native tensorflow 1D histogram
    # reshape to original spatial rank
    flat_emb = fg_embeddings_px[..., -1]
    for idx in range(rank - 2, -1, -1):
        flat_emb += tf.reduce_prod(
            spatial_shape[idx + 1:]) * fg_embeddings_px[..., idx]

    votes = tf.histogram_fixed_width(flat_emb,
                                     value_range=(0, n_pixels),
                                     nbins=n_pixels)
    return tf.reshape(votes, spatial_shape)


def embeddings_to_labels(embeddings,
                         fg_mask,
                         peak_min_distance,
                         spacing=1.,
                         min_count=5,
                         return_centers=False):
    '''Splits a foreground mask into instance labels defined by semi-convolutional embeddings.
    
    Args:
        embeddings: semi-conv embeddings of shape spatial_shape + [rank]
        fg_mask: foreground mask of shape spatial_shape
        spacing: pixel/voxel size
        peak_min_distance: minimum distance between instance centers
        min_count: minimum number of vote to consider an instance
        return_centers: if true returns the embedding centers as second argument
    
    Notes:
    embeddings obtained from semi-conv layer are expected to be in isotropic coords
    '''

    spacing = np.broadcast_to(np.asarray(spacing), len(fg_mask.shape))

    fg_embeddings = tf.boolean_mask(embeddings, fg_mask)
    spatial_shape = tf.shape(embeddings)[:-1]
    votes = count_votes(fg_embeddings, spatial_shape, spacing)

    centers = local_max(votes,
                        min_distance=peak_min_distance,
                        threshold=min_count,
                        spacing=spacing)

    centers = tf.cast(centers, tf.float32)
    centers = centers * spacing[None]

    # handle empty fg_embeddings/centers which fail in "nearest_neighbors"
    def true_fun():
        return tf.zeros(tf.shape(fg_mask), dtype=tf.int32)

    def false_fun():
        fg_labels = nearest_neighbors(fg_embeddings, centers, 1)[0][:, 0]
        fg_labels = tf.cast(fg_labels, tf.int32)
        return tf.scatter_nd(tf.where(fg_mask), fg_labels + 1,
                             tf.cast(tf.shape(fg_mask), tf.int64))

    labels = tf.cond(tf.less_equal(tf.shape(centers)[0], 0), true_fun,
                     false_fun)

    if return_centers:
        return labels, centers
    else:
        return labels


def seeded_embeddings_to_labels(embeddings,
                                fg_mask,
                                seeds=None,
                                dist_threshold=None,
                                centers=None):
    '''Converts voting embeddings to labels.
    
    dist_threshold: maximum distance from embeddings to its center. usefull to segment with partial seeds'''

    if seeds is None and centers is None:
        raise ValueError(
            'at least one of seeds or centers (priority) need to be specified')

    if centers is None:
        centers = tf.gather_nd(embeddings, seeds)

    fg_embeddings = tf.boolean_mask(embeddings, fg_mask)

    def true_fun():
        return tf.zeros(tf.shape(fg_mask), dtype=tf.int32)

    def false_fun():
        fg_labels = nearest_neighbors(fg_embeddings, centers, 1)[0][:, 0]

        if dist_threshold is not None:
            dist = tf.norm(fg_embeddings - tf.gather(centers, fg_labels),
                           axis=-1)
            fg_labels = tf.where(dist <= dist_threshold, fg_labels, -1)

        fg_labels = tf.cast(fg_labels, tf.int32)
        return tf.scatter_nd(tf.where(fg_mask), fg_labels + 1,
                             tf.cast(tf.shape(fg_mask), tf.int64))

    return tf.cond(tf.less_equal(tf.shape(centers)[0], 0), true_fun, false_fun)
