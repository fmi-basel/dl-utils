import abc
import numpy as np
import tensorflow as tf


def _unbatched_soft_jaccard(y_true, y_pred, fg_only=True, eps=1e-6):
    '''expects y_true as one-hot and y_pred as probabilities between [0, 1]

    '''
    spatial_axis = tuple(range(len(y_true.shape) - 1))
    intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)

    if fg_only:
        fg_mask = tf.cast(
            tf.reduce_any(tf.greater_equal(y_true, 0.5),
                          axis=-1,
                          keepdims=True), tf.float32)
        union = tf.reduce_sum(fg_mask * (y_pred + y_true),
                              axis=spatial_axis) - intersection
    else:
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection

    jaccard = 1 - (intersection + eps) / (union + eps)

    return jaccard


def _unbatched_label_to_hot(instance_labels):
    '''
    Generates 1-hot encoding of instance labels.
    
    Notes:
    ignores negative labels and background=0
    '''

    spatial_axis = tuple(range(len(instance_labels.shape) - 1))

    # remove background for one-hot by making it negative.
    instance_labels = instance_labels - 1
    n_classes = tf.maximum(0, tf.reduce_max(instance_labels)) + 1

    hot = tf.one_hot(tf.squeeze(instance_labels, -1), n_classes)

    return hot


def _unbatched_embedding_center(hot, y_pred):
    '''Returns the mean of  each embedding under the true instance mask'''

    spatial_axis = tuple(range(len(hot.shape) - 1))

    # mean embedding under the true instance mask
    counts = tf.expand_dims(tf.reduce_sum(hot, axis=spatial_axis), -1)
    y_pred = tf.expand_dims(y_pred, -2)
    centers = tf.reduce_sum(
        (tf.expand_dims(hot, -1) * y_pred), axis=spatial_axis,
        keepdims=True) / counts

    return centers


def _unbatched_embeddings_to_center_dist(embeddings, centers):
    '''Returns a embedding distance map per center'''

    # add 1hot dimension to embeddings
    embeddings = tf.expand_dims(embeddings, -2)

    # add spatial dimensions to centers if necessary
    while (len(centers.shape) < len(embeddings.shape)):
        centers = tf.expand_dims(centers, 0)

    return tf.norm(centers - embeddings, axis=-1)


def _unbatched_embeddings_to_prob(embeddings, centers, margin,
                                  clip_probs=None):
    '''
    Converts embeddings to probability maps by passing their distances 
    from the given centers through a gaussian function:
    
    p(e_i) = exp(-2 * (norm(e_i-center)/sigma)**2)
    
    where: margin = sigma * sqrt(-2 * ln(0.5))
    
    i.e. embeddings further than margin away from a center have a probability < 0.5
    
    Args:
        embeddings: [spacial...dims, embedding_size]
        centers: [spacial...dims, npoint, embedding_size] or 
            [npoint, embedding_size]
        margin: distance from center where instance probability = 0.5
        clip_probs: clips probabilities values if a (low,high) tuple is provided
        
    Notes:
    
    For more details see
    
    Neven, Davy, et al. "Instance segmentation by jointly optimizing spatial
    embeddings and clustering bandwidth." Proceedings of the IEEE Conference
    on Computer Vision and Pattern Recognition. 2019.
    '''
    def calc_probs(center_distances):
        sigma = margin * (-2 * np.log(0.5))**-0.5
        probs = tf.exp(-0.5 * (center_distances / sigma)**2)

        if clip_probs is None:
            return probs
        elif isinstance(clip_probs, tuple) and len(clip_probs) == 2:
            return tf.clip_by_value(probs, *clip_probs)
        else:
            raise ValueError(
                'clip_probs should be None or (low,high) tuple: . got {}'.
                format(clip_probs))

    center_dist = _unbatched_embeddings_to_center_dist(embeddings, centers)

    # convert distance from center to probability of belonging to the instance
    return calc_probs(center_dist)


def relabel_sequential(labels):
    '''Relabel positive (foreground) labels sequentially'''
    fg_mask = labels > 0
    _, seq_labels = tf.unique(tf.boolean_mask(labels, fg_mask))

    return tf.tensor_scatter_nd_update(labels, tf.where(fg_mask),
                                       seq_labels + 1)


class InstanceEmbeddingLossBase(tf.keras.losses.Loss):
    '''Base class for embedding losses.

    '''
    def __init__(self, parallel_iterations=4, *args, **kwargs):
        '''
        '''
        super().__init__(*args, **kwargs)
        self.parallel_iterations = parallel_iterations

    def call(self, y_true, y_pred):
        '''
        '''
        y_true = tf.cast(y_true, tf.int32)

        # remove batch item that have no groundtruth at all
        has_instance_mask = tf.reduce_any(y_true > 0,
                                          axis=tuple(
                                              range(1, len(y_true.shape))))

        def map_to_not_empty():
            y_true_masked = tf.boolean_mask(y_true, has_instance_mask, axis=0)
            y_pred_masked = tf.boolean_mask(y_pred, has_instance_mask, axis=0)

            loss = tf.map_fn(self._unbatched_loss,
                             [y_true_masked, y_pred_masked],
                             tf.float32,
                             parallel_iterations=self.parallel_iterations)

            return tf.reduce_mean(loss)

        return tf.cond(tf.reduce_any(has_instance_mask), map_to_not_empty,
                       lambda: 0.)

    @abc.abstractmethod
    def _unbatched_loss(self, packed):
        '''
        '''
        pass


class InstanceMeanIoUEmbeddingLoss(InstanceEmbeddingLossBase):
    '''
    
    Neven, D., Brabandere, B.D., Proesmans, M. and Gool, L.V., 2019. 
    Instance segmentation by jointly optimizing spatial embeddings and 
    clustering bandwidth. In Proceedings of the IEEE Conference on 
    Computer Vision and Pattern Recognition (pp. 8837-8845).
    '''
    def __init__(self, margin, clip_probs=None, *args, **kwargs):
        '''
        
        Args:
            margin: distance from center where instance probability = 0.5
        '''
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.clip_probs = clip_probs

        assert self.margin > 0.

    def _unbatched_loss(self, packed):
        '''
        '''

        y_true, y_pred = packed
        y_true = relabel_sequential(y_true)  # on random patch level

        one_hot = _unbatched_label_to_hot(y_true)
        centers = _unbatched_embedding_center(one_hot, y_pred)
        probs = _unbatched_embeddings_to_prob(y_pred, centers, self.margin,
                                              self.clip_probs)

        return tf.reduce_mean(_unbatched_soft_jaccard(one_hot, probs))


class MarginInstanceEmbeddingLoss(InstanceEmbeddingLossBase):
    '''Same as InstanceMeanIoUEmbeddingLoss except embeddings are 
    converted to a probability map with intra/inter margins "hinges" 
    instead of a gaussian kernel.
    
    Similar to:
    De Brabandere, B., Neven, D. and Van Gool, L., 2017. Semantic 
    instance segmentation with a discriminative loss function. arXiv 
    preprint arXiv:1708.02551.
    
    '''
    def __init__(self, intra_margin, inter_margin, *args, **kwargs):
        '''
        
        Args:
            intra_margin: distance from an embedding to its center below which the loss is zero
            inter_margin: distance from an embedding to other centers above which the loss is zero
        '''
        super().__init__(*args, **kwargs)

        self.intra_margin = intra_margin
        self.inter_margin = inter_margin

        assert self.intra_margin > 0.
        assert self.inter_margin > self.intra_margin

    def _hingify(self, y_true, rescaled_center_dist):
        '''Replaces pixel probs over hinge threshold by groundtruth value'''

        rescaled_center_dist = tf.where(
            ~tf.cast(y_true, tf.bool) & (rescaled_center_dist < 0), 0.,
            rescaled_center_dist)
        rescaled_center_dist = tf.where(
            tf.cast(y_true, tf.bool) & (rescaled_center_dist > 1.0), 1.,
            rescaled_center_dist)

        return rescaled_center_dist

    def _rescale_distance(self, center_dist):
        '''Applies f(d) linear transform to distance d so that f(intra_margin)==1 and f(inter_margin)==0'''

        a = 1 / (self.intra_margin - self.inter_margin)
        b = self.inter_margin / (self.inter_margin - self.intra_margin)

        return a * center_dist + b

    def _unbatched_loss(self, packed):
        '''
        '''

        y_true, y_pred = packed
        y_true = relabel_sequential(y_true)  # on random patch level

        one_hot = _unbatched_label_to_hot(y_true)
        centers = _unbatched_embedding_center(one_hot, y_pred)

        center_dist = _unbatched_embeddings_to_center_dist(y_pred, centers)
        rescaled_center_dist = self._rescale_distance(center_dist)
        probs = self._hingify(one_hot, rescaled_center_dist)

        return tf.reduce_mean(_unbatched_soft_jaccard(one_hot, probs, eps=1))
