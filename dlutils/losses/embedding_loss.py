from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

from scipy.ndimage.morphology import grey_dilation


def cosine_embedding_loss(neighbor_distance=7, include_background=False):  # TODO
    '''Implementation of:

    PAYER, Christian, et al. Instance Segmentation and Tracking with
    Cosine Embeddings and Recurrent Hourglass Networks. arXiv preprint
    arXiv:1806.02070, 2018.

    Parameters
    ----------
    neighbor_distance : int or tuple of ints
        Distance threshold to consider neighboring instances

    include_background : bool
        Flag indicating whether the background should be treated as an instance

    '''

    dilation_diameter = np.asarray(neighbor_distance) * 2 + 1

    def generate_instance_mask(segmentation, label):
        '''Generate weight masks for an instance and its surrounding neighbors

        '''

        center = segmentation == label

        neighborhood = grey_dilation(center, dilation_diameter)
        neighbour_labels = np.unique(
            segmentation * neighborhood.astype(np.int))

        surround = np.zeros_like(center, dtype=np.float32)
        n_neighbour = 0
        for n_label in neighbour_labels:
            if n_label != label and (include_background or n_label != 0):
                area = (segmentation == n_label).sum()
                surround[segmentation == n_label] = 1. / area
                n_neighbour += 1

        if n_neighbour > 1:
            surround = surround / n_neighbour

        center = center.astype(np.float32)
        center /= center.sum()

        return center, surround

    def while_condition(segmentation, embeddings, labels, loss, i):
        return tf.less(i, K.shape(labels)[0])

    def while_body(segmentation, embeddings, labels, loss, i):

        center_weights, surround_weights = tf.py_func(
            generate_instance_mask, [
                segmentation, labels[i]], [
                tf.float32, tf.float32])
        center_weights.set_shape(segmentation.get_shape())
        surround_weights.set_shape(segmentation.get_shape())

        center_mean = tf.reduce_sum(
            embeddings *
            center_weights,
            axis=list(
                range(
                    K.ndim(embeddings) -
                    1)),
            keep_dims=True)
        center_mean = tf.nn.l2_normalize(center_mean, dim=-1)

        cos_similarity = tf.reduce_sum(
            embeddings * center_mean, axis=-1, keep_dims=True)

        # note: original paper mixes l1 and l2
        loss_center = 1 - K.sum(cos_similarity * center_weights)
        loss_surround = K.sum(K.abs(cos_similarity * surround_weights))
        # ~ loss_center = K.sum(K.square(1-cos_similarity) * center_weights)
        # ~ loss_surround = K.sum( K.square(cos_similarity) * surround_weights )

        loss += loss_center + loss_surround

        return [segmentation, embeddings, labels, loss, tf.add(i, 1)]

    def unbatched_loss(packed_inputs):

        segmentation = packed_inputs[0]
        embeddings = packed_inputs[1]
        embeddings = tf.nn.l2_normalize(embeddings, dim=-1)

        labels, _ = tf.unique(K.reshape(segmentation, [-1]))
        loss = tf.Variable(0.)
        i = tf.Variable(0) if include_background else tf.Variable(1)
        n_labels = K.cast(K.shape(labels)[0], K.floatx())
        if not include_background:
            n_labels -= 1

        loss = tf.while_loop(while_condition, while_body, [
                             segmentation, embeddings, labels, loss, i])[-2]

        return loss / n_labels

    def loss(y_true, y_pred):
        '''
        '''
        nonlocal dilation_diameter
        dilation_diameter = np.broadcast_to(
            dilation_diameter, K.ndim(y_true) - 2)
        dilation_diameter = tuple(dilation_diameter) + (1,)

        loss = tf.map_fn(unbatched_loss, [y_true, y_pred], tf.float32)

        return K.mean(loss)

    return loss
