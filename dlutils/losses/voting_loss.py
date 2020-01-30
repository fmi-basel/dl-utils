import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import sys


# TODO draft
# review experimental loss scaling whith actual use case
def vfield_loss(margin=0.5, spacing=1.):
    '''Displacement field regression.
    
    Notes:
    compared to simple L2, handle anisotropy and loss clipping.
    '''

    spacing = np.asarray(spacing)

    def inner_loss(y_true, y_pred):
        '''
        '''

        # extract pre-computed normalization channel
        weights = y_true[..., -1:]
        y_true = y_true[..., 0:-1]

        # for each vector, compute squared distance between predicted and target
        loss = tf.norm((y_pred - y_true) * spacing, axis=-1, keepdims=True)

        loss = K.clip(loss - margin, 0., None)

        loss = tf.math.reduce_sum(loss * weights,
                                  axis=tuple(range(1, len(y_pred.shape))))

        return tf.math.reduce_mean(loss)

    return inner_loss


# TODO rewrite as keras loss
# ~~ https://arxiv.org/pdf/1708.02551.pdf
# ~~ https://www.robots.ox.ac.uk/~vgg/publications/2018/Novotny18b/novotny2018b.pdf
# TODO if inter margin is too small, shrinking everything becomes a trivial solution to reduce loss
#  anyway to normalize things to keep the spatial relationship without cumbersome fine tuning of inter_margin?
def voting_loss(intra_margin,
                inter_margin,
                alpha=1.,
                beta=1.,
                spacing=1.,
                ignore_background=True):
    ''' Computes instance voting loss comprising the following 3 terms:
    
    - minimizes variance of relative displacement over background 
        (i.e. all background vectors point in the same 
        direction, typically zero)
        
    - minimizes variance of absolute displacement over each instance 
        (i.e. all vectors of an instance points to the same position, 
        typically close to center of mass)
        
    - forces instance's centers to be at least min_instance_dist appart
    
    The background carries the same weight as a single instance.
    
    Args: TODO
        min_instance_dist (float): minimum distance between any 2 instance's "centers"
        dist_factor (float): weights the separation loss relatively to variance losses
        spacing (float, tuple): voxel spacing
    '''
    spacing = np.asarray(spacing)
    squared_inter_margin = inter_margin**2
    variance_loss = tf.constant(0.)
    i = tf.constant(0)

    def pairwise_squared_dist(A):
        expanded_a = tf.expand_dims(A, 1)
        expanded_b = tf.expand_dims(A, 0)
        distances = tf.reduce_sum(
            tf.math.squared_difference(expanded_a, expanded_b), 2)
        return distances

    def upper_triangle(A):
        '''returns upper triangle (excluding diagonal) of a 2d matrix'''

        ones = tf.ones_like(A)
        mask_a = tf.linalg.band_part(
            ones, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(ones, 0,
                                     0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
        mask.set_shape([None, None])

        return tf.boolean_mask(A, mask)

    def separation_loss(mean_positions):
        nonlocal spacing, squared_inter_margin

        def f_separation_loss():

            # turn on loss if min inter instance dist is less than threshold
            pdist = pairwise_squared_dist(mean_positions * spacing)
            pdist = upper_triangle(pdist)

            loss_separation_dist = K.clip(squared_inter_margin - pdist, 0.,
                                          None)

            return K.mean(loss_separation_dist)

        # compute loss if more than one instance
        return tf.cond(tf.greater(K.shape(mean_positions)[0], 1),
                       f_separation_loss, lambda: 0.)

    def while_condition(labels, embeddings, unique_labels, mean_positions,
                        variance_loss, i):
        return tf.less(i, K.shape(unique_labels)[0])

    def while_body(labels, embeddings, unique_labels, mean_positions,
                   variance_loss, i):
        nonlocal spacing

        instance_mask = tf.equal(labels, unique_labels[i])
        instance_embeddings = tf.boolean_mask(embeddings, instance_mask)

        # compute mean position for inter instance variance
        mean_pos = K.mean(instance_embeddings, axis=0)
        mean_positions = mean_positions.write(i, mean_pos)

        # minimize variance of absolute displacement (relative disp for background)
        loss = tf.norm((instance_embeddings - mean_pos) * spacing, axis=-1)
        # apply margin/hinge
        loss = K.clip(loss - intra_margin, 0., None)
        loss = K.mean(K.square(loss))

        variance_loss += loss

        return (labels, embeddings, unique_labels, mean_positions,
                variance_loss, tf.add(i, 1))

    def unbatched_loss(packed_inputs):
        nonlocal spacing, variance_loss, i

        labels = packed_inputs[0][..., 0]  # remove ch dims
        embeddings = packed_inputs[1]

        unique_labels, _ = tf.unique(K.reshape(labels, [-1]))
        # ignore negative labels (unkown) and possibly bg (==0)
        unique_labels = tf.boolean_mask(
            unique_labels, tf.greater_equal(unique_labels, ignore_background))
        n_labels = K.cast(K.shape(unique_labels)[0], K.floatx())

        # initialize loss
        variance_loss = 0.  # = tf.Variable(0.)

        # setup counter for 'for' loop made with tf.while_loop
        i = 0

        # setup array to collect mean position of each instance
        mean_positions = tf.TensorArray(size=K.shape(unique_labels)[0],
                                        dtype=tf.float32,
                                        infer_shape=False)

        # main loop: iterate through instances (including background) to compute variance loss + mean position
        _, _, _, mean_positions, variance_loss, _ = tf.while_loop(
            while_condition,
            while_body, [
                labels, embeddings, unique_labels, mean_positions,
                variance_loss, i
            ],
            parallel_iterations=1)

        # convert tensor array to tensor and discard background average position
        def sep_loss():
            nonlocal mean_positions

            mean_positions = mean_positions.stack()
            mean_positions = tf.boolean_mask(mean_positions,
                                             tf.greater(unique_labels, 0))
            s_loss = separation_loss(mean_positions)
            return s_loss

        s_loss = tf.cond(tf.greater(n_labels, 0), sep_loss, lambda: 0.)

        return alpha * (variance_loss / (n_labels + 1e-12)) + beta * s_loss

    def get_px_coords(vfield):
        '''Returns the coordinates of each pixel. Ignores first dim (batch), and last dim (ch)'''

        # TODO how to write a general solution for non-eager execution?
        # ranges = [tf.range(s, dtype=tf.float32) for s in tf.shape(vfield)[1:-1]] #--> only works in eager execution
        # tf.map_fn requires all outputs to have the same size

        s = tf.shape(vfield)
        if len(vfield.shape) == 4:
            ranges = tf.range(s[1],
                              dtype=tf.float32), tf.range(s[2],
                                                          dtype=tf.float32)
        elif len(vfield.shape) == 5:
            ranges = (tf.range(s[1], dtype=tf.float32),
                      tf.range(s[2], dtype=tf.float32),
                      tf.range(s[3], dtype=tf.float32))
        elif len(vfield.shape) == 6:
            ranges = (tf.range(s[1], dtype=tf.float32),
                      tf.range(s[2], dtype=tf.float32),
                      tf.range(s[3], dtype=tf.float32),
                      tf.range(s[4], dtype=tf.float32))
        else:
            raise NotImplementedError(
                'get_px_coords not implemented for {} dimensions'.format(
                    len(vfield.shape) - 2))

        coords = tf.meshgrid(*ranges, indexing='ij')
        return tf.stack(coords, axis=-1)

    def loss(y_true, y_pred, sample_weight=None):
        '''
        y_true: instance labels (0=background, -1 ignored)
        y_pred: vector field
        '''
        y_true = tf.cast(y_true, tf.int32)

        coords = get_px_coords(y_pred)

        if ignore_background:
            embeddings = y_pred + coords
        else:
            # add coords to predicted vfield, expect where the is background
            embeddings = y_pred + coords * tf.cast(tf.greater(y_true, 0),
                                                   tf.float32)

        loss = tf.map_fn(unbatched_loss, [y_true, embeddings],
                         tf.float32,
                         parallel_iterations=16)

        return K.mean(loss)

    return loss


def reshape_pseudo_dim(t, n_channels):
    '''Reshapes tensor t by spltting the last dim into (n_channels_original/n_channels, n_channels). '''

    orig_shape = K.shape(t)
    new_shape = tf.concat([
        orig_shape[:-1], orig_shape[-1:] // n_channels,
        tf.Variable([n_channels])
    ],
                          axis=0)

    return tf.reshape(t, new_shape)


# ########################################################################
# # TODO try with separate dim (like above) instead of norm
# def voting_loss_pseudo4D(min_instance_dist, dist_factor=0.05, spacing=1., instance_loss='variance'):
# '''similar to voting loss applied between 2 frames, except that
# instance vectors at time t should point to mean of vectors at
# time t+1. and inversely i.e. no spatial intra-instance constraints.'''

# spacing = np.asarray(spacing)
# squared_min_instance_dist = min_instance_dist**2

# def pairwise_squared_dist(A):
# expanded_a = tf.expand_dims(A, 1)
# expanded_b = tf.expand_dims(A, 0)
# distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
# return distances

# def upper_triangle(A):
# '''returns upper triangle (excluding diagonal) of a 2d matrix'''

# ones = tf.ones_like(A)
# mask_a = tf.linalg.band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
# mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
# mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask
# mask.set_shape([None,None])

# return tf.boolean_mask(A, mask)

# def separation_loss(mean_positions):
# nonlocal spacing, squared_min_instance_dist

# # convert tensor array to tensor and
# # discard background average position (assumes a background==0 label always exist)
# mean_positions = mean_positions.stack()[1:]

# def f_separation_loss():

# # turn on loss if min inter instance dist is less than threshold
# pdist = pairwise_squared_dist(mean_positions*spacing)
# pdist = upper_triangle(pdist)

# loss_separation_dist = K.clip(squared_min_instance_dist-pdist, 0., None)

# return K.sum(loss_separation_dist)

# # compute loss if more than one instance
# return tf.cond(tf.greater(K.shape(mean_positions)[0],1), f_separation_loss, lambda: 0.)

# def tf_sort(t):
# '''sort tensor values in increasing order'''

# # sort (looks like tf.sort was introduced only in tensorflow 1.14)
# t = tf.nn.top_k(t, K.shape(t)[0])[0][::-1]

# return t

# def while_condition(labels, vfield, unique_labels, mean_positions, variance_loss, i):
# return tf.less(i, K.shape(unique_labels)[0])

# def while_body(labels, vfield, unique_labels, mean_positions, variance_loss, i):
# nonlocal spacing

# instance_mask = tf.equal(labels, unique_labels[i])

# instance_coords = tf.where(instance_mask[...,0])
# instance_vfield = tf.gather_nd(vfield, instance_coords)

# instance_coords = tf.cast(instance_coords, tf.float32)
# instance_embeddings = instance_coords + instance_vfield

# # if bg label, minimize variance of relative displacement
# def f_bg():
# return K.mean(K.var(instance_vfield * spacing, axis=0))

# # if instance label, minimize variance of absolute displacement
# def f_instance():
# # ~ if instance_loss == 'variance':
# # ~ return K.mean(K.var(instance_embeddings * spacing, axis=0))

# # ~ elif instance_loss == 'center_mass':
# # ~ # TODO unittest
# # ~ cm = K.mean(instance_coords, axis=0)
# # ~ return K.mean(tf.norm( (instance_embeddings - cm) * spacing))

# instance_embeddings_t = tf.boolean_mask(instance_embeddings, tf.less(instance_coords[:,-1],1))
# instance_embeddings_tp = tf.boolean_mask(instance_embeddings, tf.greater_equal(instance_coords[:,-1],1))

# def loss_intra():

# loss_t = K.mean(tf.norm( (instance_embeddings_t - K.mean(instance_embeddings_tp, axis=0)) * spacing))
# loss_tp = K.mean(tf.norm( (instance_embeddings_tp - K.mean(instance_embeddings_t, axis=0)) * spacing))

# return K.mean(loss_t + loss_tp)

# return tf.cond(tf.logical_and(tf.greater(K.shape(instance_embeddings_t)[0],0),
# tf.greater(K.shape(instance_embeddings_tp)[0],0)),
# loss_intra,
# lambda: 0.)

# # ~ else:
# # ~ raise ValueError('{} is not a valid instance_loss option').format(instance_loss)

# variance_loss += tf.cond(tf.equal(unique_labels[i],0), f_bg, f_instance)

# # compute mean position for inter instance variance
# mean_pos =  K.mean(instance_embeddings, axis=0)
# mean_positions = mean_positions.write(i, mean_pos)

# return (labels, vfield, unique_labels, mean_positions, variance_loss,  tf.add(i, 1))

# def unbatched_loss(packed_inputs):
# nonlocal spacing, dist_factor

# labels = packed_inputs[0]
# vfield = packed_inputs[1]

# unique_labels,_ = tf.unique(K.reshape(labels, [-1]))
# unique_labels = tf_sort(unique_labels)
# # remove "unkown" regions (label == -1) if present
# unique_labels = tf.cond(tf.less(unique_labels[0], 0), lambda: unique_labels[1:], lambda: unique_labels)

# n_labels = K.cast(K.shape(unique_labels)[0], K.floatx())

# # initialize loss
# variance_loss = tf.constant(0.)

# # setup counter for 'for' loop made with tf.while_loop
# i = tf.constant(0)

# # setup array to collect mean position of each instance
# mean_positions = tf.TensorArray(size=K.shape(unique_labels)[0], dtype=tf.float32, infer_shape=False)

# # main loop: iterate through instances (including background) to compute variance loss + mean position
# _,_,_,mean_positions,variance_loss,_ = tf.while_loop(while_condition, while_body,
# [labels, vfield, unique_labels, mean_positions, variance_loss, i])

# s_loss = separation_loss(mean_positions)

# return (variance_loss / (n_labels + 1e-12)) + dist_factor * s_loss

# def loss(y_true, y_pred, sample_weight=None):
# '''
# y_true: instance labels (0=background, -1 ignored)
# y_pred: vector field
# '''

# # reshape pseudo 4D to 4D batch, z, x, y, time, channel
# y_true = reshape_pseudo_dim(y_true, 1)
# y_pred = reshape_pseudo_dim(y_pred, 4)

# loss = tf.map_fn(unbatched_loss, [y_true, y_pred], tf.float32)

# return  K.mean(loss)

# return loss

########################################################################

# ~ def voting_loss_pseudo4D(min_instance_dist, dist_factor=1., spacing=1.):
# ~ '''similar to voting loss applied between 2 frames, except that
# ~ instance vectors at time t should point to mean of vectors at
# ~ time t+1. and inversely i.e. no spatial intra-instance constraints.'''

# ~ spacing = np.asarray(spacing)
# ~ squared_min_instance_dist = min_instance_dist**2

# ~ def pairwise_squared_dist(A):
# ~ expanded_a = tf.expand_dims(A, 1)
# ~ expanded_b = tf.expand_dims(A, 0)
# ~ distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
# ~ return distances

# ~ def get_label_mask(segmentation, label, labels):
# ~ '''Generate weight masks for an instance and its surrounding neighbors

# ~ '''

# ~ # TODO figure equivalent tensorflow fct
# ~ return (segmentation == label,)

# ~ def debug_print(pred_positions_t, pred_positions_tp, t_mean, tp_mean):

# ~ print(pred_positions_t.shape, pred_positions_tp.shape, t_mean, tp_mean)

# ~ return (np.array([0.,], dtype=np.float32), )

# ~ def while_condition(segmentation, lapl, labels, n_labels, mean_positions, loss, i):
# ~ return tf.less(i, K.shape(labels)[0])

# ~ def while_body(segmentation, y_pred, labels, n_labels, mean_positions, loss, i):
# ~ nonlocal spacing

# ~ mask = tf.py_func(get_label_mask, [segmentation, labels[i], labels], [tf.bool])[0]
# ~ mask.set_shape(segmentation.get_shape())

# ~ # for pseudo 4d reshaped outputs
# ~ img_coords = tf.where(mask)
# ~ pred_displacement = tf.gather_nd(y_pred, img_coords)
# ~ img_coords_float = tf.cast(img_coords, tf.float32)
# ~ pred_positions = img_coords_float[:,:-1] + pred_displacement

# ~ # if bg label, minimize variance of relative displacement
# ~ def f_bg():
# ~ loss = K.mean(K.var(pred_displacement, axis=0) * spacing)

# ~ ####### clipped version
# ~ # ~loss =  K.clip(loss - (squared_min_instance_dist/4.), 0., None)
# ~ return loss

# ~ # give equal weight to bg vs mean instance loss?
# ~ #             return K.mean(K.var(pred_displacement, axis=0) * spacing) * (n_labels - 1)

# ~ # if instance, minimize variance of absolute displacement
# ~ def f_instance():

# ~ pred_positions_t = tf.boolean_mask(pred_positions, tf.equal(img_coords[:,-1],0))
# ~ pred_positions_tp = tf.boolean_mask(pred_positions, tf.equal(img_coords[:,-1],1))

# ~ # ~fake_loss =  tf.py_func(debug_print, [pred_positions_t, pred_positions_tp, t_mean, tp_mean], [tf.float32])[0]
# ~ # ~fake_loss.set_shape(loss.get_shape())

# ~ # ~return fake_loss

# ~ def loss_intra():

# ~ t_mean = K.mean(pred_positions_t, axis=0)
# ~ tp_mean = K.mean(pred_positions_tp, axis=0)
# ~ return K.mean(K.square(pred_positions_t-tp_mean) * spacing) + K.mean(K.square(pred_positions_tp-t_mean) * spacing)

# ~ return tf.cond(tf.logical_and(tf.greater(K.shape(pred_positions_t)[0],0),
# ~ tf.greater(K.shape(pred_positions_tp)[0],0)),
# ~ loss_intra,
# ~ lambda: 0.)

# ~ # ~loss_abs_var = K.mean(K.var(pred_positions, axis=0) * spacing)

# ~ ####### clipped version
# ~ # ~loss_abs_var = K.clip(loss_abs_var - (squared_min_instance_dist/4.), 0., None)

# ~ # ~return loss_abs_var #+ mean_reg *loss_rel_mean

# ~ loss += tf.cond(tf.equal(labels[i],0), f_bg, f_instance) #+ fake_loss

# ~ # compute mean position for inter instance variance
# ~ mean_pos =  K.mean(pred_positions, axis=0)
# ~ mean_positions = mean_positions.write(i, mean_pos)

# ~ return (segmentation, y_pred, labels, n_labels, mean_positions, loss,  tf.add(i, 1))

# ~ def unbatched_loss(packed_inputs):

# ~ nonlocal spacing, squared_min_instance_dist, dist_factor

# ~ segmentation = packed_inputs[0]
# ~ y_pred = packed_inputs[1]

# ~ labels, _ = tf.unique(K.reshape(segmentation, [-1]))
# ~ # sort (looks like tf.sort introduced only in tensorflow 1.14)
# ~ labels= tf.nn.top_k(labels, K.shape(labels)[0])[0][::-1]

# ~ # remove "unkown" regions (label -1) if present
# ~ labels = tf.cond(tf.less(labels[0],-0.5), lambda: labels[1:], lambda: labels)

# ~ # initialize loss
# ~ loss = tf.Variable(0.)

# ~ # setup counter for 'for' loop made with tf.while_loop
# ~ # include background = 0
# ~ i = tf.Variable(0)
# ~ n_labels = K.cast(K.shape(labels)[0], K.floatx())

# ~ # setup array to collect mean position of each instance
# ~ mean_positions = tf.TensorArray(size=K.shape(labels)[0], dtype=tf.float32, element_shape=[y_pred.shape[0]], infer_shape=False)

# ~ mean_positions, loss = tf.while_loop(while_condition, while_body, [segmentation, y_pred, labels, n_labels, mean_positions, loss, i])[-3:-1]

# ~ # ingore background average position
# ~ mean_positions = mean_positions.stack()[1:]

# ~ def f_inter():
# ~ # turn on loss if min inter instance dist is less than threshold
# ~ pdist = pairwise_squared_dist(mean_positions*spacing)
# ~ # keep upper triangle
# ~ pdist = tf.matrix_band_part(pdist, 0, -1)
# ~ loss_inter_dist = K.clip(squared_min_instance_dist-pdist, 0., None)

# ~ return K.mean(loss_inter_dist)

# ~ def f_inter_cond():
# ~ # and if the lowest of the 2 labels is greater than 0
# ~ # (Note, cannot be combined with test below because second condition in  tf.logical_and seems to always be evaluated)
# ~ return tf.cond(tf.greater(labels[-2],0.5), f_inter, lambda: 0.)

# ~ # if there is at least 2 labels
# ~ inter_loss = tf.cond(tf.greater(n_labels,2.5), f_inter_cond, lambda: 0.)

# ~ loss = (loss / (n_labels + 1e-12)) + dist_factor * inter_loss

# ~ # check that there was at least one label (>-1) in patch
# ~ loss = tf.cond(tf.greater(n_labels,0.5), lambda: loss, lambda: 0.)

# ~ return loss

# ~ def loss(y_true, y_pred):
# ~ '''
# ~ '''

# ~ # reshape output as if there was one extra dimension: batch,z,x,y,ch --> batch,z,x,y,t,3
# ~ #y_pred = K.reshape(y_pred, (None, None, None, None, -1, 3) )
# ~ y_pred = tf.unstack(y_pred ,axis=-1,)
# ~ # todo generalize to x timepoints
# ~ a = tf.stack(y_pred[0:2] ,axis=-1)
# ~ b = tf.stack(y_pred[2:4] ,axis=-1)
# ~ c = tf.stack(y_pred[4:6] ,axis=-1)
# ~ #d = tf.stack(y_pred[6:8] ,axis=-1)
# ~ y_pred = tf.stack([a,b,c], axis=-1)

# ~ loss = tf.map_fn(unbatched_loss, [y_true, y_pred], tf.float32)

# ~ # TODO 2D/3D support (mean_positions)

# ~ return  K.mean(loss)

# ~ return loss

# ~ ########################################################################
