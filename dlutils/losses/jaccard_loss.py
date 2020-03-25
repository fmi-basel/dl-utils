import tensorflow as tf


class JaccardLoss(tf.keras.losses.Loss):
    '''
    Differentiable Jaccard/mIoU loss as proposed in:
    
    Rahman, Md Atiqur, and Yang Wang. "Optimizing intersection-over-union 
    in deep neural networks for image segmentation." 
    International symposium on visual computing. Springer, Cham, 2016.
    
    Args:
    eps: epsilon to avoid divison by zero.
    hinge_probs: None or tuple(low, high) Tresholds over which pixels in prediction
        are replaced by the groundtruth (i.e. no backpropagation)
    '''
    def __init__(self, eps=1e-6, hinge_probs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps
        self.hinge_probs = hinge_probs

    def _remove_unannot(self, y_true, y_pred):
        annot_mask = tf.cast(
            tf.reduce_any(tf.greater_equal(y_true, 0.5),
                          axis=-1,
                          keepdims=True), tf.float32)

        return y_true, y_pred * annot_mask

    def call(self, y_true, y_pred):
        '''
        Args:
        
        y_true: 1 hot masks
        y_pred: probability maps between [0,1] for each label
        '''

        y_true, y_pred = self._remove_unannot(y_true, y_pred)

        if self.hinge_probs is not None:
            # replace px over hinge threshold by groundtruth value
            y_pred = tf.where(
                ~tf.cast(y_true, tf.bool) & (y_pred < self.hinge_probs[0]), 0.,
                y_pred)
            y_pred = tf.where(
                tf.cast(y_true, tf.bool) & (y_pred > self.hinge_probs[1]), 1.,
                y_pred)

        spatial_axis = tuple(range(1, len(y_true.shape) - 1))

        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection
        jaccard = 1. - (intersection + self.eps) / (union + self.eps)

        return tf.math.reduce_mean(jaccard)


class BinaryJaccardLoss(JaccardLoss):
    '''
    Single class variant of JaccardLoss.
    
    if y_true has 2 channels, the second channel is considered as annotation mask
    '''
    def _remove_unannot(self, y_true, y_pred):

        annot_mask = y_true >= 0
        y_true = tf.clip_by_value(y_true, 0., 1.)
        y_pred = y_pred * tf.cast(annot_mask, tf.float32)

        return y_true, y_pred
