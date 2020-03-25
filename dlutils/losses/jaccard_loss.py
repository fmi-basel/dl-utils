import tensorflow as tf

# TODO separate multi-class and binary cases (needed to handle unannotated labels)


class BinaryJaccardLoss(tf.keras.losses.Loss):
    '''
    Differentiable Jaccard/mIoU loss as proposed in:
    
    Rahman, Md Atiqur, and Yang Wang. "Optimizing intersection-over-union 
    in deep neural networks for image segmentation." 
    International symposium on visual computing. Springer, Cham, 2016.
    
    if y_true has 2 channels, the second channel is considered annotation mask
    '''
    def __init__(self, eps=1e-6, clip_probs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps
        self.clip_probs = clip_probs

    def call(self, y_true, y_pred):
        '''
        Args:
        
        y_true: 1 hot masks
        y_pred: probability maps between [0,1] for each label
        '''

        if y_true.shape[-1] > 1:
            fg_mask = tf.cast(y_true[..., 1:2], tf.float32)
            y_true = y_true[..., 0:1]

            y_pred = y_pred * fg_mask

        if self.clip_probs is not None:
            y_pred = tf.clip_by_value(y_pred, *self.clip_probs)

        spatial_axis = tuple(range(1, len(y_true.shape) - 1))

        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection
        jaccard = 1. - (intersection + self.eps) / (union + self.eps)

        return tf.math.reduce_mean(jaccard)


class JaccardLoss(tf.keras.losses.Loss):
    '''
    Differentiable Jaccard/mIoU loss as proposed in:
    
    Rahman, Md Atiqur, and Yang Wang. "Optimizing intersection-over-union 
    in deep neural networks for image segmentation." 
    International symposium on visual computing. Springer, Cham, 2016.
    '''
    def __init__(self,
                 eps=1e-6,
                 clip_probs=None,
                 fg_only=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps
        self.clip_probs = clip_probs
        self.fg_only = fg_only

    def call(self, y_true, y_pred):
        '''
        Args:
        
        y_true: 1 hot masks
        y_pred: probability maps between [0,1] for each label
        '''

        if self.fg_only:
            fg_mask = tf.cast(
                tf.reduce_any(tf.greater_equal(y_true, 0.5),
                              axis=-1,
                              keepdims=True), tf.float32)

            y_pred = y_pred * fg_mask

        if self.clip_probs is not None:
            # ~y_pred = tf.clip_by_value(y_pred, *self.clip_probs)

            # clip where overconfident
            y_pred = tf.where(tf.cast(y_true, tf.bool),
                              tf.minimum(y_pred, self.clip_probs[1]),
                              tf.maximum(y_pred, self.clip_probs[0]))

        spatial_axis = tuple(range(1, len(y_true.shape) - 1))

        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection
        jaccard = 1. - (intersection + self.eps) / (union + self.eps)

        return tf.math.reduce_mean(jaccard)
