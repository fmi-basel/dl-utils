import tensorflow as tf


class JaccardLoss(tf.keras.losses.Loss):
    '''
    Differentiable Jaccard/mIoU loss as proposed in:
    
    Rahman, Md Atiqur, and Yang Wang. "Optimizing intersection-over-union 
    in deep neural networks for image segmentation." 
    International symposium on visual computing. Springer, Cham, 2016.
    '''
    def __init__(self, eps=1e-6):
        super().__init__()

        self.eps = eps

    def call(self, y_true, y_pred):
        '''
        Args:
        
        y_true: 1 hot masks
        y_pred: probability maps between [0,1] for each label
        '''

        spatial_axis = tuple(range(1, len(y_true.shape) - 1))

        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection
        jaccard = 1. - (intersection + self.eps) / (union + self.eps)

        return tf.math.reduce_mean(jaccard)
