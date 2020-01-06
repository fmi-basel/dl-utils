from tensorflow.keras import backend as K
import tensorflow as tf
import abc


class PixelWeightedLossBase(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        '''Computes pixel weighted loss, using the last channel of y_true as 
        normalization weights
    
        Notes:
        ------
        - sum of weights == 1 is expected
        - implemented as channels_last
        '''

        # extract pre-computed normalization channel
        self._check_weights_stacking(y_true, y_pred)
        weights = y_true[..., -1:]
        y_true = y_true[..., 0:-1]

        loss = self._pixel_loss(y_true, y_pred)

        # sum loss over spatial and channel dims, mean over batch
        loss = tf.math.reduce_sum(loss * weights,
                                  axis=tuple(range(1, len(y_pred.shape))))
        return tf.math.reduce_mean(loss)

    def _check_weights_stacking(self, y_true, y_pred):
        if y_true.shape[-1] != y_pred.shape[-1] + 1:
            raise ValueError(
                'Weights incorrectly stacked. Expected y_true with an extra channel for the weights got y_true, y_pred shapes: {} and {}'
                .format(y_true.shape, y_pred.shape))

    @abc.abstractmethod
    def _pixel_loss(self, y_true, y_pred):
        pass


class PixelWeightedL1Loss(PixelWeightedLossBase):
    def _pixel_loss(self, y_true, y_pred):
        return K.abs(y_pred - y_true)


class PixelWeightedMSE(PixelWeightedLossBase):
    def _pixel_loss(self, y_true, y_pred):
        return K.square(y_pred - y_true)


class PixelWeightedBinaryCrossentropy(PixelWeightedLossBase):
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def _pixel_loss(self, y_true, y_pred):
        return K.binary_crossentropy(y_true,
                                     y_pred,
                                     from_logits=self.from_logits)
