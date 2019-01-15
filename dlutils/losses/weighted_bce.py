import tensorflow as tf
from keras.backend.tensorflow_backend import _to_tensor
from keras.backend.common import epsilon
from keras.backend import mean


def weighted_binary_crossentropy(pos_weight):
    '''Weighted binary crossentropy between an output tensor and a target tensor.

    # Arguments
        pos_weight: float weighting the positive class.

    '''

    def _wbce(target, output, from_logits=False):
        '''
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
        # Returns
            A tensor.
        '''

        # Note: tf.nn.weighted_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))

        return mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=target, logits=output, pos_weight=pos_weight),
            axis=-1)

    return _wbce
