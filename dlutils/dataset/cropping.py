'''tf.data pipeline compatible cropping.

'''
import tensorflow as tf


def random_crop(patch_size):
    '''returns the patch sampling function that takes the same
    random patch for all tensors in a given dictionary.

    If you just want to take a random patch from a *single*
    tensor, you should probably use tensorflow.image.random_crop

    '''

    # NOTE it's not clear whether marking this with @tf.function would
    # be more efficient (and safe).

    def _cropper(inputs):
        '''expects a dictionary of tensors as inputs.

        '''
        if isinstance(inputs, dict):
            iterable = list(inputs.values())
        elif isinstance(inputs, (tuple, list)):
            iterable = inputs
        else:
            raise NotImplementedError('Input must be dict, tuple or list!')

        input_shape = tf.shape(iterable[0])

        # NOTE shape checks fail with @tf.function.
        # for val in iterable[1:]:
        #     other_shape = tf.shape(val)
        #     # yapf: disable
        #     if (len(input_shape) != len(other_shape)
        #         or not all(x == y or ps == -1
        #                    for x, y, ps in zip(input_shape, other_shape, patch_size))):
        #         raise ValueError(
        #             'Got inputs of shape {} and {}. All inputs must have the same shape!'.
        #             format(input_shape, other_shape))
        #     # yapf: enable

        size = tf.convert_to_tensor(patch_size, name='patch_size')

        limit = input_shape - size + 1
        offset = tf.random.uniform(
            tf.shape(input_shape),
            dtype=tf.int32,
            maxval=tf.int32.max,
        ) % limit
        offset = tf.where(size <= -1, 0, offset)

        if isinstance(inputs, dict):
            return {
                key: tf.slice(value, offset, size)
                for key, value in inputs.items()
            }
        elif isinstance(inputs, (tuple, list)):
            return tuple(tf.slice(value, offset, size) for value in iterable)

    return _cropper
