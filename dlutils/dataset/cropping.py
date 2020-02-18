'''tf.data pipeline compatible cropping.

'''
import tensorflow as tf


def random_crop(patch_size):
    '''returns the patch sampling function that takes the same
    random patch for all tensors in a given dictionary.

    If you just want to take a random patch from a *single*
    tensor, you should probably use tensorflow.image.random_crop

    Parameters
    ----------
    patch_size : tuple
        patch_size to sample. Needs to have the same length as the shape
        of the future input tensors. If a dimension should not be sampled,
        set the corresponding entry in patch_size to -1.

    '''

    # patch size as tensor. Needed in _shape_to_compare and _cropper.
    patch_size = tf.convert_to_tensor(patch_size, name='patch_size')

    def _shape_to_compare(shape):
        '''remove dimensions from shape where the patch_size is negative.

        '''
        tf.assert_equal(
            tf.size(shape),
            tf.size(patch_size),
            message=
            'Input shape and patch_size must have the same length. Got input_shape={} vs patch_size={}'
            .format(shape, patch_size))
        return tf.boolean_mask(shape, patch_size >= 0)

    def _cropper(inputs):
        '''expects a dictionary of tensors as inputs.

        '''
        with tf.name_scope('random_patch'):
            if isinstance(inputs, dict):
                iterable = list(inputs.values())
            elif isinstance(inputs, (tuple, list)):
                iterable = inputs
            else:
                raise NotImplementedError('Input must be dict, tuple or list!')

            input_shape = tf.shape(iterable[0])

            # check for compatible patch_size and image shapes
            tf.debugging.assert_greater_equal(
                input_shape,
                patch_size,
                message=
                'Expected inputs to be larger than patch_size. Got input_shape={} vs patch_size={}'
                .format(input_shape, patch_size))

            for val in iterable[1:]:
                message = 'Expected inputs to have the same size. Got {} vs {}'.format(
                    input_shape, tf.shape(val))
                tf.assert_equal(_shape_to_compare(input_shape),
                                _shape_to_compare(tf.shape(val)),
                                message=message)

            # actual patch sampling
            limit = input_shape - patch_size + 1
            offset = tf.random.uniform(
                tf.shape(input_shape),
                dtype=tf.int32,
                maxval=tf.int32.max,
            ) % limit
            offset = tf.where(patch_size <= -1, 0, offset)

            if isinstance(inputs, dict):
                return {
                    key: tf.slice(value, offset, patch_size)
                    for key, value in inputs.items()
                }
            return tuple(
                tf.slice(value, offset, patch_size) for value in iterable)

    return _cropper
