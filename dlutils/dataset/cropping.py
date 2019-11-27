import tensorflow as tf


def random_crop(patch_size):
    '''returns the patch sampling function that takes the same
    random patch for all tensors in a given dictionary.

    If you just want to take a random patch from a *single*
    tensor, you should probably use tensorflow.image.random_crop

    '''

    def _cropper(inputs):
        '''expects a dictionary of tensors as inputs.

        '''
        with tf.name_scope('random_patch'):
            if isinstance(inputs, dict):
                iterable = list(inputs.values())
            elif isinstance(inputs, (tuple, list)):
                iterable = inputs
            else:
                # TODO consider accepting single tensors for simplicity
                raise NotImplementedError('Input must be dict, tuple or list!')

            # make sure input shape is the same for all.
            input_shape = iterable[0].shape
            if not all(val.shape == input_shape for val in iterable):
                raise ValueError(
                    'All input tenosrs need to have the same shape! {}'.format(
                        ' =! '.join(str(val.shape) for val in iterable)))

            size = tf.convert_to_tensor(patch_size, name='patch_size')

            limit = input_shape - size + 1
            offset = tf.random.uniform(
                tf.shape(input_shape),
                dtype=tf.int32,
                maxval=tf.int32.max,
            ) % limit

            if isinstance(inputs, dict):
                return {
                    key: tf.slice(value, offset, size)
                    for key, value in inputs.items()
                }
            elif isinstance(inputs, (tuple, list)):
                return tuple(
                    tf.slice(value, offset, size) for value in iterable)

    return _cropper
