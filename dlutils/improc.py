import tensorflow as tf
import numpy as np


def gaussian_filter(sigma, spatial_rank, truncate=4):
    '''
    Returns a gaussian filter for images (without batch dim)
    
    Args:
        sigma: An float or tuple/list of 2 floats indicating the standard deviation along each axis
        spatial_rank: number of spatial dimensions
        truncate: Truncate the filter at this many standard deviations
        
    Returns:
        callable taking a tensor to filter as input
    '''

    if not isinstance(sigma, tf.Tensor):
        sigma = tf.constant(sigma)
    sigma = tf.broadcast_to(sigma, [spatial_rank])

    def _gaussian_kernel(n_channels, dtype):
        half_size = tf.cast(tf.round(truncate * sigma), tf.int32)
        x_1d = [
            tf.range(-half_size[i], half_size[i] + 1, dtype=dtype)
            for i in range(len(half_size))
        ]
        g_1d = [
            tf.math.exp(-0.5 * tf.pow(x / tf.cast(sigma[idx], dtype), 2))
            for idx, x in enumerate(x_1d)
        ]

        g_kernel = g_1d[0]
        for g in g_1d[1:]:
            g_kernel = tf.tensordot(g_kernel, g, axes=0)

        g_kernel = g_kernel / tf.reduce_sum(g_kernel)
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel,
                                      (1, ) * spatial_rank + (n_channels, )),
                              axis=-1)

    def _filter(x):
        ndim = len(x.shape)
        if ndim not in (spatial_rank + 1, spatial_rank + 2):
            raise ValueError(
                'Wrong input shape, expected batch (optional) + {} spatial dimensions + channel, got {}'
                .format(spatial_rank, len(x.shape)))

        batched = ndim - 2 == spatial_rank

        if not batched:
            x = x[None]

        kernel = _gaussian_kernel(tf.shape(x)[-1], x.dtype)
        if spatial_rank == 2:
            y = tf.nn.depthwise_conv2d(x, kernel, (1, ) * (spatial_rank + 2),
                                       'SAME')
        elif spatial_rank == 3:
            if x.shape[-1] == 1:
                y = tf.nn.conv3d(x, kernel, (1, ) * (spatial_rank + 2), 'SAME')
            else:
                raise NotImplementedError(
                    '3D gaussian filter for more than one channel is not implemented, input shape: {}'
                    .format(x.shape))

        if not batched:
            y = y[0]
        return y

    return _filter


def local_max(image, min_distance=1, threshold=1, spacing=1):
    '''Finds local maxima that are above threshold.
    
    In order to avoid mutiple peaks from plateaus, the image is blurred 
    (with a kernel size related to min_distance). However multiple peaks 
    can still be returned for plateaus much larger than min_distance.
    
    Args:
        image: greyscale image with optional channel dim
        min_distance: scalar defining the min distance between local max
        threshold: absolute intensity threshold to consider a local max
        spacing: pixel/voxel size
    '''

    # implements x==max_pool(x) with pre-blurring to minimize
    # spurious max when neighbors have the same values

    if min_distance < 1:
        raise ValueError('min_distance should be > 1: {}'.format(min_distance))

    if image.shape[-1] != 1:
        image = tf.expand_dims(image, axis=-1)

    rank = len(image.shape) - 1
    spacing = np.broadcast_to(np.asarray(spacing), rank)

    gaussian = gaussian_filter(sigma=np.sqrt(min_distance / spacing),
                               spatial_rank=rank)
    image = tf.cast(image, tf.float32)

    blurred_image = gaussian(image)

    max_filt_size = np.maximum(1, min_distance / spacing * 2 + 1)
    max_image = tf.nn.max_pool(
        blurred_image[None],
        ksize=max_filt_size,
        strides=1,
        padding='SAME',
    )
    max_image = tf.squeeze(max_image, axis=[0])
    max_mask = (max_image <= blurred_image) & (image >= threshold)
    max_mask = tf.squeeze(max_mask, axis=[-1])

    return tf.where(max_mask)
