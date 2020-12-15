'''Image data augmentation in tensorflow.

'''
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def random_axis_flip(axis, flip_prob):
    '''reverses axis with probability threshold

    Parameters
    ----------
    axis : int
        axis index to be flipped.
    flip_prob : float
        probability of flipping.

    Returns
    -------
    flipper : func
        transformation function.

    '''
    def _flipper(input_dict):
        '''expects an input dictionary.
        '''
        draw_prob = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=1,
                                      dtype=tf.float32)

        # NOTE the cell-var-from-loop warning is disabled as the lambdas
        # are executed immediately by tf.cond and thus, evaluation happens
        # when val is still the *current* val.
        return {
            key: tf.cond(
                draw_prob <= flip_prob,
                lambda: tf.reverse(val, [axis]),  # pylint: disable = W0640
                lambda: val)  # pylint: disable = W0640
            for key, val in input_dict.items()
        }

    return _flipper


def random_gaussian_noise(noise_mu, noise_sigma, keys):
    '''adds pixel-wise IID gaussian noise to the given tensors.

    The noise distributions follows N(0, sigma) where

       sigma~N(noise_mu, noise_sigma).

    sigma is drawn for each batch independently.
    If sigma <= 0, then no noise is added to the current run.


    Parameters
    ----------
    noise_mu : float
        mean of the distribution from which sigma is drawn.
    noise_sigma : float
        standard deviation of the distribution from which sigma is drawn.
    keys : list
        list of keys indicating to which entries in the input_dict the
        noise shall be added.

    Returns
    -------
    distorter : func
        transformation function.

    '''
    if not keys:
        raise ValueError(
            'keys cant be {}! Expected a non-empty list of dict keys.'.format(
                keys))

    def _distorter(input_dict):
        '''adds gaussian noise to the entries in input_dict that
        are indexed by keys.

        '''
        sigma = tf.maximum(
            0., tf.random.normal(shape=[], mean=noise_mu, stddev=noise_sigma))

        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            noise = tf.random.normal(shape=tf.shape(output_dict[key]),
                                     mean=0,
                                     stddev=sigma)
            output_dict[key] = output_dict[key] + noise

        return output_dict

    return _distorter


def random_gaussian_offset(offset_sigma, keys):
    '''draws a random offset from N(0, offset_sigma) and
    adds it to the given input[key].

     Parameters
    ----------
    offset_sigma : float
        standard deviation of the distribution from which sigma is drawn.
    keys : list
        list of keys indicating to which entries in the input_dict the
        noise shall be added.

    Returns
    -------
    distorter : func
        transformation function.

    '''
    def _distorter(input_dict):
        '''adds offset to the entries in input_dict that
        are indexed by keys.

        '''
        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            image = output_dict[key]
            offset = tf.random.normal(shape=[], mean=0, stddev=offset_sigma)
            output_dict[key] = image + offset
        return output_dict

    return _distorter


def random_intensity_scaling(bounds, keys):
    '''Draws a random scaling factor between bounds from a uniform distribution in log space.

    Parameters
    ----------
    bounds : tuple
        scaling factor bounds
    keys : list
        list of keys indicating to which entries in the input_dict the
        scalling should be applied.

    Returns
    -------
    distorter : func
        transformation function.

    '''

    log_bounds = tuple(np.log10(bounds))

    def _distorter(input_dict):
        '''adds offset to the entries in input_dict that
        are indexed by keys.

        '''
        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            image = output_dict[key]
            scale = tf.random.uniform(shape=[],
                                      minval=log_bounds[0],
                                      maxval=log_bounds[1])
            scale = tf.math.pow(10., scale)
            output_dict[key] = image * scale
        return output_dict

    return _distorter


# TODO import from tf improc implementation + add spacing argument + test 3D patch size
def gaussian_filter(sigma):
    def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
        # https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) /
                          (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

    def _filter(x):
        kernel_size = tf.cast(4 * sigma, tf.int32)
        kernel = _gaussian_kernel(kernel_size, sigma,
                                  tf.shape(x)[-1], tf.float32)

        return tf.nn.depthwise_conv2d(x[None], kernel, [1, 1, 1, 1], 'SAME')[0]

    return _filter


def random_gaussian_filter(sigma_range, keys, active_p=1., device='/gpu'):
    '''
    Applies a random gaussian filter to inputs specified by keys
    
    Parameters
    ----------
    sigma_range : tuple
        bounds to sample sigma from a uniform distribution
    keys : list
        list of keys indicating to which entries in the input_dict should
        be blurred.
    active_p : float
        fraction of times the filter should be active
    device : string
        device to use to perform the filtering operation(cpu/gpu). Gaussian 
        filtering on CPU with a large sigma can become a bottlneck during
        training.
    

    Returns
    -------
    distorter : func
        transformation function.
    '''
    def _filter(input_dict):

        with tf.device(device):
            active = tf.less_equal(
                tf.random.uniform(shape=[], dtype=tf.float32), active_p)

            def apply_filter():
                sigma = tf.random.uniform(shape=[],
                                          minval=sigma_range[0],
                                          maxval=sigma_range[1],
                                          dtype=tf.float32)

                filter_fun = gaussian_filter(sigma)

                output_dict = {key: val for key, val in input_dict.items()}
                for key in keys:
                    output_dict[key] = filter_fun(output_dict[key])

                return output_dict

            def pass_trough():
                return {key: val for key, val in input_dict.items()}

            return tf.cond(active, apply_filter, pass_trough)

    return _filter


def random_hsv(keys,
               max_delta_hue=0,
               lower_saturation=1,
               upper_saturation=1,
               lower_value=1,
               upper_value=1):
    '''
    Randomly Adjusts hue, saturation, value in YIQ color space of inputs specified by keys
    
    c.f. tfa.image.random_hsv_in_yiq
    '''
    def _filter(input_dict):

        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            output_dict[key] = tfa.image.random_hsv_in_yiq(
                output_dict[key],
                max_delta_hue,
                lower_saturation,
                upper_saturation,
                lower_value,
                upper_value,
            )

        return output_dict

    return _filter


def centered_affine_transform_matrix(angle, shear_angle, zoomx, zoomy, shape):
    '''Returns a 2D affine homogeneous transform matrix centered the image center'''
    def _tf_deg2rad(deg):
        pi_on_180 = 0.017453292519943295
        return deg * pi_on_180

    # offset to image center
    transform_matrix = tf.reshape(
        [1, 0, shape[1] / 2, 0, 1, shape[0] / 2, 0, 0, 1], (3, 3))
    transform_matrix = tf.cast(transform_matrix, tf.float32)

    # rotation
    angle = _tf_deg2rad(angle)
    rotation_matrix = tf.reshape([
        tf.cos(angle), -tf.sin(angle), 0,
        tf.sin(angle),
        tf.cos(angle), 0, 0, 0, 1
    ], (3, 3))

    transform_matrix = tf.matmul(transform_matrix, rotation_matrix)

    # shear
    shear_angle = _tf_deg2rad(shear_angle)
    shear_matrix = tf.reshape(
        [1, -tf.sin(shear_angle), 0, 0,
         tf.cos(shear_angle), 0, 0, 0, 1], (3, 3))
    transform_matrix = tf.matmul(transform_matrix, shear_matrix)

    # zoom
    zoom_matrix = tf.reshape([1 / zoomx, 0, 0, 0, 1 / zoomy, 0, 0, 0, 1],
                             (3, 3))
    transform_matrix = tf.matmul(transform_matrix, zoom_matrix)

    # offset back to image corner
    n_offset_matrix = tf.reshape(
        [1, 0, -shape[1] / 2, 0, 1, -shape[0] / 2, 0, 0, 1], (3, 3))
    transform_matrix = tf.matmul(tf.cast(transform_matrix, tf.float32),
                                 tf.cast(n_offset_matrix, tf.float32))
    return transform_matrix


# from tensorflow.keras.preprocessing.image import apply_affine_transform # garbage, call scipy, no graph support, doesn't work with tf dataset
def random_affine_transform(interp_methods,
                            angle=180.,
                            shear=0.,
                            zoom=(1, 1),
                            zoomx=(1, 1),
                            zoomy=(1, 1),
                            fill_mode=None,
                            cvals=None):
    '''
    Parameters
    ----------
    interp_methods : dict
        interpolation 'NEAREST'|'BILINEAR' for each input.
        e.g. BILINEAR for raw image, NEAREST for labels
    angle : float
        rotation angle is uniformly sampled between [0,angle]
    shear : float
        shear angle is uniformly sampled between [0,shear]
    zoom : tuple
        bounds to sample the global scaling factor (uniformly sampled in log space)
    zoomx : tuple
        bounds to sample the x scaling factor (uniformly sampled in log space)
    zoomy : tuple
        bounds to sample the y scaling factor (uniformly sampled in log space)
    fill_mode : dict
        padding fill mode 'REFLECT'|'CONSTANT'|'SYMMETRIC' for each input
    cvals : dict
        padding values for each input if fill_mode is 'CONSTANT'.
        e.g. fill labels with negative values to avoid supervision loss on
        padded regions (provided the loss function handles it)

    Returns
    -------
    distorter : func
        transformation function.
    '''

    if cvals is None:
        cvals = {key: 0 for key in interp_methods.keys()}

    if fill_mode is None:
        fill_mode = {key: 'CONSTANT' for key in interp_methods.keys()}

    def _log_space_random_uniform(bounds):
        log_bounds = tuple(np.log10(bounds))
        val = tf.random.uniform(shape=[],
                                minval=log_bounds[0],
                                maxval=log_bounds[1])
        return tf.math.pow(10., val)

    def _transformer(input_dict):
        r_angle = tf.random.uniform(shape=[], minval=-angle, maxval=angle)
        r_shear = tf.random.uniform(shape=[], minval=-shear, maxval=shear)
        r_zoomx = _log_space_random_uniform(zoomx)
        r_zoomy = _log_space_random_uniform(zoomy)
        r_zoom = _log_space_random_uniform(zoom)

        r_zoomx = r_zoomx * r_zoom
        r_zoomy = r_zoomy * r_zoom

        output_dict = {key: val for key, val in input_dict.items()}
        for key, interp in interp_methods.items():

            # pad with 0.5 x img_shape on each side
            # should cover all reasonable cases (e.g. 360 rotation)
            # tf.pad padding size is anyway limited to <1x img_shape
            # and will not padd enough for zoom level < 0.5
            input_shape = output_dict[key].shape
            padding = tf.shape(output_dict[key]) // 2 * [1, 1, 0]
            paddings = tf.stack([padding, padding], axis=-1)
            output_dict[key] = tf.pad(output_dict[key],
                                      paddings,
                                      mode=fill_mode[key],
                                      constant_values=cvals[key])

            transforms = centered_affine_transform_matrix(
                r_angle, r_shear, r_zoomx, r_zoomy, tf.shape(output_dict[key]))
            transforms = tf.reshape(transforms, (-1, ))[:-1]

            output_dict[key] = tfa.image.transform(
                output_dict[key][None],
                transforms,
                interpolation=interp,
            )[0]

            # crop after transform
            output_dict[key] = output_dict[key][padding[0]:-padding[0],
                                                padding[1]:-padding[1]]
            output_dict[key].set_shape(input_shape)

        return output_dict

    return _transformer


def random_warp(max_amplitude,
                interp_methods,
                fill_mode=None,
                cvals=None,
                device='/gpu'):
    '''
    Deforms the inputs with a smoothed random flow field of amplitude 
    uniformly sampled between [0,max_amplitude].
    
    Parameters
    ----------
    max_amplitude : float
        max amplitude of the flow field
    interp_methods : dict
        interpolation 'NEAREST'|'BILINEAR' for each input.
        e.g. BILINEAR for raw image, NEAREST for labels
    fill_mode : dict
        padding fill mode 'REFLECT'|'CONSTANT'|'SYMMETRIC' for each input
    cvals : dict
        padding values for each input if fill_mode is 'CONSTANT'.
        e.g. fill labels with negative values to avoid supervision loss on
        padded regions (provided the loss function handles it)
    device : string
        device to use (cpu/gpu). Gaussian smoothing of the flow field
        on CPU with large amplitude can become a bottlneck during training.

    Returns
    -------
    distorter : func
        transformation function.
    '''

    if cvals is None:
        cvals = {key: 0 for key in interp_methods.keys()}

    if fill_mode is None:
        fill_mode = {key: 'CONSTANT' for key in interp_methods.keys()}

    def _warp_bilinear(img, flow):

        grid_x, grid_y = tf.meshgrid(
            tf.range(tf.shape(img)[2], dtype=flow.dtype),
            tf.range(tf.shape(img)[1], dtype=flow.dtype))

        grid_x = grid_x + flow[..., 0]
        grid_y = grid_y + flow[..., 1]

        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.reshape(grid, (-1, 2))

        img_resampled_flat = tfa.image.interpolate_bilinear(img,
                                                            grid[None],
                                                            indexing='xy')
        return tf.reshape(img_resampled_flat, tf.shape(img))

    def _warp_nearest(img, flow):

        grid_x, grid_y, grid_ch = tf.meshgrid(
            tf.range(tf.shape(img)[2], dtype=flow.dtype),
            tf.range(tf.shape(img)[1], dtype=flow.dtype),
            tf.range(tf.shape(img)[3], dtype=flow.dtype))

        grid_x = grid_x + flow[..., 0:1]
        grid_y = grid_y + flow[..., 1:2]

        grid = tf.stack([grid_y, grid_x, grid_ch], axis=-1)
        grid = tf.cast(grid, flow.dtype)
        query_points = tf.cast(tf.round(grid), tf.int32)[None]

        return tf.gather_nd(img, query_points, batch_dims=1)

    def _warp(img, flow, order):

        if order == 'BILINEAR':
            return _warp_bilinear(img, flow)
        elif order == 'NEAREST':
            return _warp_nearest(img, flow)
        else:
            raise ValueError('{} interpolation not supported'.format(order))

    def _warper(input_dict):
        with tf.device(device):
            keys = list(interp_methods.keys())

            amplitude = tf.random.uniform(shape=[],
                                          minval=0.,
                                          maxval=max_amplitude,
                                          dtype=tf.float32)
            pad_size = tf.cast(amplitude, tf.int32) + 1
            paddings = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
            smoothing_sigma = 2 * amplitude

            flow = tf.random.uniform(input_dict[keys[0]].shape[:2] + (2, ))
            flow = gaussian_filter(smoothing_sigma)(flow)
            flow = flow - tf.reduce_min(flow)
            flow = flow / (1e-6 + tf.reduce_max(flow))
            flow = (flow * 2 - 1) * amplitude
            flow = tf.pad(flow, paddings)  # zero padding

            output_dict = {key: val for key, val in input_dict.items()}
            for key, interp in interp_methods.items():
                # pad before transform
                input_shape = output_dict[key].shape
                input_dtype = output_dict[key].dtype
                output_dict[key] = tf.cast(output_dict[key], tf.float32)

                output_dict[key] = tf.pad(output_dict[key],
                                          paddings,
                                          mode=fill_mode[key],
                                          constant_values=cvals[key])

                output_dict[key] = _warp(output_dict[key][None], flow,
                                         interp)[0]

                # crop after transform
                output_dict[key] = output_dict[key][pad_size:-pad_size,
                                                    pad_size:-pad_size]
                output_dict[key].set_shape(input_shape)
                output_dict[key] = tf.cast(output_dict[key], input_dtype)

            return output_dict

    return _warper
