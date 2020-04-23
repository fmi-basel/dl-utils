'''Image data augmentation in tensorflow.

'''
import tensorflow as tf
import tensorflow_addons as tfa


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


# TODO
# pad, crop as separate operation so that they can be applied only once each
# docstrings
# keys, input range check
# unit test
# 3D version?


def random_intensity_scaling(bounds, keys):
    '''draws a random scaling factor between bounds from a uniform distribution.

     Parameters
    ----------
    bounds : tuple
        bounds of scaling factor uniform distribution
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
            scale = tf.random.uniform(shape=[],
                                      minval=bounds[0],
                                      maxval=bounds[1])
            output_dict[key] = image * scale
        return output_dict

    return _distorter


# from tensorflow.keras.preprocessing.image import apply_affine_transform # garbage, call scipy, no graph support, doesn't work with tf dataset
def random_affine_transform(interp_orders,
                            angle=180,
                            shear=0.,
                            zoom=(1, 1),
                            zoomx=(1, 1),
                            zoomy=(1, 1),
                            fill_mode=None,
                            cvals=None):
    '''interp_order: dict of interpolation NEAREST/BILINEAR with key matching 
    tensor to be processed in input dict. e.g. order=1 for raw image, 0 for label

    fill_mode ='REFLECT' 'CONSTANT' 'SYMMETRIC'
    '''

    if cvals is None:
        cvals = {key: 0 for key in interp_orders.keys()}

    if fill_mode is None:
        fill_mode = {key: 'CONSTANT' for key in interp_orders.keys()}

    def _tf_deg2rad(deg):
        pi_on_180 = 0.017453292519943295
        return deg * pi_on_180

    def _get_transforms(angle, shear, zoomx, zoomy, shape):

        # offset to iamge center
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
        shear = _tf_deg2rad(shear)
        shear_matrix = tf.reshape(
            [1, -tf.sin(shear), 0, 0,
             tf.cos(shear), 0, 0, 0, 1], (3, 3))
        transform_matrix = tf.matmul(transform_matrix, shear_matrix)

        # zoom
        zoom_matrix = tf.reshape([zoomx, 0, 0, 0, zoomy, 0, 0, 0, 1], (3, 3))
        transform_matrix = tf.matmul(transform_matrix, zoom_matrix)

        # offset back to iamge corner
        n_offset_matrix = tf.reshape(
            [1, 0, -shape[1] / 2, 0, 1, -shape[0] / 2, 0, 0, 1], (3, 3))
        transform_matrix = tf.matmul(tf.cast(transform_matrix, tf.float32),
                                     tf.cast(n_offset_matrix, tf.float32))

        return tf.reshape(transform_matrix, (-1, ))[:-1]

    def _transformer(input_dict):
        r_angle = tf.random.uniform(shape=[], minval=-angle, maxval=angle)
        r_shear = tf.random.uniform(shape=[], minval=-shear, maxval=shear)
        r_zoomx = tf.random.uniform(shape=[], minval=zoomx[0], maxval=zoomx[1])
        r_zoomy = tf.random.uniform(shape=[], minval=zoomy[0], maxval=zoomy[1])
        r_zoom = tf.random.uniform(shape=[], minval=zoom[0], maxval=zoom[1])

        r_zoomx = r_zoomx * r_zoom
        r_zoomy = r_zoomy * r_zoom

        output_dict = {key: val for key, val in input_dict.items()}
        for key, order in interp_orders.items():

            # TODO calculate min padding based on transform?
            # pad before transform
            input_shape = output_dict[key].shape
            padding = tf.shape(output_dict[key]) // 2 * [1, 1, 0]
            paddings = tf.stack([padding, padding], axis=-1)
            output_dict[key] = tf.pad(output_dict[key],
                                      paddings,
                                      mode=fill_mode[key],
                                      constant_values=cvals[key])

            transforms = _get_transforms(r_angle, r_shear, r_zoomx, r_zoomy,
                                         tf.shape(output_dict[key]))
            output_dict[key] = tfa.image.transform(
                output_dict[key][None],
                transforms,
                interpolation=order,
            )[0]

            # crop after transform
            output_dict[key] = output_dict[key][padding[0]:-padding[0],
                                                padding[1]:-padding[1]]
            output_dict[key].set_shape(input_shape)

        return output_dict

    return _transformer


# TODO move to utils
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


# TODO option to choose device
# TODO generalized filter augmentation (median, min, max, unsharp, etc.)
# TODO if not active, don't apply the filter at all (tf.cond ...)
def random_gaussian_filter(sigma_range, keys, active_p=1.):
    def _filter(input_dict):
        active = tf.random.uniform(shape=[], dtype=tf.float32) < active_p
        active = tf.cast(active, tf.float32)

        sigma = tf.random.uniform(shape=[],
                                  minval=sigma_range[0],
                                  maxval=sigma_range[1],
                                  dtype=tf.float32)

        sigma = tf.maximum(sigma * active, 0.5)

        with tf.device('/gpu:0'):
            output_dict = {key: val for key, val in input_dict.items()}
            for key in keys:
                output_dict[key] = gaussian_filter(sigma)(output_dict[key])

        return output_dict

    return _filter


def random_hsv(keys,
               max_delta_hue=0,
               lower_saturation=1,
               upper_saturation=1,
               lower_value=1,
               upper_value=1):
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


# TODO reparametrize with random quantile clipping
def random_clipping(min_center, min_sigma, max_center, max_sigma, keys):
    def _filter(input_dict):

        minval = tf.random.normal(shape=[], mean=min_center, stddev=min_sigma)
        maxval = tf.random.normal(shape=[], mean=max_center, stddev=max_sigma)

        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            output_dict[key] = tf.clip_by_value(output_dict[key], minval,
                                                maxval)

        return output_dict

    return _filter


# warp #####################################################################
# TODO double check, seems to leave single pixel holes (obvious on label image)
# bilinear grid sampling from spatial transformer impelmentation
# https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    # x = tf.cast(x, 'float32')
    # y = tf.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


# TODO tfa.image.dense_image_warp (more precisely resample bilinear) does not work in graph mode on gpu??
# TODO random sampling of smoothing? relative to amplitude?
def random_warp(max_amplitude, interp_orders, fill_mode=None, cvals=None):

    if cvals is None:
        cvals = {key: 0 for key in interp_orders.keys()}

    if fill_mode is None:
        fill_mode = {key: 'CONSTANT' for key in interp_orders.keys()}

    def _warp_bilinear(img, flow):

        grid_x, grid_y = tf.meshgrid(
            tf.range(tf.shape(img)[2], dtype=flow.dtype),
            tf.range(tf.shape(img)[1], dtype=flow.dtype))

        grid_x = grid_x + flow[..., 0]
        grid_y = grid_y + flow[..., 1]

        return bilinear_sampler(img, grid_x[None], grid_y[None])

    def _warp_nearest(img, flow):

        grid_x, grid_y, grid_ch = tf.meshgrid(
            tf.range(tf.shape(img)[2], dtype=flow.dtype),
            tf.range(tf.shape(img)[1], dtype=flow.dtype),
            tf.range(tf.shape(img)[3], dtype=flow.dtype))

        grid_x = grid_x + flow[..., 0:1]
        grid_y = grid_y + flow[..., 1:2]

        stacked_grid = tf.cast(tf.stack([grid_y, grid_x, grid_ch], axis=-1),
                               flow.dtype)
        query_points = tf.cast(tf.round(stacked_grid), tf.int32)[None]

        return tf.gather_nd(img, query_points, batch_dims=1)

    def _warp(img, flow, order):

        if order == 1:
            return _warp_bilinear(img, flow)
        elif order == 0:
            return _warp_nearest(img, flow)
        else:
            raise ValueError(
                '{} order interpolation not supported'.format(order))

    def _warper(input_dict):
        with tf.device('/gpu:0'):
            keys = list(interp_orders.keys())

            amplitude = tf.random.uniform(shape=[],
                                          minval=0.,
                                          maxval=max_amplitude,
                                          dtype=tf.float32)
            pad_size = tf.cast(amplitude, tf.int32) + 1
            paddings = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
            smoothing = 2 * amplitude

            flow = tf.random.uniform(input_dict[keys[0]].shape[:2] + (2, ), -1,
                                     1)
            flow = gaussian_filter(smoothing)(flow)
            flow = flow - tf.reduce_min(flow)
            flow = flow / (1e-6 + tf.reduce_max(flow))
            flow = (flow * 2 - 1) * amplitude
            flow = tf.cast(flow, tf.float32)

            flow = tf.pad(flow, paddings)  # zero padding

            output_dict = {key: val for key, val in input_dict.items()}
            for key, order in interp_orders.items():
                # pad before transform
                input_shape = output_dict[key].shape
                input_dtype = output_dict[key].dtype
                output_dict[key] = tf.cast(output_dict[key], tf.float32)

                output_dict[key] = tf.pad(output_dict[key],
                                          paddings,
                                          mode=fill_mode[key],
                                          constant_values=cvals[key])

                output_dict[key] = _warp(output_dict[key][None], flow,
                                         order)[0]
                #output_dict[key] = tfa.image.dense_image_warp(output_dict[key][None],flow)[0]

                # crop after transform
                output_dict[key] = output_dict[key][pad_size:-pad_size,
                                                    pad_size:-pad_size]
                output_dict[key].set_shape(input_shape)
                output_dict[key] = tf.cast(output_dict[key], input_dtype)

            return output_dict

    return _warper
