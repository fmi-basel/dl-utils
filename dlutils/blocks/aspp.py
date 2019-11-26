from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers.merge import concatenate

from dlutils.layers.dilated_conv import DilatedConv2D


def _conv(**conv_params):
    '''
    '''
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate', (1, 1))
    padding = conv_params.setdefault("padding", "same")

    if dilation_rate == 1 or (isinstance(dilation_rate, (tuple, list))
                              and all(x == 1 for x in dilation_rate)):
        Conv = Conv2D
    else:
        Conv = DilatedConv2D

    def f(input):
        conv = Conv(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation='relu')(input)
        return conv

    return f


def aspp_block(x, num_features=256, rate_scale=6, n_levels=3, with_bn=True):
    '''constructs an atrous spatial pyramid pooling layer as used in the
    Deeplab models.

    Parameters
    ----------
    num_features : int
        number of filters for each convolution.
    rate_scale : int
        dilation rate added to each level in the pyramid.
    n_levels : int
        number of parallel (dilated) convolutions.
    with_bn : bool
        add batchnorm to each convolution.

    Notes
    -----
    Global feature pooling is not yet supported.

    '''
    if K.image_data_format() == 'channels_last':
        bn_axis = 3  # NOTE we would have to adjust this for 3D.
    else:
        bn_axis = 1
        assert False, 'bn_axis must be 3'

    # Construct parallel convolutions with different dilation rates.
    pyramid = []
    for level in range(1, n_levels + 1):
        z = _conv(
            filters=num_features,
            kernel_size=3,
            dilation_rate=level * rate_scale)(x)

        if with_bn:
            z = BatchNormalization(axis=bn_axis)(z)
        pyramid.append(z)

    # local kernel layer.
    z = _conv(filters=num_features, kernel_size=(1, 1), padding='same')(x)
    if with_bn:
        z = BatchNormalization(axis=bn_axis)(z)
    pyramid.append(z)

    # concatenate
    y = concatenate(pyramid, axis=bn_axis)

    # and project
    y = _conv(filters=num_features, kernel_size=(1, 1), padding='same')(y)
    if with_bn:
        y = BatchNormalization(axis=bn_axis)(y)

    return y
