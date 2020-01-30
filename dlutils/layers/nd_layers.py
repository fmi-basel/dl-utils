'''Convenience functions to get layers corresponding to the requested dimension'''

from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose


def get_nd_conv(ndim):
    if ndim == 2:
        return Conv2D
    elif ndim == 3:
        return Conv3D
    else:
        raise ValueError(
            'convolution on {} spatial dims not supported, expected to 2 or 3'.
            format(ndim))


def get_nd_conv_transposed(ndim):
    if ndim == 2:
        return Conv2DTranspose
    elif ndim == 3:
        return Conv3DTranspose
    else:
        raise ValueError(
            'transposed convolution on {} spatial dims not supported, expected to 2 or 3'
            .format(ndim))


def get_nd_maxpooling(ndim):
    if ndim == 2:
        return MaxPool2D
    elif ndim == 3:
        return MaxPool3D
    else:
        raise ValueError(
            'maxpooling {} spatial dims not supported, expected to 2 or 3'.
            format(ndim))


def get_nd_upsampling(ndim):
    if ndim == 2:
        return UpSampling2D
    elif ndim == 3:
        return UpSampling3D
    else:
        raise ValueError(
            'upsampling {} spatial dims not supported, expected to 2 or 3'.
            format(ndim))
