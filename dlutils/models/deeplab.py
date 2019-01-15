from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.engine import Model
from keras.layers.merge import concatenate

from dlutils.layers.upsampling import BilinearUpSampling2D
from dlutils.layers.dilated_conv import DilatedConv2D
from dlutils.models.resnext import ResnextConstructor


def _conv(**conv_params):
    '''
    '''
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate', (1, 1))
    padding = conv_params.setdefault("padding", "same")

    if all(x == 1 for x in dilation_rate):
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


def aspp_block(x,
               num_filters=256,
               rate_scale=6,
               n_levels=3,
               with_bn=True):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        assert False, 'bn_axis must be 3'

    pyramid = []

    for level in range(1, n_levels + 1):
        z = _conv(
            filters=num_filters,
            kernel_size=(3, 3),
            dilation_rate=(level * rate_scale, rate_scale))(x)

        if with_bn:
            z = BatchNormalization(axis=bn_axis)(z)
        pyramid.append(z)

    # local kernel layer.
    z = _conv(filters=num_filters, kernel_size=(1, 1), padding='same')(x)
    if with_bn:
        z = BatchNormalization(axis=bn_axis)(z)
    pyramid.append(z)

    # concatenate
    y = concatenate(pyramid, axis=3)

    # y = _conv_bn_relu(filters=1, kernel_size=(1, 1), padding='same')(y)
    y = _conv(filters=num_filters * 4, kernel_size=(1, 1), padding='same')(y)
    if with_bn:
        y = BatchNormalization()(y)

    return y


def Deeplab(input_shape, output_stride=1, rate_scale=1, **backbone_params):
    '''
    '''

    backbone = ResnextConstructor(**backbone_params).construct_without_decoder(
        input_shape=input_shape)

    x = backbone.output

    x = aspp_block(
        x,
        256,
        rate_scale=rate_scale,
        output_stride=output_stride,
        input_shape=input_shape)

    x = _conv(filters=1, kernel_size=(1, 1), padding='same')(x)
    x = BilinearUpSampling2D(
        (1, input_shape[0], input_shape[1]), factor=output_stride)(x)
    out = _conv(
        filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=backbone.inputs, outputs=out)
    return model
