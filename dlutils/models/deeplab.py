from keras.layers import Conv2D
from keras.engine import Model

from dlutils.layers.upsampling import BilinearUpSampling2D
from dlutils.layers.dilated_conv import DilatedConv2D
from dlutils.blocks.aspp import aspp_block
from dlutils.models.resnext import ResnextConstructor


def _conv(**conv_params):
    '''TODO Remove.

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


def Deeplab(input_shape,
            output_stride=1,
            n_classes=1,
            rate_scale=6,
            **backbone_params):
    '''NOTE [wip]

    TODO Consider turning this into a head.

    '''
    backbone = ResnextConstructor(**backbone_params).construct_without_decoder(
        input_shape=input_shape)

    x = backbone.output

    x = aspp_block(x, num_features=256, rate_scale=rate_scale)

    x = BilinearUpSampling2D(
        (1, input_shape[0], input_shape[1]), factor=output_stride)(x)
    out = Conv2D(
        filters=n_classes,
        kernel_size=(1, 1),
        padding='same',
        activation='sigmoid')(x)

    model = Model(inputs=backbone.inputs, outputs=out)
    return model
