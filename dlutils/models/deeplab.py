from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.layers import merge
from keras.engine import Model
from keras.layers import Reshape, RepeatVector

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
               rate_scale=1,
               output_stride=16,
               input_shape=(512, 512, 3)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        assert False, 'bn_axis must be 3'

    #conv3_3_1 = ZeroPadding2D(padding=(6 * rate_scale, 6 * rate_scale))(x)
    conv3_3_1 = _conv(
        filters=num_filters,
        kernel_size=(3, 3),
        dilation_rate=(6 * rate_scale, 6 * rate_scale))(x)
    conv3_3_1 = BatchNormalization(axis=bn_axis)(conv3_3_1)

    #conv3_3_2 = ZeroPadding2D(padding=(12 * rate_scale, 12 * rate_scale))(x)
    conv3_3_2 = _conv(
        filters=num_filters,
        kernel_size=(3, 3),
        dilation_rate=(12 * rate_scale, 12 * rate_scale))(x)
    conv3_3_2 = BatchNormalization(axis=bn_axis)(conv3_3_2)

    #conv3_3_3 = ZeroPadding2D(padding=(18 * rate_scale, 18 * rate_scale))(x)
    conv3_3_3 = _conv(
        filters=num_filters,
        kernel_size=(3, 3),
        dilation_rate=(18 * rate_scale, 18 * rate_scale))(x)
    conv3_3_3 = BatchNormalization(axis=bn_axis)(conv3_3_3)

    conv1_1 = _conv(filters=num_filters, kernel_size=(1, 1), padding='same')(x)
    conv1_1 = BatchNormalization(axis=bn_axis)(conv1_1)

    # global_feat = AveragePooling2D((input_shape[0] / output_stride,
    #                                 input_shape[1] / output_stride))(x)
    # global_feat = _conv(
    #     filters=num_filters, kernel_size=(1, 1), padding='same')(global_feat)
    # global_feat = BatchNormalization()(global_feat)
    # global_feat = BilinearUpSampling2D(
    #     (256, input_shape[0] / output_stride, input_shape[1] / output_stride),
    #     factor=input_shape[1] / output_stride)(global_feat)

    xdim = conv1_1.get_shape()[1].value
    ydim = conv1_1.get_shape()[2].value

    # global_feat = GlobalAveragePooling2D()(x)
    # global_feat = RepeatVector(xdim * ydim)(global_feat)
    # global_feat = Reshape((xdim, ydim,
    #                        global_feat.get_shape()[-1].value))(global_feat)

    y = merge(
        [
            conv3_3_1,
            conv3_3_2,
            conv3_3_3,
            # conv3_3_4,
            conv1_1,
            # global_feat,
        ],
        mode='concat',
        concat_axis=3)

    # y = _conv_bn_relu(filters=1, kernel_size=(1, 1),padding='same')(y)
    y = _conv(filters=256, kernel_size=(1, 1), padding='same')(y)
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
