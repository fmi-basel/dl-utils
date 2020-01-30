import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Lambda, Input, Layer
from tensorflow_addons.layers import GroupNormalization

from dlutils.layers.nd_layers import get_nd_conv, get_nd_conv_transposed
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from dlutils.models.hourglass import bottleneck_conv_block, _stack_layers, hourglass_block, Activation

# TODO consider removing anistropic pooling and spacing argument from hourglass if strided/transposed conv is the preferred solution


# simple ###############################################################
def add_voting_heads(base_model, n_classes):
    '''Splits outputs of a delta loop base_model into instance vector field and semantic class.
    
    Args:
        base_model: delta_loop base model, should output at least 
            n_classes + n_spatial-dimensions channels
        n_classes: number semantic classes
    '''

    spatial_dims = len(base_model.inputs[0].shape) - 2
    y_preds, deltas = base_model.outputs

    if y_preds.shape[-1] < n_classes + spatial_dims:
        raise ValueError(
            'base_model has less than n_classes + n_spatial_dims channels: {} < {} + {}'
            .format(y_preds.shape[-1], n_classes, spatial_dims))

    vfield = y_preds[..., 0:spatial_dims]
    semantic_class = y_preds[..., spatial_dims:spatial_dims + n_classes]

    # rename outputs
    deltas = Lambda(lambda x: x, name='deltas')(deltas)
    vfield = Lambda(lambda x: x, name='vfield')(vfield)
    semantic_class = Lambda(lambda x: x, name='semantic_class')(semantic_class)

    return Model(inputs=base_model.inputs,
                 outputs=[vfield, semantic_class, deltas],
                 name=base_model.name)


# TODO handle optional init state input
# number of weight should not change


def add_learnable_resampling_layers(base_model, input_shape, factor, n_levels,
                                    downsampling_channels,
                                    upsampling_channels):
    '''Adds strided convolutions and transposed convolutions to the input 
    and ouput respectively of a delta loop base_model.
    
    Args:
        input_shape: at least the number of channels should be defined
        base_model: delta loop base model
        factor (int or tuple(int)): rescaling factor in
        n_levels: number of levels in base model (needed to determine padding size)
        downsampling_channels: number of channels after strided convolution, 
            should match base_model input channels.
        upsampling_channels: number of output channels after transposed convolution
    
    Notes:
    Intended to resample an input to ~isotropic coordinates and/or reduce 
    the resolution before being processed by base_model. base_model should
    be build with spacing after resampling
    
    The recurrent state of base_model will not match the input resolution. 
    To maintain the option to use an external init state, resampling layers 
    should be part of the loop (see TODO below)
    '''

    spatial_dims = len(base_model.inputs[0].shape) - 2
    factor = np.broadcast_to(np.array(factor), spatial_dims)

    in_kernel_size = tuple(max(3, f) for f in factor)
    out_kernel_size = tuple(max(3, 2 * f) for f in factor)

    Conv = get_nd_conv(spatial_dims)
    ConvTranspose = get_nd_conv_transposed(spatial_dims)

    conv_in = Conv(downsampling_channels,
                   kernel_size=in_kernel_size,
                   strides=factor,
                   padding='same')
    conv_out = ConvTranspose(upsampling_channels,
                             kernel_size=out_kernel_size,
                             strides=factor,
                             padding='same')

    # padd to cover downsampling by strided conv + hourglass
    # (padding in hourglass network should do nothing, already satisfied)
    # TODO infer n_levels from base_model instead of external argument

    total_factor = tuple(f * 2**(n_levels - 1) for f in factor)
    input_padding = DynamicPaddingLayer(total_factor, ndim=spatial_dims + 2)
    output_trimming = DynamicTrimmingLayer(ndim=spatial_dims + 2)

    inputs = Input(shape=input_shape)
    x = input_padding(inputs)
    x = conv_in(x)

    y_preds, deltas = base_model(x)
    y_preds = tf.stack(
        [output_trimming([inputs, conv_out(yp)]) for yp in y_preds], axis=0)

    return Model(inputs=inputs,
                 outputs=[y_preds, deltas],
                 name=base_model.name)
