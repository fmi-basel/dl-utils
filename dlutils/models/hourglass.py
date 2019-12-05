'''Loose implementation of hourglass network.

Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass
    networks for human pose estimation." European Conference on
    Computer Vision. Springer, Cham, 2016.

Notes:
- skip branches are concatenated instead of summed (i.e. like in U-net)'''

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Activation, Dropout, Input
from tensorflow_addons.layers import GroupNormalization

from dlutils.layers.nd_layers import get_nd_conv, get_nd_maxpooling, get_nd_upsampling
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer

# NOTE
# layers declared in the main function and used in the returned inner function are not shared
# somehow NOT equivalent to fcts = [lambda x: a * x for a  in range(10)]
# see unit test test_double_bottleneck_blocks()


def _stack_layers(layers):
    '''
    '''
    def block(x):
        for layer in layers:
            x = layer(x)
        return x

    return block


def bottleneck_conv_block(channels=32,
                          spatial_dims=2,
                          activation='relu',
                          norm_groups=4):
    '''
    Notes
    -----
    pre-activation to keep the residual path clear as described in:
    
    HE, Kaiming, et al. Identity mappings in deep residual networks.
    In: European conference on computer vision. Springer, Cham, 2016.
    S. 630-645.
    '''

    Conv = get_nd_conv(spatial_dims)

    seq = _stack_layers([
        Activation(activation),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels // 2, kernel_size=1, padding='same'),
        Activation(activation),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels // 2, kernel_size=3, padding='same'),
        Activation(activation),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels, kernel_size=1, padding='same'),
    ])

    def block(x):
        return x + seq(x)

    return block


def hourglass_block(n_levels=4,
                    channels=32,
                    spatial_dims=2,
                    pooling_interval=1,
                    activation='relu',
                    norm_groups=4):
    pooling_interval = np.broadcast_to(np.array(pooling_interval),
                                       spatial_dims)

    Conv = get_nd_conv(spatial_dims)
    MaxPool = get_nd_maxpooling(spatial_dims)
    UpSampling = get_nd_upsampling(spatial_dims)

    # define layers ################################################
    # conv block for down path
    downs = [
        bottleneck_conv_block(channels, spatial_dims, activation, norm_groups)
        for _ in range(n_levels)
    ]

    # conv blocks for residual/skip paths
    skips = [
        bottleneck_conv_block(channels, spatial_dims, activation, norm_groups)
        for _ in range(n_levels)
    ]

    # conv block for middle layer
    mid = bottleneck_conv_block(channels, spatial_dims, activation,
                                norm_groups)

    # conv blocks for up path
    ups = [
        _stack_layers([
            Conv(channels, kernel_size=1,
                 padding='same'),  # reduce concatenated channels by half
            Activation(activation),
            bottleneck_conv_block(channels, spatial_dims, activation,
                                  norm_groups)
        ]) for _ in range(n_levels - 1)
    ]

    pools = [
        MaxPool((l % pooling_interval == 0) + 1) for l in range(1, n_levels)
    ]
    upsamples = [
        UpSampling((l % pooling_interval == 0) + 1)
        for l in range(1, n_levels)
    ]

    def block(x):

        # down, keeping a handle on intermediate outputs to build skip connections
        level_outputs = [downs[0](x)]
        for down, pool in zip(downs[1:], pools):
            level_outputs.append(down(pool(level_outputs[-1])))

        # residual/skip
        for idx, skip in enumerate(skips):
            level_outputs[idx] = skip(level_outputs[idx])

        # middle
        x = mid(level_outputs.pop(-1))

        # up
        for level_var, up, upsample in zip(level_outputs[::-1], ups[::-1],
                                           upsamples[::-1]):

            x = upsample(x)
            x = tf.concat([x, level_var], axis=-1)
            x = up(x)

        return x

    return block


def single_hourglass(output_channels,
                     n_levels=4,
                     channels=32,
                     spatial_dims=2,
                     pooling_interval=1,
                     activation='relu',
                     norm_groups=4):
    '''Combines an hourglass block with input/output blocks to 
    increase/decrease the number of channels.
    
    Notes:
    Expects the input to be divisible by 2**((n_levels-1)//pooling_interval)
    (i.e. already padded)'''

    Conv = get_nd_conv(spatial_dims)

    hglass = _stack_layers([
        Conv(channels, kernel_size=1, padding='same'),
        bottleneck_conv_block(channels, spatial_dims, activation, norm_groups),
        hourglass_block(n_levels, channels, spatial_dims, pooling_interval,
                        activation, norm_groups),
        bottleneck_conv_block(channels, spatial_dims, activation, norm_groups),
        Activation(activation),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels, kernel_size=3, padding='same'),
        Activation(activation),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(output_channels, kernel_size=1, padding='same'),
    ])

    return hglass


def delta_loop(output_channels, recurrent_block, default_n_steps=3):
    '''Recursively applies a given block to refine its output.
    
    Args:
        output_channels: number of output channels.
        recurrent_block: a network taking (input_channels + output_channels) as 
        input and outputting output_channels
        n_steps: number of times the block is applied
    '''
    def block(x, state=None, n_steps=None):

        if state is None:
            # ~recurrent_shape = tf.concat([tf.shape(x)[:-1], tf.constant([output_channels])], axis=0)
            # ~state = tf.zeros(recurrent_shape, x.dtype)

            # TODO figure out how to get theshape to serialize properly
            state = tf.zeros_like(x[..., 0])[..., None]
            state = tf.keras.backend.repeat_elements(state,
                                                     output_channels,
                                                     axis=-1)

        if n_steps is None:
            n_steps = default_n_steps
        # static unrolling #############################################

        # ~outputs = []
        # ~for _ in range(n_steps): # static unrolled loop

        # ~delta = recurrent_block(tf.concat([x, state], axis=-1))
        # ~state = state + delta
        # ~outputs.append(state)

        # ~return tf.stack(outputs, axis=0)

        # dynamic alternative ##########################################
        i = tf.constant(0)
        outputs = tf.TensorArray(tf.float32, size=n_steps)

        def cond(i, outputs, state):
            return tf.less(i, n_steps)

        def body(i, outputs, state):
            delta = recurrent_block(tf.concat([x, state], axis=-1))
            state = state + delta
            outputs = outputs.write(i, state)

            i = tf.add(i, 1)
            return (i, outputs, state)

        i, outputs, state = tf.while_loop(cond, body, [i, outputs, state])

        outputs = outputs.stack()
        print(outputs.shape)

        return outputs

    return block


def GenericRecurrentHourglassBase(input_shape,
                                  output_channels,
                                  pass_init=False,
                                  default_n_steps=3,
                                  n_levels=4,
                                  channels=32,
                                  spatial_dims=2,
                                  spacing=1,
                                  activation='relu',
                                  norm_groups=4):

    # approximate isotropic field of view by adjusting pooling interval
    spacing = np.broadcast_to(np.array(spacing), spatial_dims)
    normalized_spacing = spacing / spacing.min()
    pooling_interval = (np.floor(np.log2(normalized_spacing)) + 1).astype(int)
    factor = 2**((n_levels - 1) // pooling_interval)

    input_padding = DynamicPaddingLayer(factor=factor, ndim=spatial_dims + 2)
    hglass = single_hourglass(output_channels, n_levels, channels,
                              spatial_dims, pooling_interval, activation,
                              norm_groups)
    r_hglass = delta_loop(output_channels, hglass, default_n_steps)
    output_trimming = DynamicTrimmingLayer(ndim=spatial_dims + 2)

    # TODO is it possible to change n_steps and initial state at run time? (i.e. have optional inputs/attributes)
    if pass_init is False:
        x = Input(shape=input_shape)

        y = input_padding(x)
        y = r_hglass(y)
        y = [output_trimming([x, single_iter_y]) for single_iter_y in y]
        y = tf.stack(y)

        return Model(inputs=[x], outputs=[y])

    else:
        x = Input(shape=input_shape)
        state = Input(shape=input_shape[:-1] + (output_channels, ))

        y = input_padding(x)
        padded_state = input_padding(state)
        y = r_hglass(y, padded_state)
        y = [output_trimming([x, single_iter_y]) for single_iter_y in y]
        y = tf.stack(y, axis=0)

        return Model(inputs=[x, state], outputs=[y])


# alternatively:
# dynamic loop #########################################################
# class DeltaLoop(Layer):
# '''Recursively applies a given block to refine its output.'''

# def __init__(self, output_channels, recurrent_block, **kwargs):
# '''
# Args:
# output_channels: number of output channels.
# recurrent_block: a network taking (input_channels + output_channels) as
# input and outputting output_channels
# '''
# super().__init__(**kwargs)

# self.recurrent_block = recurrent_block
# self.output_channels = output_channels

# @tf.function
# def call(self, x, n_steps=3, training=None):

# # unpack initial state if given, else init as zero
# if isinstance(x, list):
# state = x[1]
# x = x[0]
# else:
# recurrent_shape = tf.concat([tf.shape(x)[:-1], (self.output_channels,)], axis=0)
# state = tf.zeros(recurrent_shape, x.dtype)

# outputs =  tf.TensorArray(tf.float32, size=n_steps)

# # TODO dynamic loop not converted to graph correctly
# for i in tf.range(n_steps): # dynamically unrolled loop
# # ~for i in range(n_steps): # dynamically unrolled loop

# delta = self.recurrent_block(tf.concat([x, state], axis=-1))
# state = state + delta
# outputs = outputs.write(i,state)

# return outputs.stack()

# def get_config(self):
# config = super().get_config()
# config['recurrent_block'] = self.recurrent_block
# config['output_channels'] = self.output_channels

# return config

# # TODO temporary fix for testing
# # if single hglass is not wrapped in model (layer?), trainable weights are not found
# def single_hglass_model(single_hglass, input_shape):

# x = Input(shape=input_shape)
# y = single_hglass(x)

# return Model(inputs=[x], outputs=[y])

# class RecurrentHourGlass(Model):
# def __init__(self, output_channels, n_steps=3, n_levels=4, channels=32, spatial_dims=2, spacing=1, activation='relu', norm_groups=4, **kwargs):
# super().__init__(**kwargs)

# self.output_channels = output_channels
# self.n_steps = n_steps
# self.n_levels = n_levels
# self.channels = channels
# self.spatial_dims = spatial_dims
# self.spacing = spacing
# self.activation = activation
# self.norm_groups = norm_groups

# # approximate isotropic field of view by adjusting pooling interval
# spacing = np.broadcast_to(np.array(spacing), spatial_dims)
# normalized_spacing = spacing / spacing.min()
# pooling_interval = (np.floor(np.log2(normalized_spacing)) + 1).astype(int)
# factor = 2**((n_levels-1)//pooling_interval)

# self.input_padding = DynamicPaddingLayer(factor=factor, ndim=spatial_dims+2)
# hglass = single_hourglass(output_channels, n_levels, channels, spatial_dims, pooling_interval, activation, norm_groups)
# hglass = single_hglass_model(hglass, input_shape=(None,None,2)) # 2 ch concat of input and state

# self.r_hglass = DeltaLoop(output_channels, hglass)
# self.output_trimming = DynamicTrimmingLayer(ndim=spatial_dims+3)

# # ~@tf.function
# def call(self, x, training=None):

# y = self.input_padding(x)
# y = self.r_hglass(y, self.n_steps, training)

# # TODO check if trimming layer can be rewritten to take the target shape as input instead of a reference tensor
# recur_shape = tf.concat([tf.reshape(tf.constant(self.n_steps), (1,)), tf.shape(x)], axis=0)
# recur_x_shape_proxy = tf.broadcast_to(x[None,...], recur_shape)
# y = self.output_trimming([recur_x_shape_proxy, y])

# return y

# def get_config(self):
# config = super().get_config()

# config['output_channels'] = self.output_channels
# config['n_steps'] = self.n_steps
# config['n_levels'] = self.n_levels
# config['channels'] = self.channels
# config['spatial_dims'] = self.spatial_dims
# config['spacing'] = self.spacing
# config['activation'] = self.activation
# config['norm_groups'] = self.norm_groups

# return config
