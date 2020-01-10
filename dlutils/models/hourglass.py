'''Loose implementation of hourglass network.

Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass
    networks for human pose estimation." European Conference on
    Computer Vision. Springer, Cham, 2016.

Notes:
- skip branches are concatenated instead of summed (i.e. like in U-net)'''

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Input, LeakyReLU
from tensorflow_addons.layers import GroupNormalization

from dlutils.layers.nd_layers import get_nd_conv, get_nd_maxpooling, get_nd_upsampling
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from dlutils.models.utils import anisotropic_kernel_size, n_anisotropic_ops

Activation = LeakyReLU


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


def bottleneck_conv_block(channels=32, spatial_dims=2, norm_groups=4):
    '''
    Notes
    -----
    pre-activation to keep the residual path clear as described in:
    
    HE, Kaiming, et al. Identity mappings in deep residual networks.
    In: European conference on computer vision. Springer, Cham, 2016.
    S. 630-645.
    '''

    Conv = get_nd_conv(spatial_dims)

    conv_in = Conv(channels, kernel_size=1, padding='same')

    seq = _stack_layers([
        Activation(),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels // 2, kernel_size=1, padding='same'),
        Activation(),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels // 2, kernel_size=3, padding='same'),
        Activation(),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels, kernel_size=1, padding='same'),
    ])

    def block(x):
        if x.shape[-1] != channels:
            # if needed, brings the number of input channels to the same as output channels
            # not strictly a bottleneck anymore
            x = Activation()(conv_in(x))

        return x + seq(x)

    return block


def hourglass_block(
        n_levels=4,
        channels=32,
        channels_growth=2,
        spatial_dims=2,
        spacing=1,
        norm_groups=4,
):
    Conv = get_nd_conv(spatial_dims)
    MaxPool = get_nd_maxpooling(spatial_dims)
    UpSampling = get_nd_upsampling(spatial_dims)

    # must be divisible by 2*norm_groups because of channels//2 in bottleneck block
    level_channels = [
        int((channels * channels_growth**l) // (2 * norm_groups) * 2 *
            norm_groups) for l in range(n_levels)
    ]

    # define layers ################################################
    # conv block for down path
    downs = [
        bottleneck_conv_block(level_channels[l], spatial_dims, norm_groups)
        for l in range(n_levels)
    ]

    # conv blocks for residual/skip paths
    skips = [
        bottleneck_conv_block(level_channels[l], spatial_dims, norm_groups)
        for l in range(n_levels)
    ]

    # conv block for middle layer
    mid = bottleneck_conv_block(level_channels[-1], spatial_dims, norm_groups)

    # conv blocks for up path
    ups = [
        _stack_layers([
            Conv(level_channels[l], kernel_size=1,
                 padding='same'),  # reduce concatenated channels by half
            LeakyReLU(),
            bottleneck_conv_block(level_channels[l], spatial_dims, norm_groups)
        ]) for l in range(n_levels - 1)
    ]

    pools = [
        MaxPool(anisotropic_kernel_size(spacing, l, n_levels))
        for l in range(1, n_levels)
    ]
    upsamples = [
        UpSampling(anisotropic_kernel_size(spacing, l, n_levels))
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
        x = tf.identity(x, name="middle")

        # up
        for idx, (level_var, up, upsample) in enumerate(
                zip(level_outputs[::-1], ups[::-1], upsamples[::-1])):

            x = upsample(x)
            # name the concatenation layers for easy access: level 0: last up layer (back to image resolution)
            x = tf.concat([x, level_var],
                          axis=-1,
                          name='concat_l{}'.format(n_levels - 2 - idx))
            x = up(x)

        return x

    return block


def single_hourglass(output_channels,
                     n_levels=4,
                     channels=32,
                     channels_growth=2,
                     spatial_dims=2,
                     spacing=1,
                     norm_groups=4):
    '''Combines an hourglass block with input/output blocks to 
    increase/decrease the number of channels.
    
    Notes:
    Expects the input to be already padded)
    '''

    Conv = get_nd_conv(spatial_dims)

    hglass = _stack_layers([
        Conv(channels, kernel_size=1, padding='same'),
        bottleneck_conv_block(channels, spatial_dims, norm_groups),
        hourglass_block(n_levels, channels, channels_growth, spatial_dims,
                        spacing, norm_groups),
        bottleneck_conv_block(channels, spatial_dims, norm_groups),
        Activation(),
        GroupNormalization(groups=norm_groups, axis=-1),
        Conv(channels, kernel_size=3, padding='same'),
        Activation(),
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
            # NOTE: does not serialize model properly, channel size lost when reloading
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
        deltas = tf.TensorArray(tf.float32, size=n_steps)

        def cond(i, outputs, deltas, state):
            return tf.less(i, n_steps)

        def body(i, outputs, deltas, state):
            delta = recurrent_block(tf.concat([x, state], axis=-1))
            state = state + delta

            deltas = deltas.write(i, delta)
            outputs = outputs.write(i, state)

            i = tf.add(i, 1)
            return (i, outputs, deltas, state)

        i, outputs, deltas, state = tf.while_loop(cond,
                                                  body,
                                                  [i, outputs, deltas, state],
                                                  swap_memory=True)

        return outputs.stack(), deltas.stack()

    return block


def get_model_name(n_levels,
                   channels,
                   channels_growth,
                   output_channels,
                   spatial_dims,
                   spacing=1,
                   **kwargs):
    '''
    '''
    spacing = np.broadcast_to(np.array(spacing), spatial_dims)

    name = 'hourglass-L{}-C{}-G{}-O{}-D{}'.format(n_levels, channels,
                                                  channels_growth,
                                                  output_channels,
                                                  spatial_dims)
    if spatial_dims == 3:
        name += '-S{:.2g}_{:.2g}_{:.2g}'.format(*spacing)
    return name


def GenericRecurrentHourglassBase(input_shape,
                                  output_channels,
                                  external_init_state=False,
                                  default_n_steps=3,
                                  n_levels=4,
                                  channels=32,
                                  channels_growth=2,
                                  spatial_dims=2,
                                  spacing=1,
                                  norm_groups=4):
    '''Constructs a hourglass/U-net like network
    
    Args:
        input_shape: at least the number of channels should be defined
        output_channels: number of output channels in the last layer
        external_init_state: if True, the network will take a second input 
            to initialize the recurrent state (default tf.zeros)
        default_n_steps: number of times the network is recurrently applied
        n_levels: ~ depth of the network, also corresponds to the number
            of pooling layers
        channels: number of channels of the first layer
        channels_growth: channel multiplication factor for subsequent layers.
            1 - fixed number of channels as in hourglass, 
            2 - double each layer as in U-net
            Note that the exact number of channels will be floored to be 
            divisible by 2*norm_groups
        spatial_dims: number of spatial dimensions, 2 or 3
        spacing: pixel/voxel's size
        norm_groups: number of normalization groups of groupnorm layers
        
    Notes:
    if anisotropic spacing is provided, the pooling size will be 2 along
    the dimension with the smallest spacing and 1 or 2 along the other 
    dimension(s) depending on the layer, so as to approximate an isotropic 
    field of view.
    
    For examples for a doubled z spacing: (0.5,0.25,0.25) the pooling size
    will be (1,2,2) for the first layer and (2,2,2) for the remaining ones
    '''

    n_not_pooling = n_anisotropic_ops(spacing, base=2)
    factor = 2**np.maximum(0, ((n_levels - 1) - n_not_pooling))
    input_padding = DynamicPaddingLayer(factor=factor, ndim=spatial_dims + 2)

    hglass = single_hourglass(output_channels, n_levels, channels,
                              channels_growth, spatial_dims, spacing,
                              norm_groups)
    r_hglass = delta_loop(output_channels, hglass, default_n_steps)
    output_trimming = DynamicTrimmingLayer(ndim=spatial_dims + 2)

    def trim_stack(x, stack):
        '''Trims a stack of intermediates outputs one by one'''
        tensors = [output_trimming([x, sl]) for sl in stack]
        return tf.stack(tensors)

    # TODO is it possible to change n_steps and initial state at run time? (i.e. have optional inputs/attributes)
    x = Input(shape=input_shape)
    y = input_padding(x)

    if external_init_state is False:
        inputs = [x]
        y, deltas = r_hglass(y)

    else:
        state = Input(shape=input_shape[:-1] + (output_channels, ))
        inputs = [x, state]

        padded_state = input_padding(state)
        y, deltas = r_hglass(y, padded_state)

    y = trim_stack(x, y)
    deltas = trim_stack(x, deltas)

    return Model(inputs=inputs,
                 outputs=[y, deltas],
                 name=get_model_name(n_levels, channels, channels_growth,
                                     output_channels, spatial_dims, spacing))
