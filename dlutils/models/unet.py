from warnings import warn

import tensorflow as tf
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization

from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from dlutils.layers.nd_layers import get_nd_conv
from dlutils.layers.nd_layers import get_nd_maxpooling
from dlutils.layers.nd_layers import get_nd_upsampling


def get_model_name(width, n_levels, dropout, with_bn, *args, **kwargs):
    '''
    '''
    warn(
        'get_model_name(..) for unet is deprecated. '
        'Please use UnetBuilder.get_model_name(..) in the future.',
        DeprecationWarning)
    name = 'UNet-{}-{}'.format(width, n_levels)
    if with_bn:
        name += '-BN'
    if dropout is not None:
        name += '-D{}'.format(dropout)
    return name


def _abbreviate(class_name: str):
    '''creates an abbreviation for a class name from its capitalization.

    For example:

    _abbreviate(LayerNormalization.__name__) => 'LN'
    '''
    return ''.join(filter(lambda char: char.isupper(), class_name))


class UnetBuilder:
    '''builder class for vanilla unets. Customizable by passing constructors
    for convolution, upsampling, downsampling and normalization layers as well
    as conv_params and norm_params.

    Modifications beyond that are intended to be done through specialization
    of the class. For example:

    class UnetWithWeirdBlocksBuilder(UnetBuilder):
        def add_single_block(self, input_tensor, filters, **kwargs):
            ...  # define the block
            return block_out_tensor

    See also GenericUnetBase for a use case.

    '''
    def __init__(self,
                 conv_layer,
                 upsampling_layer,
                 downsampling_layer,
                 n_levels,
                 n_blocks,
                 base_features,
                 norm_layer=None,
                 activation_layer=tf.keras.layers.LeakyReLU,
                 conv_params=None,
                 norm_params=None):
        '''
        '''
        self.n_levels = n_levels
        self.n_blocks = n_blocks
        self.base_features = base_features

        self.conv_params = {
            'activation': 'linear',  # activation is added separately.
            'padding': 'same',
            'kernel_size': 3
        }
        if conv_params is not None:
            self.conv_params.update(conv_params)

        self.norm_params = {'axis': -1}
        if norm_params is not None:
            self.norm_params.update(norm_params)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.upsampling_layer = upsampling_layer
        self.downsampling_layer = downsampling_layer
        self.activation_layer = activation_layer

    def get_model_name(self):
        '''
        '''
        name = 'unet-W{}-L{}-B{}'.format(self.base_features, self.n_levels,
                                         self.n_blocks)
        if self.norm_layer is not None:
            name += '-' + _abbreviate(self.norm_layer.__name__)
        return name

    def add_single_block(self, input_tensor, filters, **kwargs):
        '''
        '''
        with tf.name_scope('Block'):
            out_tensor = self.conv_layer(filters=filters,
                                         **self.conv_params)(input_tensor)
            if self.norm_layer is not None:
                out_tensor = self.norm_layer(**self.norm_params)(out_tensor)
            out_tensor = self.activation_layer()(out_tensor)
        return out_tensor

    def add_downsampling(self, input_tensor):
        '''
        '''
        return self.downsampling_layer()(input_tensor)

    def add_upsampling(self, input_tensor):
        '''
        '''
        return self.upsampling_layer()(input_tensor)

    def add_combiner(self, input_tensors):
        '''
        '''
        return tf.keras.layers.Concatenate()(input_tensors)

    def features_of_level(self, level):
        '''
        '''
        return self.base_features * 2**level

    def build_unet_block(self, input_tensor):
        '''
        '''
        skips = {}
        x = input_tensor

        # downstream
        for level in range(self.n_levels - 1):
            for _ in range(self.n_blocks):
                x = self.add_single_block(
                    x, filters=self.features_of_level(level))
            skips[level] = x
            x = self.add_downsampling(x)

        # bottom
        for _ in range(self.n_blocks):
            x = self.add_single_block(
                x, filters=self.features_of_level(self.n_levels - 1))

        # upstream
        for level in reversed(range(self.n_levels - 1)):

            x = self.add_upsampling(x)
            x = self.add_combiner([x, skips[level]])

            for _ in range(self.n_blocks):
                x = self.add_single_block(
                    x, filters=self.features_of_level(level))

        return x


def GenericUnetBase(input_shape=None,
                    input_tensor=None,
                    batch_size=None,
                    with_bn=False,
                    width=1,
                    n_levels=5,
                    n_blocks=2):
    '''UNet constructor for 2D and 3D.

    Parameters
    ----------
    input_shape: tuple or None
        Expected shape of the input tensor. Either input_shape or input_tensor
        have to be defined.
    input_tensor: Tensor or None
        Input tensor. Either input_shape or input_tensor have to be defined.
    batch_size: int or None
        Expected batch size.
    with_bn: bool
        If True, instantiate model with BatchNormalization.
    width: float
        Scales the number of features used in all layers. width=1.0 corresponds
        to the default of 64 features in the first level.
    n_levels: int
        Number of levels in the unet.
    n_blocks: int
        Number of blocks in each level.

    Notes
    -----
    * All dimensions are treated identically.
    * If you need more customization of the architecture, you might be
      interested in specializing UnetBuilder.

    '''
    if input_tensor is None and input_shape is None:
        raise ValueError('Either input_shape or input_tensor must be given!')

    if input_tensor is None:
        img_input = Input(batch_shape=(batch_size, ) + input_shape,
                          name='input')
    else:
        img_input = input_tensor

    ORIGINAL_FEATURES = 64

    # dont count batch and channel dimension.
    spatial_ndim = len(img_input.shape) - 2

    # determine normalization
    norm_layer = BatchNormalization if with_bn else None

    builder = UnetBuilder(conv_layer=get_nd_conv(spatial_ndim),
                          downsampling_layer=get_nd_maxpooling(spatial_ndim),
                          upsampling_layer=get_nd_upsampling(spatial_ndim),
                          norm_layer=norm_layer,
                          n_levels=n_levels,
                          n_blocks=n_blocks,
                          base_features=int(width * ORIGINAL_FEATURES))

    # add padding...
    padding_factor = 2**n_levels
    x = DynamicPaddingLayer(factor=padding_factor,
                            ndim=spatial_ndim + 2,
                            name='dpad')(img_input)

    # construct unet.
    x = builder.build_unet_block(x)

    # ...and remove padding.
    x = DynamicTrimmingLayer(ndim=spatial_ndim + 2,
                             name='dtrim')([img_input, x])

    inputs = (get_source_inputs(input_tensor)
              if input_tensor is not None else img_input)
    return Model(inputs=inputs, outputs=x, name=builder.get_model_name())


# Alias for backward compatibility
UnetBase = GenericUnetBase
