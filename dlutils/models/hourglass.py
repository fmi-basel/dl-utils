'''hybrid model of resne(x)t and hourglass.

'''

# TODO:
# n_stacks
# separate input layer with options to downscale
# intermediate supervision
# 3D - filter selection, dynamic padding (patch size before that)
# droupout in bottlneck block

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range

from keras.engine import Input
from keras.engine import Model
from keras.engine.topology import get_source_inputs

from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import add
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout

from keras.backend import get_uid

from dlutils.layers.grouped_conv import GroupedConv2D
from dlutils.layers.padding import DynamicPaddingLayer
from dlutils.layers.padding import DynamicTrimmingLayer


def get_unique_layer_name(name):
    '''
    '''
    return '{}_{}'.format(name, get_uid(name))


def bottleneck_conv_block(n_features,
                        activation='relu',
                        with_bn=True,
                        cardinality=1):
    '''
    '''
    # conv layer definitions.
    Conv = Conv2D
    GroupedConv = GroupedConv2D

    conv_kwargs = dict(activation=None, strides=(1, 1), padding='same')

    assert int(cardinality) - cardinality <= 1e-6
    cardinality = int(cardinality)

    def block(input_tensor):
        '''
        '''
        x = input_tensor
        
        if with_bn:
            x = BatchNormalization(name=get_unique_layer_name('bn'))(x)
        x = Conv(
            n_features//2,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            **conv_kwargs)(x)
        x = Activation(activation, name=get_unique_layer_name(activation))(x)
        
        if with_bn:
            x = BatchNormalization(name=get_unique_layer_name('bn'))(x)
        if cardinality == 1:
            x = Conv(
                n_features//2,
                kernel_size=(3),
                name=get_unique_layer_name('c3x3'),
                **conv_kwargs)(x)
        else:
            x = GroupedConv(
                n_features//2,
                kernel_size=(3),
                cardinality=cardinality,
                name=get_unique_layer_name('g{:d}c3x3'.format(cardinality)),
                **conv_kwargs)(x)
        x = Activation(activation, name=get_unique_layer_name(activation))(x)
        
        if with_bn:
            x = BatchNormalization(name=get_unique_layer_name('bn'))(x)
        x = Conv(
            n_features,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            **conv_kwargs)(x)
        
        x = add([input_tensor, x], name=get_unique_layer_name('add'))
        x = Activation(activation, name=get_unique_layer_name(activation))(x)

        return x

    return block


def hourglass_block(n_features, n_levels, n_blocks_per_level, cardinality,
               with_bn, dropout):
    '''
    '''

    block_params = dict(cardinality=cardinality, with_bn=with_bn)

    # TODO Enable 3D
    base_block = bottleneck_conv_block
    pooling = MaxPooling2D
    upsampling = UpSampling2D

    def block(input_tensor):
        '''
        '''
        links = []
        x = input_tensor

        # top down
        for level in range(n_levels):
            links.append(x)
            for _ in range(n_blocks_per_level):
                x = base_block(n_features, **block_params)(x)
                if dropout > 0.:
                    x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
            
            x = pooling(2, name=get_unique_layer_name('down2'))(x)

        # compressed representation
        for _ in range(n_blocks_per_level*3):
            x = base_block(n_features, **block_params)(x)
            if dropout > 0.:
                x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
        
        # side connections
        for level in range(n_levels):
            for _ in range(n_blocks_per_level):
                links[level] = base_block(n_features, **block_params)(links[level])
                if dropout > 0.:
                    links[level] = Dropout(dropout, name=get_unique_layer_name('do'))(links[level])
        
        # expanding path.
        for level in reversed(range(n_levels)):
            x = upsampling(2, name=get_unique_layer_name('up2'))(x)
            x = add([x, links[level]], name=get_unique_layer_name('add'))

            for _ in range(n_blocks_per_level):
                x = base_block(n_features, **block_params)(x)
                if dropout > 0.:
                    x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
        return x

    return block


def get_model_name(width, cardinality, n_levels, n_blocks, dropout, with_bn,
                   **kwargs):
    '''
    '''
    name = 'hourglass-W{}-C{}-L{}-B{}'.format(width, cardinality, n_levels,
                                           n_blocks)
    if with_bn:
        name += '-BN'
    if dropout is not None:
        name += '-D{}'.format(dropout)
    return name


def GenericHourglassBase(input_shape=None,
                      input_tensor=None,
                      batch_size=None,
                      dropout=None,
                      with_bn=False,
                      width=1,
                      cardinality=1,
                      n_levels=5,
                      n_blocks=1):
    '''TODO doc
    
    Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass 
    networks for human pose estimation." European Conference on 
    Computer Vision. Springer, Cham, 2016.
    
    - 7x7 stride 2 : input channels --> base_features
    - res module
    - res module
    
    - hourglass
        - 1x1
        - split 1x1; 1x1(out features)1x1 base_features
        - add 2 above + residual
    - hourglass
    
    - 1x1 : base features --> ?
    - 1x1 : ? --> output features

    '''
    n_features = int(width * 64)

    if cardinality < 1 or n_features % cardinality != 0:
        raise ValueError(
            'cardinality must be integer and a divisor of n_features.'
            ' ({} / {} != 0'.format(n_features, cardinality))

    # Assemble input
    # NOTE we use flexible sized inputs per default.
    if input_tensor is None:
        img_input = Input(
            batch_shape=(batch_size, ) + (None, None, input_shape[-1]),
            name='input')
    else:
        img_input = input_tensor
    
    x = DynamicPaddingLayer(factor=2**n_levels, name='dpad')(img_input)
    
    # TODO move out, with option for downscaling
    x = Conv2D(
            n_features,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            padding='same')(x)
    x = Activation('relu', name=get_unique_layer_name('relu'))(x)
    
    # TODO add stack of hourglass
    x = hourglass_block(
        dropout=dropout,
        with_bn=with_bn,
        n_levels=n_levels,
        n_features=n_features,
        cardinality=cardinality,
        n_blocks_per_level=n_blocks)(x)

    x = DynamicTrimmingLayer(name='dtrim')([img_input, x])
    
    # TODO move out, with option for downscaling
    x = Conv2D(
            n_features,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            padding='same')(x)
    x = Activation('relu', name=get_unique_layer_name('relu'))(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(
        inputs=inputs,
        outputs=x,
        name=get_model_name(
            width=width,
            cardinality=cardinality,
            n_levels=n_levels,
            n_blocks=n_blocks,
            dropout=dropout,
            with_bn=with_bn))
