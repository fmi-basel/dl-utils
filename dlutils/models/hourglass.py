'''hybrid model of resne(x)t and hourglass.

'''

# TODO:
# somehow change layer naming for 3D? 3x3 --> 3x3x3
# add options to downscale input
# intermediate supervision
# 3D dynamic padding
# groupdfilter for 3D

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
from keras.layers import Conv2D, Conv3D
from keras.layers import UpSampling2D, UpSampling3D
from keras.layers import MaxPooling2D, MaxPooling3D
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
                        dropout=0.,
                        cardinality=1,
                        dim_3D=False):
    '''
    '''
    # conv layer definitions.
    if dim_3D:
        Conv = Conv3D
        if cardinality != 1:
            raise NotImplementedError('3D version of GroupedConv not implemented')
        #GroupedConv = GroupedConv3D
    else:
        Conv = Conv2D
        GroupedConv = GroupedConv2D

    conv_kwargs = dict(activation=None, strides=(1), padding='same')

    assert int(cardinality) - cardinality <= 1e-6
    cardinality = int(cardinality)

    def block(input_tensor):
        '''
        '''
        x = input_tensor
        
        # 1x1 channels//2
        if with_bn:
            x = BatchNormalization(name=get_unique_layer_name('bn'))(x)
        x = Conv(
            n_features//2,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            **conv_kwargs)(x)
        x = Activation(activation, name=get_unique_layer_name(activation))(x)
        if dropout > 0.:
            x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
        
        # 3x3 channels//2
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
        if dropout > 0.:
            x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
        
        # 1x1 channels
        if with_bn:
            x = BatchNormalization(name=get_unique_layer_name('bn'))(x)
        x = Conv(
            n_features,
            kernel_size=(1),
            name=get_unique_layer_name('c1x1'),
            **conv_kwargs)(x)
        x = Activation(activation, name=get_unique_layer_name(activation))(x)
        if dropout > 0.:
            x = Dropout(dropout, name=get_unique_layer_name('do'))(x)
        
        x = add([input_tensor, x], name=get_unique_layer_name('add'))
        return x

    return block

def hourglass_block(n_features, n_levels, n_blocks_per_level, cardinality,
               with_bn, dropout, dim_3D=False):
    '''
    '''

    block_params = dict(cardinality=cardinality, with_bn=with_bn, 
                        dropout=dropout, dim_3D=dim_3D)

    # TODO Enable 3D
    base_block = bottleneck_conv_block
    
    if dim_3D:
        pooling = MaxPooling3D
        upsampling = UpSampling3D
    else:
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
            
            x = pooling(2, name=get_unique_layer_name('down2'))(x)

        # compressed representation
        for _ in range(n_blocks_per_level*3):
            x = base_block(n_features, **block_params)(x)

        # expanding path.
        for level in reversed(range(n_levels)):
            for _ in range(n_blocks_per_level):
                links[level] = base_block(n_features, **block_params)(links[level])
            
            x = upsampling(2, name=get_unique_layer_name('up2'))(x)
            x = add([x, links[level]], name=get_unique_layer_name('add'))

            for _ in range(n_blocks_per_level):
                x = base_block(n_features, **block_params)(x)
        return x

    return block
    
def hourglass_stack(n_stacks, n_features, n_levels, n_blocks_per_level, cardinality,
               with_bn, dropout, dim_3D=False):
    '''
    '''

    def block(x):
        '''
        '''

        for _ in range(n_stacks):
            residual = x
            x = hourglass_block(
                dropout=dropout,
                with_bn=with_bn,
                n_levels=n_levels,
                n_features=n_features,
                cardinality=cardinality,
                n_blocks_per_level=n_blocks_per_level,
                dim_3D=dim_3D)(x)
            x = add([x , residual], name=get_unique_layer_name('add'))
            
        return x

    return block

def input_block(n_features, n_levels, cardinality, with_bn, dropout, dim_3D=False):
    '''     
    TODO add option to use downscaling as in paper
    or
    keep original size and increase number of channel with conv
    '''
    
    block_params = dict(cardinality=cardinality, with_bn=with_bn, 
                        dropout=dropout, dim_3D=dim_3D)
    if dim_3D:
        ndim = 5
    else:
        ndim = 4
    
    def block(input_tensor):
        '''
        '''
        x=input_tensor
        
        if dim_3D: #only downscale x,y for anisotropic data, compromise for memory usage
            x = Conv3D(
                    n_features,
                    kernel_size=(3,7,7),
                    strides=(1,2,2),
                    name=get_unique_layer_name('c3x7x7'),
                    padding='same')(x)
            x = Activation('relu', name=get_unique_layer_name('relu'))(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
            x = MaxPooling3D((1,2,2), name=get_unique_layer_name('down2'))(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
        else:
            x = Conv2D(
                    n_features,
                    kernel_size=7,
                    strides=2,
                    name=get_unique_layer_name('c7x7'),
                    padding='same')(x)
            x = Activation('relu', name=get_unique_layer_name('relu'))(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
            x = MaxPooling2D(2, name=get_unique_layer_name('down2'))(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
            x = bottleneck_conv_block(n_features, **block_params)(x)
        
        x = DynamicPaddingLayer(factor=2**n_levels, ndim=ndim, name='dpad')(x)
        
        # alternatively don't downscale, simply change the number of channes to n_features
        # ~ x = Conv(
                # ~ n_features,
                # ~ kernel_size=(1),
                # ~ name=get_unique_layer_name('c1x1'),
                # ~ padding='same')(x)
        # ~ x = Activation('relu', name=get_unique_layer_name('relu'))(x)
        
        return x
    return block

def get_model_name(width, cardinality, n_stacks, n_levels, n_blocks, dropout, with_bn,
                   **kwargs):
    '''
    '''
    name = 'hourglass-W{}-C{}-S{}-L{}-B{}'.format(width, cardinality, n_stacks, n_levels,
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
                      n_stacks=1,
                      n_levels=5,
                      n_blocks=1):
    '''
    From paper:
    
    Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass 
    networks for human pose estimation." European Conference on 
    Computer Vision. Springer, Cham, 2016.
    '''
    n_features = int(width * 8)
    if len(input_shape) == 4:
        dim_3D = True
        ndim = 5
    else:
        dim_3D = False
        ndim = 4

    if cardinality < 1 or n_features % cardinality != 0:
        raise ValueError(
            'cardinality must be integer and a divisor of n_features.'
            ' ({} / {} != 0'.format(n_features, cardinality))

    # Assemble input
    # NOTE we use flexible sized inputs per default.
    if input_tensor is None:
        img_input = Input(
            # ~ batch_shape=(batch_size, ) + (None, None, input_shape[-1]),
            batch_shape=(batch_size, ) + 
                         tuple(None for _ in range(len(input_shape)-1)) +
                         (input_shape[-1],),
            name='input')
    else:
        img_input = input_tensor
    
    x = input_block(
        n_features=n_features,
        n_levels=n_levels,
        cardinality=cardinality,
        with_bn=with_bn,
        dropout=dropout,
        dim_3D=dim_3D)(img_input)
    
    # TODO implement intermediate module and recover outputs after each hourglass for supervision
    x = hourglass_stack(
        dropout=dropout,
        with_bn=with_bn,
        n_stacks=n_stacks,
        n_levels=n_levels,
        n_features=n_features,
        cardinality=cardinality,
        n_blocks_per_level=n_blocks,
        dim_3D=dim_3D)(x)
    
    # TODO create separate output_block(), with option for upscaling
    
    # TODO linear interp, upscale output to match labels size (or downscale label)
    if dim_3D:
        x = UpSampling3D((1,4,4), name=get_unique_layer_name('up1-4-4'))(x)
    else:
        x = UpSampling2D((4,4), name=get_unique_layer_name('up4-4'))(x)

    x = DynamicTrimmingLayer(ndim=ndim, name='dtrim')([img_input, x])
    if dim_3D:
        Conv = Conv3D
    else:
        Conv = Conv2D
        
    x = Conv(
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
            n_stacks=n_stacks,
            n_levels=n_levels,
            n_blocks=n_blocks,
            dropout=dropout,
            with_bn=with_bn))
