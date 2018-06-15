from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Cropping2D
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import add
from keras.layers import MaxPooling2D
from keras.engine.topology import get_source_inputs

from keras import backend as K

from dlutils.models.utils import get_crop_shape
import numpy as np

import logging


def get_model_name(cardinality=1,
                   n_levels=4,
                   n_blocks=2,
                   dropout_rate=None,
                   *args,
                   **kwargs):
    '''generate model name from its parameters.
    '''
    name = 'resnet-{}-{}-dec{}'.format(cardinality, n_levels, n_blocks)
    if dropout_rate is not None:
        name += '-D{}'.format(dropout_rate)

    logger = logging.getLogger(__name__)
    if args is not None:
        logger.warning('Unused parameters: {}'.format(args))
    if kwargs is not None:
        logger.warning('Unused parameters: {}'.format(kwargs))
    return name


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   dropout_rate=0):
    """The identity block is the block that has no conv layer at shortcut.

    NOTE This is the original block taken from the keras implementation.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # placement inspired by "Wide Resnets" Paper
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    NOTE This is the original block taken from the keras implementation.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (1, 1), strides=strides,
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(
        filters3, (1, 1), strides=strides,
        name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def _construct_resnet(img_input,
                      bn_axis,
                      n_levels,
                      cardinality=1,
                      dropout_rate=0):
    '''
    '''
    assert 0 < cardinality
    assert 2 <= n_levels <= 5, \
        'n_levels must be within {2, .., 5}'

    n_features = int(cardinality * 64)

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(
        n_features, (7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = C1 = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(
        x,
        3, [n_features, n_features, n_features * 4],
        stage=2,
        block='a',
        strides=(1, 1))
    x = identity_block(
        x,
        3, [n_features, n_features, n_features * 4],
        stage=2,
        block='b',
        dropout_rate=dropout_rate)
    x = C2 = identity_block(
        x,
        3, [n_features, n_features, n_features * 4],
        stage=2,
        block='c',
        dropout_rate=dropout_rate)

    if n_levels <= 2:
        return [C1, C2]

    n_features *= 2
    x = conv_block(
        x, 3, [n_features, n_features, n_features * 4], stage=3, block='a')
    for ii in xrange(1, 4):
        x = identity_block(
            x,
            3, [n_features, n_features, n_features * 4],
            stage=3,
            block=chr(98 + ii),
            dropout_rate=dropout_rate)
    C3 = x

    if n_levels <= 3:
        return [C1, C2, C3]

    n_features *= 2
    x = conv_block(
        x, 3, [n_features, n_features, n_features * 4], stage=4, block='a')
    for ii in xrange(1, 6):
        x = identity_block(
            x,
            3, [n_features, n_features, n_features * 4],
            stage=4,
            block=chr(98 + ii),
            dropout_rate=dropout_rate)
    C4 = x

    if n_levels <= 4:
        return [C1, C2, C3, C4]

    n_features *= 2
    x = conv_block(
        x, 3, [n_features, n_features, n_features * 4], stage=5, block='a')
    for ii in xrange(1, 3):
        x = identity_block(
            x,
            3, [n_features, n_features, n_features * 4],
            stage=5,
            block=chr(98 + ii),
            dropout_rate=dropout_rate)
    C5 = x

    return [C1, C2, C3, C4, C5]


def _construct_decoding_path(feature_levels, n_blocks, dropout_rate=0):
    '''construct the base decoding block.

    '''
    assert 1 <= n_blocks, \
        'n_blocks must be >= 1'
    n_features = feature_levels[-1].get_shape()[3].value

    x = feature_levels[-1]
    for level in xrange(2, len(feature_levels) + 1):

        n_features /= 2
        features = [n_features / 4, n_features / 4, n_features]

        x = UpSampling2D(2)(x)
        y = feature_levels[-level]

        crop_shape = get_crop_shape(
            [y.get_shape()[idx].value for idx in xrange(1, 3)],
            [x.get_shape()[idx].value for idx in xrange(1, 3)])

        if np.any(crop_shape > 0):
            x = Cropping2D(
                cropping=crop_shape, name='UP{:02}_CRPX'.format(level))(x)

        crop_shape = get_crop_shape(
            [x.get_shape()[idx].value for idx in xrange(1, 3)],
            [y.get_shape()[idx].value for idx in xrange(1, 3)])

        if np.any(crop_shape > 0):
            y = Cropping2D(
                cropping=crop_shape, name='UP{:02}_CRPY'.format(level))(y)
        x = concatenate([x, y], axis=3, name='UP{:02}_CONC'.format(level))

        x = conv_block(
            x, 3, features, stage=5 + level, block=chr(98), strides=1)

        for block in xrange(1, n_blocks):
            x = identity_block(
                x,
                3,
                features,
                stage=5 + level,
                block=chr(98 + block),
                dropout_rate=dropout_rate)

    return x


def ResnetBase(input_shape=None,
               input_tensor=None,
               batch_size=None,
               weight_file=None,
               dropout=None,
               with_bn=False,
               cardinality=1,
               n_levels=4,
               n_blocks=2):
    '''base constructor for resnet-based architectures.

    '''
    # input handling
    if input_tensor is None:
        img_input = Input(shape=input_shape, name='input')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # build encoding and decoding path.
    feature_levels = _construct_resnet(
        img_input,
        bn_axis,
        cardinality=cardinality,
        n_levels=n_levels,
        dropout_rate=dropout)
    outputs = _construct_decoding_path(
        [
            img_input,
        ] + feature_levels, n_blocks, dropout_rate=dropout)

    # handle inputs.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    final_model = Model(
        inputs=inputs,
        outputs=outputs,
        name=get_model_name(cardinality, n_levels, n_blocks, dropout))

    if weight_file is not None:
        final_model.load_weights(weight_file)

    return final_model


if __name__ == '__main__':
    pass
