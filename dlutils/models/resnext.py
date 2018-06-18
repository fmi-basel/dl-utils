from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range

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
from keras.layers import LeakyReLU
from keras.layers import Lambda
from keras.engine.topology import get_source_inputs

from keras import backend as K

from dlutils.models.utils import get_crop_shape
import numpy as np

import logging


def get_model_name(width, cardinality, n_levels, n_blocks, dropout, *args,
                   **kwargs):
    '''generate model name from its parameters.
    '''
    name = 'resnext-W{}-C{}-L{}-dec{}'.format(width, cardinality, n_levels,
                                              n_blocks)
    if dropout is not None:
        name += '-D{}'.format(dropout)

    logger = logging.getLogger(__name__)
    if args is not None:
        logger.warning('Unused parameters: {}'.format(args))
    if kwargs is not None:
        logger.warning('Unused parameters: {}'.format(kwargs))
    return name


class ResnextConstructor(object):
    def __init__(self,
                 width=1,
                 cardinality=32,
                 n_levels=4,
                 n_blocks=2,
                 dropout=None):
        '''collect parameters.
        '''
        self.width = width
        self.cardinality = cardinality
        self.n_levels = n_levels
        self.n_blocks = n_blocks
        self.dropout = dropout if dropout is not None else 0
        self.padding = 'same'

    def add_bn_activation(self, input_tensor):
        '''
        '''
        x = BatchNormalization(axis=self.bn_axis)(input_tensor)
        x = LeakyReLU()(x)
        return x

    def add_grouped_convolution(self, input_tensor, kernel_size, filters,
                                strides):
        '''
        '''
        if self.cardinality == 1:
            return Conv2D(
                input_tensor,
                kernel_size=kernel_size,
                strides=strides,
                padding=self.padding)

        assert filters % self.cardinality == 0, \
            'Number of filters must be divisible by cardinality'
        depth = filters // self.cardinality

        groups = []
        for group_idx in range(self.cardinality):
            group = Lambda(
                lambda z: z[:, :, :, group_idx * depth:group_idx * depth + depth]
            )(input_tensor)
            groups.append(
                Conv2D(
                    depth,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=self.padding)(group))

        return concatenate(groups)

    def add_residual_block(self,
                           input_tensor,
                           features_in,
                           features_out,
                           strides=(1, 1),
                           project=False):
        """
        """
        x = Conv2D(
            features_in,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=self.padding)(input_tensor)
        x = self.add_bn_activation(x)

        x = self.add_grouped_convolution(
            x, kernel_size=(3, 3), filters=features_in, strides=strides)
        x = self.add_bn_activation(x)

        if self.dropout > 0:
            x = Dropout(self.dropout)(x)

        x = Conv2D(
            features_out,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=self.padding)(x)

        x = BatchNormalization(axis=self.bn_axis)(x)

        shortcut = input_tensor
        if (strides != (1, 1) and strides != 1) or project:
            shortcut = Conv2D(
                features_out,
                kernel_size=(1, 1),
                strides=strides,
                padding=self.padding)(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = add([shortcut, x])
        x = LeakyReLU()(x)
        return x

    def construct_decoding_path(self, feature_levels):
        '''
        '''
        assert 1 <= self.n_blocks, \
            'n_blocks must be >= 1'
        n_features = feature_levels[-1].get_shape()[3].value

        x = feature_levels[-1]
        for level in range(2, len(feature_levels) + 1):

            n_features //= 2

            x = UpSampling2D(2)(x)
            y = feature_levels[-level]

            crop_shape = get_crop_shape(
                [y.get_shape()[idx].value for idx in range(1, 3)],
                [x.get_shape()[idx].value for idx in range(1, 3)])

            if np.any(np.asarray(crop_shape) > 0):
                x = Cropping2D(
                    cropping=crop_shape, name='UP{:02}_CRPX'.format(level))(x)

                crop_shape = get_crop_shape(
                    [x.get_shape()[idx].value for idx in range(1, 3)],
                    [y.get_shape()[idx].value for idx in range(1, 3)])

            if np.any(np.asarray(crop_shape) > 0):
                y = Cropping2D(
                    cropping=crop_shape, name='UP{:02}_CRPY'.format(level))(y)
                x = concatenate(
                    [x, y], axis=3, name='UP{:02}_CONC'.format(level))

            for block in range(self.n_blocks):
                x = self.add_residual_block(
                    x,
                    features_in=n_features // 2,
                    features_out=n_features,
                    strides=1,
                    project=block == 0)

        return x

    def construct_encoding_path(self, img_input):
        '''
        '''
        assert 2 <= self.n_levels <= 5, \
            'n_levels must be within {2, .., 5}'

        n_features = int(128 * self.width)

        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = Conv2D(
            n_features, (7, 7), strides=(2, 2), padding='same',
            name='conv1')(x)
        x = C1 = self.add_bn_activation(x)

        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for block in range(3):
            x = self.add_residual_block(
                x, n_features, 2 * n_features, strides=1, project=block == 0)
        C2 = x
        if self.n_levels <= 2:
            return [C1, C2]

        n_features *= 2
        for block in range(4):
            x = self.add_residual_block(
                x, n_features, 2 * n_features, strides=2 if block == 0 else 1)
        C3 = x
        if self.n_levels <= 3:
            return [C1, C2, C3]

        n_features *= 2
        for block in range(6):
            x = self.add_residual_block(
                x, n_features, 2 * n_features, strides=2 if block == 0 else 1)
        C4 = x
        if self.n_levels <= 4:
            return [C1, C2, C3, C4]

        n_features *= 2
        for block in range(3):
            x = self.add_residual_block(
                x, n_features, 2 * n_features, strides=2 if block == 0 else 1)
        C5 = x

        return [C1, C2, C3, C4, C5]

    def construct(self, input_shape=None, input_tensor=None, batch_size=None):
        '''
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
            self.bn_axis = 3
        else:
            self.bn_axis = 1

        # build encoding and decoding path.
        feature_levels = self.construct_encoding_path(img_input)
        outputs = self.construct_decoding_path([
            img_input,
        ] + feature_levels)

        # handle inputs.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        final_model = Model(
            inputs=inputs,
            outputs=outputs,
            name=get_model_name(self.width, self.cardinality,
                                self.n_levels, self.n_blocks,
                                self.dropout))

        return final_model


def ResneXtBase(input_shape=None, input_tensor=None, batch_size=None,
                **kwargs):
    return ResnextConstructor(**kwargs).construct(input_shape, input_tensor,
                                                  batch_size)
