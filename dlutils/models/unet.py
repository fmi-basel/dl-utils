from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Deconvolution2D
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Cropping2D
from keras.layers import BatchNormalization
from keras.engine.topology import get_source_inputs

import logging


def get_model_name(cardinality, n_levels, dropout, with_bn, *args, **kwargs):
    name = 'UNet-{}-{}'.format(cardinality, n_levels)
    if with_bn:
        name += '-BN'
    if dropout is not None:
        name += '-D{}'.format(dropout)

    logger = logging.getLogger(__name__)
    if args is not None:
        logger.warning('Unused parameters: {}'.format(args))
    if kwargs is not None:
        logger.warning('Unused parameters: {}'.format(kwargs))
    return name


def UnetBase(input_shape=None,
             input_tensor=None,
             batch_size=None,
             weight_file=None,
             dropout=None,
             with_bn=False,
             cardinality=1,
             n_levels=5):
    '''construct a basic U-Net without any output layer.

    TODO describe parameters.

    '''
    logger = logging.getLogger(__name__)

    conv_params = dict(kernel_size=(3, 3), padding='same', activation='relu')

    if dropout == 0.:  # disable dropout if zero.
        dropout = None

    if dropout is not None:
        assert 0. < dropout < 1.0, \
            'Dropout has to be within 0. and 1.'

    assert 1 / 64. < cardinality, \
        'cardinality has to be larger than 1 / 64'

    assert 1 <= n_levels, \
        'n_levels has to be larger than 0'

    def contracting_block(x, n_features, level):
        '''constrcut the default block of a UNet.
        '''
        base_name = 'CB_L{:02}'.format(level)
        x = Convolution2D(n_features, name=base_name + '_C0', **conv_params)(x)
        if with_bn:
            x = BatchNormalization(name=base_name + '_BN0')(x)
        x = Convolution2D(n_features, name=base_name + '_C1', **conv_params)(x)
        if with_bn:
            x = BatchNormalization(name=base_name + '_BN1')(x)

        if dropout is not None:
            x = Dropout(dropout, name=base_name + '_DROP')(x)
        return x

    def get_crop_shape(x_shape, y_shape):
        '''determine crop delta for a concatenation.

        NOTE Assumes that y is larger than x.
        '''
        assert len(x_shape) == len(y_shape)
        assert len(x_shape) >= 2
        shape = []

        for xx, yy in zip(x_shape, y_shape):
            delta = yy - xx
            if delta % 2 == 1:
                shape.append((int(delta / 2), int(delta / 2) + 1))
            else:
                shape.append((int(delta / 2), int(delta / 2)))
        return shape

    def expanding_block(x, y, n_features, level):
        '''construct the default expanding block
        '''
        base_name = 'EB_L{:02}'.format(level)

        x_shape = (2 * x.get_shape()[1].value, 2 * x.get_shape()[2].value)
        x = Deconvolution2D(
            n_features,
            kernel_size=2,
            strides=2,
            name=base_name + '_DC',
            padding=conv_params['padding'])(x)
        if with_bn:
            x = BatchNormalization(name=base_name + '_BN0')(x)

        # Ensure both channels have the same shape
        x = Cropping2D(
            cropping=get_crop_shape(
                [y.get_shape()[idx].value for idx in xrange(1, 3)], x_shape),
            name=base_name + '_CRP')(x)

        # Concatenate with corresponding level
        x = concatenate([x, y], axis=3, name=base_name + '_CONC')
        for ii in xrange(1, 3):
            x = Convolution2D(
                n_features, name=base_name + '_C{}'.format(ii),
                **conv_params)(x)
            if with_bn:
                x = BatchNormalization(name=base_name + '_BN{}'.format(ii))(x)

        return x

    # Assemble input
    if input_tensor is None:
        img_input = Input(
            batch_shape=(batch_size, ) + input_shape, name='input')
    else:
        img_input = input_tensor

    # width of first feature map.
    n_features_basic = int(64 * cardinality)

    x = img_input
    cb_out = [
        None,
    ] * n_levels

    # build contracting path
    for level, n_features in ((level, n_features_basic * (2**level))
                              for level in xrange(n_levels)):

        cb_out[level] = contracting_block(
            x, n_features=n_features, level=level)
        x = cb_out[level]

        if level < n_levels - 1:  # dont add pooling for the very lowest layer!
            x = MaxPooling2D(
                padding='same', name='CB_L{:02}_MP'.format(level))(x)

    # input layer to expanding path
    x = cb_out[-1]

    # build expanding path
    for level, n_features in ((level, n_features_basic * (2**level))
                              for level in xrange(n_levels - 2, -1, -1)):
        x = expanding_block(
            x, cb_out[level], n_features=n_features, level=level)

    # handle the case where input_tensor are other layers.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(
        inputs,
        x,
        name=get_model_name(cardinality, n_levels, dropout, with_bn))

    if weight_file is not None:
        # TODO replace with logger
        logger.info('Loading weights from :{}'.format(weight_file))
        model.load_weights(weight_file)

    return model


if __name__ == '__main__':
    pass
