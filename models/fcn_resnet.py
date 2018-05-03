from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Cropping2D
from keras.layers import ZeroPadding2D

from keras.applications.resnet50 import ResNet50, identity_block, conv_block

from models.utils import get_crop_shape
import numpy as np


def ResnetBase(input_shape,
               batch_size=None,
               weight_file=None,
               dropout=None,
               with_bn=False):
    '''Base constructor of resnet-based architectures.

    '''
    if input_shape is None:
        input_shape = (224, 224, 3)

    # reasonable layers to truncate: 10, 22, 40, 49
    truncate_layer = '40'

    input_tensor = Input(
        batch_shape=(batch_size, ) + input_shape, name='input')
    input_tensor = ZeroPadding2D((2, 2))(input_tensor)
    last_layer_name = 'activation_{}'.format(truncate_layer)

    # truncate imagenet-trained resnet50
    # TODO consider using pretrained weights.
    base_model = ResNet50(
        include_top=False, input_tensor=input_tensor, weights=None)
    base_model_out = base_model.get_layer(last_layer_name).output
    model = Model(inputs=base_model.input, outputs=base_model_out)
    no_features = base_model_out.get_shape()[3].value

    # Here goes the experimental decoding path
    n_decoding_blocks = 2
    n_levels = 4
    blocks = 'abcdefg'
    merge_layers = [
        'activation_40', 'activation_22', 'activation_10', 'activation_1',
        'input'
    ][-n_levels:]

    x = model.output
    for level in xrange(n_levels):
        no_features /= 2
        
        x = UpSampling2D(2)(x)

        y = model.get_layer(merge_layers[level]).output

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
                cropping=crop_shape,
                name='UP{:02}_CRPY'.format(level))(y)
        x = concatenate([x, y], axis=3, name='UP{:02}_CONC'.format(level))

        # TODO add shortcut.
        features = [no_features / 4, no_features / 4, no_features]

        x = conv_block(
            x, 3, features, stage=5 + level, block=blocks[0], strides=1)
        print x.shape
        for block in xrange(1, n_decoding_blocks):
            x = identity_block(
                x, 3, features, stage=5 + level, block=blocks[block])

    model = Model(inputs=base_model.input, outputs=x)

    if weight_file is not None:
        logger.info('Loading weights from :{}', weight_file)
        model.load_weights(weight_file)

    return model


if __name__ == '__main__':
    pass
