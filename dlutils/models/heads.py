from keras.layers import Convolution2D
from keras.engine import Model

from dlutils.models.deeplab import aspp_block
from dlutils.layers.upsampling import BilinearUpSampling2D


def add_fcn_output_layers(model,
                          names,
                          n_classes,
                          activation='sigmoid',
                          kernel_size=1):
    '''attaches fully-convolutional output layers to the
    last layer of the given model.

    '''
    last_layer = model.layers[-1].output

    if isinstance(names, list) and isinstance(n_classes, list):
        assert len(names) == len(n_classes)
    if not isinstance(activation, list):
        activation = len(names) * [
            activation,
        ]
    # TODO handle other cases

    outputs = []
    for name, classes, act in zip(names, n_classes, activation):
        outputs.append(
            Convolution2D(
                classes,
                kernel_size=kernel_size,
                name=name,
                activation=act,
                padding='same')(last_layer))
    model = Model(model.inputs, outputs, name=model.name)
    return model


def add_aspp_output_layers(model,
                           names,
                           n_classes,
                           filters,
                           rate=1,
                           n_levels=3,
                           activation='sigmoid',
                           with_upscaling=False):
    '''attaches atrous spatial pyramid layers to the
    last layer of the given model.

    '''
    last_layer = model.layers[-1].output

    if isinstance(names, list) and isinstance(n_classes, list):
        assert len(names) == len(n_classes)
    if not isinstance(activation, list):
        activation = len(names) * [
            activation,
        ]
    # TODO handle other cases

    x = aspp_block(
        last_layer, num_filters=filters, rate_scale=rate, n_levels=n_levels)

    target_shape = model.input_shape

    outputs = []
    for name, classes, act in zip(names, n_classes, activation):
        y = Convolution2D(
            classes,
            kernel_size=1,
            name=name + '-low' if with_upscaling else name,
            activation=act,
            padding='same')(x)
        if with_upscaling:
            y = BilinearUpSampling2D(target_shape=target_shape, name=name)(y)
        outputs.append(y)

    model = Model(model.inputs, outputs, name=model.name)
    return model
