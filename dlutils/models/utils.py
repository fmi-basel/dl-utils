from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Convolution2D
from keras.engine import Model


def get_model_name(name, **kwargs):
    '''generate a model name describing the architecture.

    '''
    if name == 'resnet':
        from dlutils.models.fcn_resnet import get_model_name as resnet_name
        return resnet_name(**kwargs)
    elif name == 'unet':
        from dlutils.models.unet import get_model_name as unet_name
        return unet_name(**kwargs)
    elif name == 'resnext':
        from dlutils.models.resnext import get_model_name as resnext_name
        return resnext_name(**kwargs)
    else:
        raise NotImplementedError('Model {} not known!'.format(name))


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
                classes, kernel_size=kernel_size, name=name,
                activation=act)(last_layer))
    model = Model(model.inputs, outputs, name=model.name)
    return model


def get_crop_shape(x_shape, y_shape):
    '''determine crop delta for a concatenation.

    NOTE Assumes that y is larger than x.
    '''
    assert len(x_shape) == len(y_shape)
    assert len(x_shape) >= 2
    shape = []

    for xx, yy in zip(x_shape, y_shape):
        delta = yy - xx
        if delta < 0:
            delta = 0
        if delta % 2 == 1:
            shape.append((int(delta / 2), int(delta / 2) + 1))
        else:
            shape.append((int(delta / 2), int(delta / 2)))
    return shape


def get_batch_size(model):
    '''
    '''
    return model.input_shape[0]


def get_patch_size(model):
    '''
    '''
    return model.input_shape[1:-1]


def get_input_channels(model):
    '''
    '''
    return model.input_shape[-1]
