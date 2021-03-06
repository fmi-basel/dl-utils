import warnings
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, Conv3D, Layer
from tensorflow.keras.models import Model

from dlutils.blocks.aspp import aspp_block
from dlutils.layers.upsampling import BilinearUpSampling2D
from dlutils.layers.semi_conv import generate_coordinate_grid
from dlutils.layers.nd_layers import get_nd_semiconv, get_nd_conv


def add_fcn_output_layers(model,
                          names,
                          n_classes,
                          activation='sigmoid',
                          kernel_size=1):
    '''attaches fully-convolutional output layers to the
    last layer of the given model.

    '''
    last_layer = model.layers[-1].output
    if len(last_layer.shape) == 5:
        Conv = Conv3D
    else:
        Conv = Conv2D

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
            Conv(classes,
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
                           with_upscaling=False,
                           with_bn=False):
    '''attaches atrous spatial pyramid layers to the
    last layer of the given model.

    '''
    last_layer = model.layers[-1].output
    if len(last_layer.shape) == 5:
        Conv = Conv3D
    else:
        Conv = Conv2D

    if isinstance(names, list) and isinstance(n_classes, list):
        assert len(names) == len(n_classes)
    if not isinstance(activation, list):
        activation = len(names) * [
            activation,
        ]

    x = aspp_block(last_layer,
                   num_features=filters,
                   rate_scale=rate,
                   n_levels=n_levels,
                   with_bn=False)

    target_shape = model.input_shape

    outputs = []
    for name, classes, act in zip(names, n_classes, activation):
        y = Conv(classes,
                 kernel_size=1,
                 name=name + '-low' if with_upscaling else name,
                 activation=act,
                 padding='same')(x)
        if with_upscaling:
            y = BilinearUpSampling2D(target_shape=target_shape, name=name)(y)
        outputs.append(y)

    model = Model(model.inputs, outputs, name=model.name)
    return model


def add_instance_seg_heads(model,
                           n_classes,
                           spacing=1.,
                           kernel_size=1,
                           class_activation=True):
    '''Attaches a semi-convolutional embeddings layer and a semantic 
    classification convolutional layer to the given model.
    
    Args:
        model: model to which output layers are added
        n_classes: number semantic classes
        spacing: pixel/voxel spacing of the semi-conv embeddings
        kernel_size: kernel size of the appended layers
    '''

    spatial_dims = len(model.inputs[0].shape) - 2
    spacing = tuple(
        float(val) for val in np.broadcast_to(spacing, spatial_dims))

    if len(model.outputs) > 1:
        warnings.warn(
            'The model as {} outputs. #outputs > 1 will be ingnored'.format(
                len(model.outputs)))

    if not class_activation:
        activation = None
    elif n_classes > 1:
        activation = 'softmax'
    else:
        activation = 'sigmoid'

    last_layer = model.outputs[0]

    conv = get_nd_conv(spatial_dims)(n_classes,
                                     kernel_size=kernel_size,
                                     activation=activation,
                                     name='semantic_class',
                                     padding='same')

    semi_conv = get_nd_semiconv(spatial_dims)(spacing=spacing,
                                              kernel_size=kernel_size,
                                              name='embeddings',
                                              padding='same')

    semantic_class = conv(last_layer)
    embeddings = semi_conv(last_layer)

    return Model(inputs=model.inputs,
                 outputs=[embeddings, semantic_class],
                 name=model.name)


def split_output_into_instance_seg(model,
                                   n_classes,
                                   spacing=1.,
                                   class_activation=True):
    '''Splits the output of model into instance semi-conv embeddings and semantic class.
    
    Args:
        model: A base model that outputs at least n_classes + n_spatial-dimensions channels
        n_classes: number semantic classes
        spacing: pixel/voxel spacing of the semi-conv embeddings
    '''

    spatial_dims = len(model.inputs[0].shape) - 2
    spacing = tuple(
        float(val) for val in np.broadcast_to(spacing, spatial_dims))
    y_preds = model.outputs[0]

    if y_preds.shape[-1] < n_classes + spatial_dims:
        raise ValueError(
            'model has less than n_classes + n_spatial_dims channels: {} < {} + {}'
            .format(y_preds.shape[-1], n_classes, spatial_dims))

    vfield = y_preds[..., 0:spatial_dims]
    coords = generate_coordinate_grid(tf.shape(vfield), spatial_dims) * spacing
    embeddings = coords + vfield

    semantic_class = y_preds[..., spatial_dims:spatial_dims + n_classes]
    if class_activation:
        semantic_class = tf.nn.softmax(semantic_class, axis=-1)

    # rename outputs
    embeddings = Layer(name='embeddings')(embeddings)
    semantic_class = Layer(name='semantic_class')(semantic_class)

    return Model(inputs=model.inputs,
                 outputs=[embeddings, semantic_class],
                 name=model.name)
