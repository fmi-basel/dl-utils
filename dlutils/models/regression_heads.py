from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Convolution2D
from keras.engine import Model

from builtins import range


def add_bb_regression_heads(model,
                            n_classes,
                            kernel_size=3,
                            n_filters=256,
                            with_size=False):
    '''attach fully-convolutional output layers for bounding box
    regression to the last layer of the given model.

    Parameters
    ----------
    model : Keras.model.Model
        Base model to be extended.
    n_classes : int
        Number of classes for the anchor classification.
    kernel_size : int
        Size of convolutional kernels.
    n_filters : int
        Number of features in each path.
    with_size : bool
        Whether to regress both bounding box offset and size.

    Returns
    -------
    extended_model : Keras.model.Model
        Model extended with bounding box regression heads.


    The default values follow the design of RetinaNet [1]:

    [1] Lin et al. Focal loss for dense object detection, arxiv 2018.

    '''
    last_layer = model.layers[-1].output

    outputs = []

    # anchor classification head.
    x = last_layer
    for i in range(2):
        x = Convolution2D(
            n_filters,
            kernel_size=kernel_size,
            name='anchor_c3x3_{}'.format(i),
            activation='relu')(x)

    outputs.append(
        Convolution2D(
            n_classes,
            kernel_size=kernel_size,
            name='anchor_pred',
            activation='sigmoid')(x))

    # offset regression head.
    x = last_layer
    for i in range(2):
        x = Convolution2D(
            n_filters,
            kernel_size=kernel_size,
            name='offset_c3x3_{}'.format(i),
            activation='relu')(x)

    regression_channels = 4 if with_size else 2
    outputs.append(
        Convolution2D(
            regression_channels,
            kernel_size=kernel_size,
            name='offset_pred',
            activation='linear')(x))  # linear activation for offset

    model = Model(model.inputs, outputs, name=model.name)
    return model
