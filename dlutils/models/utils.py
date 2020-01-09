from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections.abc import Iterable


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
    elif name == 'rxunet':
        from dlutils.models.rxunet import get_model_name as rxunet_name
        return rxunet_name(**kwargs)

    else:
        raise NotImplementedError('Model {} not known!'.format(name))


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


def n_anisotropic_ops(spacing, base=2):
    '''Computes the number operations to skip along each axis to approximate an isotropic field of view.
    '''
    def _np_log_n(x, base):
        return np.log(x) / np.log(base)

    spacing = np.array(spacing, ndmin=1)
    normalized_spacing = spacing / spacing.min()

    return np.floor(_np_log_n(normalized_spacing, base)).astype(int)


def anisotropic_kernel_size(spacing, level, n_levels, base_size=2):
    '''
    Computes a level dependent kernel size needed to roughly approximate
    an isotropic field of view.
    
    Notes:
    Currently only implemented to choose between 1 or 2, i.e. intended
    for pooling/no pooling
    
    For examples for a doubled z spacing: (0.5,0.25,0.25) the pooling size 
    should be (1,2,2) once (e.g. first layer) and (2,2,2) for the rest
    '''

    if base_size != 2:
        raise NotImplementedError(
            'base kernel size != 2 not implemented, base_size={} found'.format(
                base_size))

    if not isinstance(spacing, Iterable):
        return base_size

    not_pooling_interval = np.ceil(
        n_levels / (n_anisotropic_ops(spacing, 2) + 1e-12)).astype(int)
    # shift to start with "not pooling"
    shift = min(not_pooling_interval.min(), n_levels)

    return tuple(2 - ((level + shift) % not_pooling_interval == 0))
