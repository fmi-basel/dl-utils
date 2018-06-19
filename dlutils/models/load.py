from keras.models import load_model as keras_load_model

from dlutils.layers.grouped_conv import GroupedConv2D

CUSTOM_LAYERS = {'GroupedConv2D': GroupedConv2D}


def load_model(*args, **kwargs):
    '''
    '''
    custom_objects = kwargs.pop('custom_objects', dict())
    custom_objects.update(CUSTOM_LAYERS)
    return keras_load_model(*args, custom_objects=custom_objects, **kwargs)
