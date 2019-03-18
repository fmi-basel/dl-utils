from __future__ import absolute_import


def construct_base_model(name, **model_params):
    '''Base model factory.

    '''
    if name == 'resnet':
        from dlutils.models.fcn_resnet import ResnetBase
        return ResnetBase(**model_params)
    elif name == 'unet':
        from dlutils.models.unet import GenericUnetBase
        return GenericUnetBase(**model_params)
    elif name == 'resnext':
        from dlutils.models.resnext import ResneXtBase
        return ResneXtBase(**model_params)
    elif name == 'rxunet':
        from dlutils.models.rxunet import GenericRxUnetBase
        return GenericRxUnetBase(**model_params)
    elif name == 'hourglass':
        from dlutils.models.hourglass import GenericHourglassBase
        return GenericHourglassBase(**model_params)
    else:
        raise NotImplementedError('Model {} not known!'.format(name))
