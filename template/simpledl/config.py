from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import yaml
import os
import logging

from tempfile import mkstemp

DEFAULT_CONFIG = {
    'training': {
        'epochs': 100,
        'steps_per_epoch': None,
        'learning_rate': None,
        'verbose': 1
    },
    'dataset': {
        'patch_size': (500, 500),
        'task_type': None,
        'buffer':
        True,  # disable if you have a huge dataset that doesnt fit in RAM.
        'batch_size': 4,
    },
    'augmentation': None,
    'model': {
        'input_shape': (None, None, 1),
    }
}


def get_config(template_config_path=None, **kwargs):
    '''create config based on default config or from a given template.

    '''
    config = Config()
    if template_config_path is None:
        return config
    template_config = read_config(template_config_path)
    config.update_from(template_config)
    return config


def read_config(path):
    '''read config from .yaml

    '''
    if not os.path.exists(path):
        raise IOError('There is no existing config at {}'.format(path))
    config = Config()
    config.update_from(yaml.safe_load(open(path, 'r')))
    return config


def write_config(config, path):
    '''save config as .yaml file.

    '''
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # TODO consider handling this in a nicer fashion
    if config.get('augmentation', None) is not None:
        for key, val in config['augmentation'].items():
            if isinstance(val, (bool, str, int, float)):
                continue
            config['augmentation'][key] = 'cant dump'

    yaml.safe_dump(
        {key: val
         for key, val in config.items()},
        open(path, 'w'),
        default_flow_style=False)


class Config(dict):
    '''
    '''

    logger = logging.getLogger(__name__)

    def __init__(self, path=None, **kwargs):
        '''
        '''
        self.update_from(DEFAULT_CONFIG)
        self.update_from(kwargs)
        self.path = path

    def __enter__(self):
        '''
        '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        '''
        Config.logger.info('Writing config at %s', self.path)

        if self.path is None:
            self.path = mkstemp(prefix='config', suffix='.yaml')[1]

        write_config(self, self.path)

    def update_from(self, other):
        '''
        '''
        for key, val in other.items():
            if self.get(key, None) is None:
                self[key] = val
            elif isinstance(self[key], dict):
                self[key].update(val)
            else:
                self[key] = val
