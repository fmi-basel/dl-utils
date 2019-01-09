from __future__ import absolute_import
import abc
import six

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from .config import Config


@six.add_metaclass(abc.ABCMeta)
class ConfigGeneratorBase(object):
    def __init__(self, base_config, param_grid):
        '''
        '''
        self.base_config = base_config

    @abc.abstractmethod
    def _get_sample(self, idx):
        '''
        '''
        pass

    def __getitem__(self, idx):
        '''draw idx-th config from sampler.

        '''
        config = Config(**self.base_config)

        # update config with params of grid point.
        for key, val in self._get_sample(idx).items():

            # handle nested dicts
            split_keys = key.split('/')
            if len(split_keys) == 1:
                config[key] = val
            else:
                handle = config
                for subkey in split_keys[:-1]:
                    handle = handle[subkey]
                handle[split_keys[-1]] = val
        return config


class ConfigGrid(ConfigGeneratorBase):
    '''Generator to draw configs from a regular grid.

    '''

    def __init__(self, base_config, param_grid):
        '''
        '''
        super(ConfigGrid, self).__init__(base_config, param_grid)
        self.parameter_sampler = ParameterGrid(param_grid)

    def __len__(self):
        return len(self.parameter_sampler)

    def _get_sample(self, idx):
        '''
        '''
        return self.parameter_sampler[idx]


class ConfigSampler(ConfigGeneratorBase):
    '''Generator to draw config samples randomly.

    '''

    def __init__(self, base_config, param_grid, n_iter):
        '''
        '''
        super(ConfigSampler, self).__init__(base_config, param_grid)
        self.parameter_sampler = ParameterSampler(param_grid, n_iter=n_iter)

    def __len__(self):
        return len(self.parameter_sampler)

    def _get_sample(self, idx):
        '''

        Notes
        -----
        '''
        for count, config in enumerate(self.parameter_sampler):
            if count == idx:
                return config
