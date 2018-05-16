from dlutils.training.generator import TrainingGenerator
from dlutils.training.generator import LazyTrainingHandle

import numpy as np


class Handle(LazyTrainingHandle):
    '''
    '''

    def __init__(self, shape):
        self.shape = shape

    def load(self, **kwargs):
        self['input'] = np.random.randn(*self.shape)
        self['output_a'] = self['input'] > 1
        self['output_b'] = self['input'] < -1

    def get_input_keys(self):
        '''returns a list of input keys
        '''
        return ['input']

    def get_output_keys(self):
        '''returns a list of output keys
        '''
        return ['output_a', 'output_b']

    def clear(self):
        '''clears any input/output data attributes.

        '''
        for key in self.get_input_keys() + self.get_output_keys():
            self[key] = None


def test_2d_sampler():
    '''
    '''

    n_handles = 5
    img_shape = (500, 500)
    patch_size = (64, 64)
    batch_size = 5
    samples_per_handle = 5

    handles = [Handle(shape=img_shape) for _ in xrange(n_handles)]

    generator = TrainingGenerator(
        handles,
        patch_size=patch_size,
        batch_size=batch_size,
        seed=11,
        samples_per_handle=samples_per_handle)
    expected_length = int(float(n_handles * samples_per_handle) / batch_size)
    assert len(generator) == expected_length
    assert len(generator) > 0

    # TODO fix issue that len(..) == 0 is possible with len(handles) > 0

    # check first and last batch
    in_batch, out_batch = generator[0]
    assert in_batch.keys()[0] == 'input'
    assert all(
        x == y
        for x, y in zip(in_batch['input'].shape,
                        [batch_size, patch_size[0], patch_size[1], 1]))


if __name__ == '__main__':
    test_2d_sampler()
