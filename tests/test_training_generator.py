from dlutils.training.generator import TrainingGenerator
from dlutils.training.generator import LazyTrainingHandle
from dlutils.training.augmentations import ImageDataAugmentation

from scipy.stats import norm as gaussian_dist
from scipy.misc import face as example_image

import numpy as np

import pytest


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from keras.backend import clear_session
    clear_session()


class Handle(LazyTrainingHandle):
    '''
    '''

    def __init__(self, shape):
        self.shape = shape

    def load(self, **kwargs):
        self['input'] = np.random.randn(*self.shape)
        self['input'][5:10, ...] += 10
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


@pytest.mark.parametrize("n_handles,samples_per_handle,batch_size",
                         [(100, 1, 7), (3, 20, 10), (19, 4, 75)])
def test_2d_sampler(n_handles, samples_per_handle, batch_size):
    '''
    '''
    n_channels = 1
    img_shape = (500, 500)
    patch_size = (64, 64)

    handles = [
        Handle(shape=img_shape + (n_channels, )) for _ in xrange(n_handles)
    ]

    generator = TrainingGenerator(
        handles,
        patch_size=patch_size,
        batch_size=batch_size,
        seed=11,
        samples_per_handle=samples_per_handle)
    expected_length = int(float(n_handles * samples_per_handle) / batch_size)
    assert len(generator) == expected_length
    assert len(generator) > 0

    # Check batch dimensions
    for batch_idx in xrange(expected_length):
        in_batch, out_batch = generator[0]

        print in_batch['input'].shape,
        expected_shape = (batch_size, patch_size[0], patch_size[1], n_channels)
        assert in_batch.keys()[0] == 'input'

        for key, val in in_batch.iteritems():
            np.testing.assert_equal(
                val.shape, expected_shape,
                'Shapes of {} are mismatching: {} != {}'.format(
                    key, in_batch[key].shape, expected_shape))

        for key, val in out_batch.iteritems():
            np.testing.assert_equal(
                val.shape, expected_shape,
                'Shapes of {} are mismatching: {} != {}'.format(
                    key, val.shape, expected_shape))


def test_generator_with_augmentation(n_handles=5,
                                     samples_per_handle=5,
                                     batch_size=5):
    '''
    '''
    n_channels = 1
    img_shape = (500, 500)
    patch_size = (64, 64)

    handles = [
        Handle(shape=img_shape + (n_channels, )) for _ in xrange(n_handles)
    ]
    
    generator = TrainingGenerator(
        handles,
        patch_size=patch_size,
        batch_size=batch_size,
        seed=11,
        samples_per_handle=samples_per_handle,
        augmentator=ImageDataAugmentation(
            intensity_scaling=gaussian_dist(loc=1, scale=1),
            intensity_shift=gaussian_dist(loc=0, scale=15),
            zoom=gaussian_dist(loc=1, scale=0.05),
            rotation=10.))

    expected_length = int(float(n_handles * samples_per_handle) / batch_size)
    assert len(generator) == expected_length
    assert len(generator) > 0

    # Check batch dimensions
    for batch_idx in xrange(expected_length):
        in_batch, out_batch = generator[0]

        print in_batch['input'].shape,
        expected_shape = (batch_size, patch_size[0], patch_size[1], n_channels)
        assert in_batch.keys()[0] == 'input'

        for key, val in in_batch.iteritems():
            np.testing.assert_equal(
                val.shape, expected_shape,
                'Shapes of {} are mismatching: {} != {}'.format(
                    key, in_batch[key].shape, expected_shape))

        for key, val in out_batch.iteritems():
            np.testing.assert_equal(
                val.shape, expected_shape,
                'Shapes of {} are mismatching: {} != {}'.format(
                    key, val.shape, expected_shape))


if __name__ == '__main__':
    test_generator_with_augmentation()