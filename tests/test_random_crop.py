import itertools

import pytest
import numpy as np
import tensorflow as tf

from dlutils.dataset.cropping import random_crop


def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...

    Snippet from https://docs.python.org/3.6/library/itertools.html
    '''
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


CROP_TEST_PARAMS = list(
    itertools.product([(5, 10, 11, 2), (20, 10, 3), (5, 10)], [2, 5]))


@pytest.mark.parametrize('patch_size, n_inputs', CROP_TEST_PARAMS)
def test_random_crop_on_dict(patch_size, n_inputs):
    '''test tf.data compatible random crop function on an input_dict.

    '''
    tf.random.set_seed(17)
    img_shape = tuple(2 * val for val in patch_size)
    img = np.arange(np.prod(img_shape)).reshape(img_shape)

    cropper = random_crop(patch_size)

    input_dict = {key: key * img for key in range(1, n_inputs + 1)}
    first_patch = cropper(input_dict)

    for _ in range(10):
        patch = cropper(input_dict)

        print('Patch')
        for key, val in patch.items():
            print(key, val.shape)

        # check if patches have the correct shape
        for key in input_dict.keys():
            vals = patch[key].numpy()
            assert vals.ndim == img.ndim
            assert np.all(vals.shape == patch_size)

        assert len(patch.keys()) == n_inputs

        # check if all inputs were cropped in the same location
        for first_key, second_key in pairwise(patch):
            assert np.all(patch[first_key].numpy() *
                          second_key == patch[second_key].numpy() * first_key)

        # make sure we're not drawing the same patch over and over
        for key in patch.keys():
            assert not np.all(patch[key].numpy() == first_patch[key].numpy())


@pytest.mark.parametrize('patch_size, n_inputs', CROP_TEST_PARAMS)
def test_random_crop_on_list(patch_size, n_inputs):
    '''test tf.data compatible random crop function on a list.

    NOTE we currently expect the test to fail as there's no extra handling
    of lists/tuples/dicts

    '''
    tf.random.set_seed(17)
    img_shape = tuple(2 * val for val in patch_size)
    img = np.arange(np.prod(img_shape)).reshape(img_shape)

    cropper = random_crop(patch_size)

    input_list = [key * img for key in range(1, n_inputs + 1)]
    first_patch = cropper(input_list)

    for _ in range(10):
        patch = cropper(input_list)

        print('Patch')
        for val in patch:
            print(val.shape)

        # check if patches have the correct shape
        for vals in patch:
            vals = vals.numpy()
            assert vals.ndim == img.ndim
            assert np.all(vals.shape == patch_size)

        assert len(patch) == n_inputs

        # check if all inputs were cropped in the same location
        for (ii, first), (jj, second) in pairwise(enumerate(patch, start=1)):
            print(ii, jj)
            assert np.all(first.numpy() * jj == second.numpy() * ii)

        # make sure we're not drawing the same patch over and over
        for ii, vals in enumerate(patch):
            assert not np.all(vals.numpy() == first_patch[ii].numpy())


@pytest.mark.parametrize(
    'shapes',
    [[(13, 15, 1),
      (13, 15, 2)], [(4, 5, 6, 7),
                     (4, 5, 8, 7)], [(13, 15, 1),
                                     (13, 15)], [(13, 4, 1), (13, 4, 1),
                                                 (13, 4, 1), (11, 4, 1)]])
def test_random_crop_mismatching_shapes(shapes):
    '''test if shape mismatches in the input raise.

    '''
    patch_size = (13, 13, 1)
    inputs = [np.ones(shape) for shape in shapes]

    cropper = random_crop(patch_size)
    with pytest.raises(ValueError):
        cropper(inputs)
