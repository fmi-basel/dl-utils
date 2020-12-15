import itertools

import pytest
import tensorflow as tf
import numpy as np

from dlutils.dataset.augmentations import random_axis_flip
from dlutils.dataset.augmentations import random_gaussian_noise
from dlutils.dataset.augmentations import random_gaussian_offset, random_intensity_scaling
from dlutils.dataset.augmentations import random_gaussian_filter
from dlutils.dataset.augmentations import random_hsv
from dlutils.dataset.augmentations import centered_affine_transform_matrix, random_affine_transform
from dlutils.dataset.augmentations import random_warp


def _get_binomial_ci(prob, reps):
    '''approximates the .95 interval for the empirical mean in binomial trials.

    '''
    center = prob * reps
    dx = 1.96 * np.sqrt(prob * (1 - prob) / reps) * reps
    return center - dx, center + dx


@pytest.fixture(autouse=True)
def set_tf_seed():
    '''
    '''
    tf.random.set_seed(13)


@pytest.mark.parametrize('patch_size, axis, flip_prob',
                         itertools.product([(3, 5, 2), (7, 2, 5)], [0, 1, 2],
                                           [0.2, 0.5, 0.8]))
def test_random_flip(patch_size, axis, flip_prob):
    '''test random flips.

    '''
    reps = 100
    original = {
        'img': np.random.randn(*patch_size),
        'segm': np.random.randn(*patch_size) > 0.
    }

    flipper = random_axis_flip(axis=axis, flip_prob=flip_prob)
    flip_counter = 0

    for _ in range(reps):
        augmented = flipper(original)

        none_flipped = all(
            np.all(augmented[key] == original[key]) for key in original.keys())
        all_flipped = all(
            np.all(np.flip(augmented[key], axis=axis) == original[key])
            for key in original.keys())

        assert (none_flipped != all_flipped) or (patch_size[axis] == 1)

        if all_flipped:
            flip_counter += 1

    expected_flips_lower, expected_flips_upper = _get_binomial_ci(
        flip_prob, reps)

    assert expected_flips_lower <= flip_counter <= expected_flips_upper


# yapf: disable
@pytest.mark.parametrize('patch_size, noise_params',
                         itertools.product([(3, 5, 10,), (10, 12,)],
                                           [(5., 1,), (3, 0.1,)]))
# yapf: enable
def test_random_noise(patch_size, noise_params):
    '''
    '''
    reps = 10
    noise_mu, noise_sigma = noise_params

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_gaussian_noise(noise_mu, noise_sigma, ['img'])

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())

        noise = augmented['img'].numpy() - original['img'].numpy()

        # just a crude sanity check. As we dont know which sigma was
        # drawn, this is supposed to "absorb" uncertainty in both the
        # expected range and the empirical estimate.
        assert -4 <= noise.mean() <= 4
        assert noise_mu - 8 * noise_sigma <= noise.std(
        ) <= noise_mu + 8 * noise_sigma


# yapf: disable
@pytest.mark.parametrize('patch_size, offset_sigma',
                         itertools.product([(3, 5, 10,), (10, 12,)],
                                            (5.3, 1, 100)))
# yapf: enable
def test_random_offset(patch_size, offset_sigma):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_gaussian_offset(offset_sigma, ['img'])

    previous = distorter(original)
    offsets = []

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())

        offset = augmented['img'].numpy() - original['img'].numpy()

        assert -4 * offset_sigma <= offset.mean() <= 4 * offset_sigma
        assert -0.001 <= offset.std() <= 0.001  # should be flat.

        offsets.append(offset.mean())

    # sanity check for overall offset distribution.
    assert (-1.96 * offset_sigma / np.sqrt(reps) <= np.mean(offsets) <=
            1.96 * offset_sigma / np.sqrt(reps))
    assert 0.9 * offset_sigma <= np.std(offsets) <= 1.1 * offset_sigma


# yapf: disable
@pytest.mark.parametrize('patch_size, bounds',
                         itertools.product([(3, 5, 10,1), (10, 12,1)],
                                            [(0.5, 2), (0.1,7)]))
# yapf: enable
def test_random_intensity_scaling(patch_size, bounds):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_intensity_scaling(bounds, ['img'])

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())

        scale = augmented['img'].numpy() / original['img'].numpy()

        assert bounds[0] <= scale.mean() <= bounds[1]
        assert -0.001 <= scale.std() <= 0.001  # should be flat.


# yapf: disable
@pytest.mark.parametrize('patch_size,sigma_range,active_p,device',
                         itertools.product([(25, 25,1)],
                                            [(0.5, 3)],
                                            [0.5, 1., 0.],
                                            ['/gpu', '/cpu']
                                            ))
# yapf: enable
def test_random_gaussian_filter(patch_size, sigma_range, active_p, device):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_gaussian_filter(sigma_range, ['img'], active_p, device)

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        if active_p >= 1.:
            assert not np.all(
                previous['img'].numpy() == augmented['img'].numpy())
        elif active_p <= 0.:
            assert np.all(previous['img'].numpy() == augmented['img'].numpy())

        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())


# yapf: disable
@pytest.mark.parametrize('patch_size',
                         [(25, 25,3), (25, 25, 25,3)])
# yapf: enable
def test_random_hsv(patch_size):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_hsv(['img'], max_delta_hue=1.)

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())


# yapf: disable
@pytest.mark.parametrize('expected_matrix,angle,shear_angle,zoomx,zoomy,shape',[
                            (np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 0., 0., 1., 1., (100,100)), # identity
                            (np.array([[2., 0., -50.], [0., .5, 25.], [0., 0., 1.]]), 0., 0., .5, 2., (100,100)), # scale
                            (np.array([[.5, 0., 50.], [0., 2., -100.], [0., 0., 1.]]), 0., 0., 2., .5, (200,200)), # scale
                            (np.array([[0., -1., 100.], [1., 0., 0.], [0., 0., 1.]]), 90., 0., 1., 1., (100,100)), # rotation
                            (np.array([[1., -1., 50.], [0., 0., 50.], [0., 0., 1.]]), 0., 90., 1., 1., (100,100)), # shear
                            ])
# yapf: enable
def test_centered_affine_transform_matrix(expected_matrix, angle, shear_angle,
                                          zoomx, zoomy, shape):

    matrix = centered_affine_transform_matrix(angle, shear_angle, zoomx, zoomy,
                                              shape)
    np.testing.assert_almost_equal(expected_matrix, matrix, decimal=5)


# yapf: disable
@pytest.mark.parametrize('patch_size,angle,shear,zoom',
                         itertools.product([(25, 25,1), (25, 25,3)],
                                            [10, 180.],
                                            [0, 15.],
                                            [(0.5, 2.), (1,1)]))
# yapf: enable
def test_random_affine_transform(patch_size, angle, shear, zoom):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segmA':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32),
        'segmB':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_affine_transform(interp_methods={
        'img': 'BILINEAR',
        'segmA': 'NEAREST',
        'segmB': 'BILINEAR'
    },
                                        angle=angle,
                                        shear=shear,
                                        zoom=zoom,
                                        zoomx=(1, 1),
                                        zoomy=(1, 1),
                                        fill_mode={
                                            'img': 'REFLECT',
                                            'segmA': 'CONSTANT',
                                            'segmB': 'CONSTANT'
                                        },
                                        cvals={
                                            'img': 0,
                                            'segmA': -1,
                                            'segmB': 0
                                        })

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        # distorted segm with nearest interpolation can remain unchanged for small deformations
        # assert not np.all(previous['segmA'].numpy() == augmented['segmA'].numpy())
        assert not np.all(
            previous['segmB'].numpy() == augmented['segmB'].numpy())

        assert not np.all(
            augmented['segmA'].numpy() == augmented['segmB'].numpy())

        assert augmented['segmA'].numpy().min() >= -1
        assert augmented['segmB'].numpy().min() >= 0

        original_labelsA = set(np.unique(original['segmA']))
        augmented_labelsA = set(np.unique(augmented['segmA'].numpy()))
        assert augmented_labelsA.issubset(original_labelsA.union({-1}))


# yapf: disable
@pytest.mark.parametrize('patch_size,max_amplitude',
                         itertools.product([(25, 25,1), (25, 25,3)],
                                            [3, 11.]))
# yapf: enable
def test_random_warp(patch_size, max_amplitude):
    '''
    '''
    reps = 10

    original = {
        'img':
        tf.convert_to_tensor(np.random.randn(*patch_size), dtype=tf.float32),
        'segmA':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32),
        'segmB':
        tf.convert_to_tensor(np.random.randn(*patch_size) > .0,
                             dtype=tf.float32)
    }

    distorter = random_warp(max_amplitude,
                            interp_methods={
                                'img': 'BILINEAR',
                                'segmA': 'NEAREST',
                                'segmB': 'BILINEAR'
                            },
                            fill_mode={
                                'img': 'REFLECT',
                                'segmA': 'CONSTANT',
                                'segmB': 'CONSTANT'
                            },
                            cvals={
                                'img': 0,
                                'segmA': -1,
                                'segmB': 0
                            })

    previous = distorter(original)

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert not np.all(
            previous['segmA'].numpy() == augmented['segmA'].numpy())
        assert not np.all(
            previous['segmB'].numpy() == augmented['segmB'].numpy())

        assert not np.all(
            augmented['segmA'].numpy() == augmented['segmB'].numpy())

        assert augmented['segmA'].numpy().min() >= -1
        assert augmented['segmB'].numpy().min() >= 0

        original_labelsA = set(np.unique(original['segmA']))
        augmented_labelsA = set(np.unique(augmented['segmA'].numpy()))
        assert augmented_labelsA.issubset(original_labelsA.union({-1}))
