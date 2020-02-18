import itertools

import pytest
import tensorflow as tf
import numpy as np

from dlutils.dataset.augmentations import random_axis_flip
from dlutils.dataset.augmentations import random_gaussian_noise
from dlutils.dataset.augmentations import random_gaussian_offset


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
        tf.convert_to_tensor(
            np.random.randn(*patch_size) + 17, dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(
            np.random.randn(*patch_size) > .0, dtype=tf.float32)
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
        tf.convert_to_tensor(
            np.random.randn(*patch_size) + 17, dtype=tf.float32),
        'segm':
        tf.convert_to_tensor(
            np.random.randn(*patch_size) > .0, dtype=tf.float32)
    }

    distorter = random_gaussian_offset(offset_sigma, ['img'])

    previous = distorter(original)
    offsets = []

    for _ in range(reps):
        augmented = distorter(original)

        assert not np.all(previous['img'].numpy() == augmented['img'].numpy())
        assert np.all(original['segm'].numpy() == augmented['segm'].numpy())

        noise = augmented['img'].numpy() - original['img'].numpy()

        assert -4 * offset_sigma <= noise.mean() <= 4 * offset_sigma
        assert -0.001 <= noise.std() <= 0.001  # should be flat.

        offsets.append(noise.mean())

    # sanity check for overall offset distribution.
    assert (-1.96 * offset_sigma / np.sqrt(reps) <= np.mean(offsets) <=
            1.96 * offset_sigma / np.sqrt(reps))
    assert 0.9 * offset_sigma <= np.std(offsets) <= 1.1 * offset_sigma
