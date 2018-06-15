from dlutils.training.sampling import get_random_patch

import numpy as np

import pytest


def check_dimensions(x_shape, y_shape):
    '''
    '''
    if len(x_shape) != len(y_shape) or any(x != y
                                           for x, y in zip(x_shape, y_shape)):
        raise RuntimeError('Mismatching shapes: {} != {}'.format(
            x_shape, y_shape))


@pytest.mark.parametrize("patch_size,image_size,n_channels,n_images",
                         [((64, 64), (500, 500), 1, 1),
                          ((100, 33), (500, 500), 3, 5),
                          ((1, 50, 50), (100, 130, 130), 1, 3),
                          ((1, 70, 53), (75, 77, 99), 7, 1)])
def test_patch_sampler(patch_size, image_size, n_channels, n_images):
    '''
    '''
    images = [
        np.random.randn(*(image_size) + (n_channels, ))
        for _ in range(n_images)
    ]

    patches = get_random_patch(images, patch_size)

    assert len(patches) > 0
    assert len(patches) == n_images
    for patch in patches:
        check_dimensions(patch.shape, patch_size + (n_channels, ))


if __name__ == '__main__':
    test_patch_sampler(
        patch_size=(64, 64), image_size=(500, 500), n_channels=1, n_images=1)
