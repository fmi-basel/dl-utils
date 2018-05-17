from dlutils.training.augmentations import flip_axis
from dlutils.training.augmentations import get_zoom_transform
from dlutils.training.augmentations import apply_transform
from dlutils.training.augmentations import get_rotation_transform
from dlutils.training.augmentations import get_shear_transform
from dlutils.training.augmentations import get_combined_transform

import numpy as np

from scipy.misc import face as example_image

# import matplotlib.pyplot as plt

import pytest


def check_dimensions(x_shape, y_shape):
    '''
    '''
    if len(x_shape) != len(y_shape) or any(x != y
                                           for x, y in zip(x_shape, y_shape)):
        raise RuntimeError('Mismatching shapes: {} != {}'.format(
            x_shape, y_shape))


def test_flip():
    '''
    '''
    img = example_image()
    img_shape = img.shape

    for dim in xrange(2):
        flipped = flip_axis(img, dim)
        check_dimensions(img_shape, flipped.shape)


@pytest.mark.parametrize("factor", [0.5, 0.75, 1.0, 1.25, 1.5])
def test_zoom(factor):
    '''
    '''
    img = example_image()
    transform = get_zoom_transform(factor, img.ndim)
    scaled = apply_transform(img, transform)

    check_dimensions(img.shape, scaled.shape)

    # _, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(scaled)
    # plt.show()


@pytest.mark.parametrize("angle", np.linspace(-90, 90, 7).tolist())
def test_rotate(angle):
    '''
    '''
    img = example_image()
    transform = get_rotation_transform(angle, img.ndim)
    scaled = apply_transform(img, transform)

    check_dimensions(img.shape, scaled.shape)

    # _, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(scaled)
    # plt.show()


@pytest.mark.parametrize("angle", np.linspace(-30, 30, 5).tolist())
def test_shear(angle):
    '''
    '''
    img = example_image()
    transform = get_shear_transform(angle, img.ndim)
    scaled = apply_transform(img, transform)

    check_dimensions(img.shape, scaled.shape)

    # _, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(scaled)
    # plt.show()


@pytest.mark.parametrize("factor,angle",
                         zip([0.5, 1.5, 0.5, 1.5, 0.5],
                             np.linspace(-30, 30, 5).tolist()))
def test_concatenated(factor, angle):
    '''
    '''
    img = example_image()
    transforms = [
        get_zoom_transform(factor, img.ndim),
        get_shear_transform(angle / 2, img.ndim),
        get_rotation_transform(angle, img.ndim)
    ]
    combined_transforms = get_combined_transform(transforms)

    scaled = apply_transform(img, combined_transforms)

    check_dimensions(img.shape, scaled.shape)

    # _, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(scaled)
    # plt.show()


if __name__ == '__main__':
    test_concatenated(0.85, -45)