import numpy as np

from augmentations import flip_axis


def get_random_patch_corner(img_shape, patch_size):
    '''get the corner coordinate of a random patch

    '''
    for ii, (x, y) in enumerate(zip(patch_size, img_shape)):
        if x >= y:
            raise ValueError(
                "patch_size is larger than img_shape: {} >= {}".format(
                    patch_size, img_shape))

    coordinate = [
        np.random.choice(y - x) for x, y in zip(patch_size, img_shape)
    ]
    return coordinate


def get_random_patch(channels,
                     patch_size,
                     horizontal_flip=False,
                     vertical_flip=False):
    '''sample a random patch from all given channels.

    channels : list of 2D images
    '''
    assert isinstance(channels, list)

    # TODO consider pre-sampling augmentations

    # sample from all channels
    patch_coord = get_random_patch_corner(channels[0].shape, patch_size)
    slices = [slice(x, x + dx) for x, dx in zip(patch_coord, patch_size)]
    patches = [channel[slices] for channel in channels]

    # post-sampling augmentations
    if horizontal_flip:
        if np.random.random() < 0.5:
            for idx, patch in enumerate(patches):
                patches[idx] = flip_axis(patch, 1)

    if vertical_flip:
        if np.random.random() < 0.5:
            for idx, patch in enumerate(patches):
                patches[idx] = flip_axis(patch, 0)

    return patches
