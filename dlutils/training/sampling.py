import numpy as np


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


def get_random_patch(channels, patch_size, augmentator=None):
    '''sample a random patch from all given channels.

    channels : list of 2D images
    '''
    assert isinstance(channels, list)

    # TODO consider pre-sampling augmentations
    if augmentator is not None:
        pass

    # sample from all channels
    patch_coord = get_random_patch_corner(channels[0].shape, patch_size)
    slices = [slice(x, x + dx) for x, dx in zip(patch_coord, patch_size)]
    patches = [channel[slices] for channel in channels]

    # # post-sampling augmentations
    if augmentator is not None:
        patches[0], patches[1:] = augmentator.post_sampling_augmentation(
            inputs=[patches[0], ], targets=patches[1:])
        patches[0] = patches[0][0]

    return patches
