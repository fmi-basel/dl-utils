import numpy as np
from scipy.ndimage.interpolation import affine_transform


def flip_axis(x, axis):
    '''flip axis of image for data augmentation.

    '''
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def get_zoom_transform(factor, ndim):
    '''
    '''
    transform = np.diag(np.ones(ndim + 1))
    for idx in xrange(ndim - 1):  # only scale spatial dimensions
        transform[idx, idx] = factor
    return transform


def get_shear_transform(angle, ndim, axis=1):
    '''
    '''
    angle = np.deg2rad(angle)
    transform = np.diag(np.ones(ndim + 1))
    transform[0, axis] = -np.sin(angle)
    transform[axis, axis] = np.cos(angle)
    return transform


def get_rotation_transform(angle, ndim, axis=0):
    '''
    Parameters
    ----------
    angle : float
        rotation angle in degrees

    NOTE rotation only in-plane
    '''
    angle = np.deg2rad(angle)
    transform = np.diag(np.ones(ndim + 1))
    transform[axis, axis] = np.cos(angle)
    transform[axis + 1, axis] = -np.sin(angle)
    transform[axis + 1, axis + 1] = np.cos(angle)
    transform[axis, axis + 1] = np.sin(angle)
    return transform


def get_combined_transform(transforms):
    '''
    '''
    combined = None

    for transform in transforms:
        combined = np.dot(combined,
                          transform) if combined is not None else transform
    return combined


def transform_matrix_offset(matrix, shape):
    '''
    '''
    offset_matrix = np.diag(np.ones(len(matrix)))
    reset_matrix = np.diag(np.ones(len(matrix)))

    for idx, dim in zip(xrange(len(matrix) - 1), shape):
        dx = float(dim) / 2 + 0.5
        offset_matrix[idx, -1] = dx
        reset_matrix[idx, -1] = -dx

    return np.dot(np.dot(offset_matrix, matrix), reset_matrix)


def apply_transform(x, transform):
    '''
    '''
    isbool = x.dtype == bool
    if isbool:
        x = x.astype(np.float32)
    transform = transform_matrix_offset(transform, x.shape)
    x = affine_transform(x, transform, mode='nearest')
    if isbool:
        x = x >= 0.5
    return x


def draw(dist):
    '''
    '''
    if hasattr(dist, 'rvs'):
        return dist.rvs(1)[0]
    if isinstance(dist, float):  # Uniform[0, val]
        return np.random.rand(1) * dist
    else:
        raise RuntimeError('Cant draw value from {}'.format(type(dist)))


class ImageDataAugmentation(dict):
    def __init__(self, **kwargs):
        '''
        '''
        self['zoom'] = None
        self['rotation'] = None
        self['shear'] = None
        self['intensity_scaling'] = None
        self['intensity_shift'] = None
        for key, val in kwargs.iteritems():
            self[key] = val

    def pre_sampling_augmentation(self, inputs, targets):
        '''
        '''
        raise NotImplementedError(
            'Pre-sampling augmentations are not implemented')

    def post_sampling_augmentation(self, inputs, targets):
        '''applies all augmentations that can be done
        post-sampling.

        '''
        image_dims = inputs[0].shape[:-1]
        assert len(image_dims) >= 2

        if self.get('flip', False):
            for dim, _ in enumerate(image_dims):
                if np.random.random() < 0.5:

                    # horizontal and vertical flips are applied to both in
                    # and output patches
                    for idx, patch in enumerate(inputs):
                        inputs[idx] = flip_axis(patch, dim)

                    for idx, patch in enumerate(targets):
                        targets[idx] = flip_axis(patch, dim)

        transforms = []
        if self['zoom'] is not None:
            zoom_factor = draw(self['zoom'])
            transforms.append(get_zoom_transform(zoom_factor, inputs[0].ndim))

        if self['shear'] is not None:
            shear_angle = draw(self['shear'])
            transforms.append(get_shear_transform(shear_angle, inputs[0].ndim))

        if self['rotation'] is not None:
            rotation_angle = draw(self['rotation'])
            transforms.append(
                get_rotation_transform(rotation_angle, inputs[0].ndim))

        # Spatial transforms are also applied to both input and target
        if len(transforms) > 0:
            combined = get_combined_transform(transforms)
            for idx, patch in enumerate(inputs):
                inputs[idx] = apply_transform(patch, combined)

            for idx, patch in enumerate(targets):
                targets[idx] = apply_transform(patch, combined)

        # intensity rescaling/shift
        if self['intensity_shift'] is not None or \
           self['intensity_scaling'] is not None:
            shift = draw(self['intensity_shift']
                         ) if self['intensity_shift'] is not None else 0
            scaling = draw(self['intensity_scaling']
                           ) if self['intensity_scaling'] is not None else 1.

            for idx, patch in enumerate(inputs):
                inputs[idx] = (patch * scaling) + shift

        return inputs, targets
