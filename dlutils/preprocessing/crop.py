from scipy.ndimage import find_objects


def crop_object(images, labels, margins=None):
    '''
    Crop all images based on object found in labels.

    To crop labels, pass it in the images list as well.
    '''

    if margins is None:
        margins = (0,) * labels.ndim
    loc = find_objects(labels >= 1)[0]
    loc = tuple(slice(max(sli.start - margin, 0), sli.stop + margin)
                for sli, margin in zip(loc, margins))

    return [image[loc] for image in images]
