from scipy.ndimage import find_objects

def crop_object(image, labels, margins=None):
    '''
    Crop both, image and labels based on object found in labels
    '''
    
    if margins is None:
        margins = (0,)*image.ndim
    loc = find_objects(labels >= 1)[0]
    loc = tuple(slice(max(sli.start-margin,0),sli.stop+margin) for sli,margin in zip(loc,margins))

    return image[loc], labels[loc]

