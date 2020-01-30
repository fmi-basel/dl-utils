import numpy as np

from scipy.ndimage.measurements import center_of_mass


def build_target_vfield(annot):
    '''builds a vfield pointing to instances'' center of mass from given annotations.'''

    # ~annot = annot.squeeze() # not responsability --> case stack with single slice

    unique_l = np.unique(annot)
    unique_instance_labels = unique_l[unique_l > 0]

    cms = center_of_mass(annot > 0, annot, unique_instance_labels)
    cms = [np.asarray(cm) for cm in cms]

    displacement = np.zeros(annot.shape + (annot.ndim, ))

    for cm, l in zip(cms, unique_instance_labels):
        pts = np.argwhere(annot == l)
        displacement[annot == l] = cm - pts

    return displacement
