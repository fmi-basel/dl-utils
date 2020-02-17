from sklearn.cluster import KMeans
from numba import njit
import numpy as np

from improc.feature import local_max
from improc.morphology import clean_up_labels


@njit
def accumulate_votes_2D(abs_disp, votes_shape):
    '''Casts vote from embeddings / vector displacement field.
    
    Notes:
    currently restricted to bins = image voxels. i.e. embeddings must
    represents positions within image bounds
    '''

    # only counts votes that are within the image boundary
    votes = np.zeros(votes_shape, dtype=np.uint)

    for i in range(abs_disp.shape[0]):
        x, y = abs_disp[i, 0], abs_disp[i, 1]

        # verify that the displacement is within the image bounds
        if x >= 0 and x < votes.shape[0] and y >= 0 and y < votes.shape[1]:
            votes[x, y] += 1

    return votes


@njit
def accumulate_votes_3D(abs_disp, votes_shape):
    '''Casts vote from embeddings / vector displacement field.
    
    Notes:
    currently restricted to bins = image voxels. i.e. embeddings must
    represents positions within image bounds
    '''

    # only counts votes that are within the image boundary
    votes = np.zeros(votes_shape, dtype=np.uint)

    for i in range(abs_disp.shape[0]):
        z, x, y = abs_disp[i, 0], abs_disp[i, 1], abs_disp[i, 2]

        # verify that the displacement is within the image bounds
        if z >= 0 and z < votes.shape[0] and x >= 0 and x < votes.shape[
                1] and y >= 0 and y < votes.shape[2]:
            votes[z, x, y] += 1

    return votes


@njit
def accumulate_votes_4D(abs_disp, votes_shape):
    '''Casts vote from embeddings / vector displacement field.
    
    Notes:
    currently restricted to bins = image voxels. i.e. embeddings must
    represents positions within image bounds
    '''

    # only counts votes that are within the image boundary
    votes = np.zeros(votes_shape, dtype=np.uint)

    for i in range(abs_disp.shape[0]):
        z, x, y, t = abs_disp[i, 0], abs_disp[i, 1], abs_disp[i, 2], abs_disp[
            i, 3]

        # verify that the displacement is within the image bounds
        if z >= 0 and z < votes.shape[0] and x >= 0 and x < votes.shape[
                1] and y >= 0 and y < votes.shape[
                    2] and t >= 0 and t < votes.shape[3]:
            votes[z, x, y, t] += 1

    return votes


# Note: numba does not support fancy indexing --> separate 3D and 4D cases
def accumulate_votes(abs_disp, fg_mask=None):

    votes_shape = abs_disp.shape[:-1]

    # only count votes in forground mask
    if fg_mask is None:
        abs_disp = abs_disp.reshape(-1, abs_disp.shape[-1])
    else:
        abs_disp = abs_disp[fg_mask]

    if abs_disp.shape[-1] == 3:
        votes = accumulate_votes_3D(abs_disp, votes_shape)
    elif abs_disp.shape[-1] == 4:
        votes = accumulate_votes_4D(abs_disp, votes_shape)
    elif abs_disp.shape[-1] == 2:
        votes = accumulate_votes_2D(abs_disp, votes_shape)
    else:
        raise NotImplementedError(
            "numba 'accumulate_votes' not implemented for ndim = {}".format(
                abs_disp.shape[-1]))

    return votes


@njit
def fg_from_votes_2D(abs_disp, votes, count_threshold=5):
    '''Returns the foreground mask defined as regions that voted
    for bins with more than count_threshold votes.'''

    out_shape = abs_disp.shape[:-1]
    abs_disp = abs_disp.reshape(-1, 2)
    fg_mask = np.zeros(abs_disp.shape[:-1], dtype=np.bool_)

    for i in range(abs_disp.shape[0]):
        x, y = abs_disp[i, 0], abs_disp[i, 1]

        # verify that the displacement is within the image bounds
        if x >= 0 and x < votes.shape[0] and y >= 0 and y < votes.shape[1]:
            fg_mask[i] = votes[x, y] > count_threshold

    return fg_mask.reshape(out_shape)


@njit
def fg_from_votes_3D(abs_disp, votes, count_threshold=5):
    '''Returns the foreground mask defined as regions that voted
    for bins with more than count_threshold votes.'''

    out_shape = abs_disp.shape[:-1]
    abs_disp = abs_disp.reshape(-1, 3)
    fg_mask = np.zeros(abs_disp.shape[:-1], dtype=np.bool_)

    for i in range(abs_disp.shape[0]):
        z, x, y = abs_disp[i, 0], abs_disp[i, 1], abs_disp[i, 2]

        # verify that the displacement is within the image bounds
        if z >= 0 and z < votes.shape[0] and x >= 0 and x < votes.shape[
                1] and y >= 0 and y < votes.shape[2]:
            fg_mask[i] = votes[z, x, y] > count_threshold

    return fg_mask.reshape(out_shape)


@njit
def fg_from_votes_4D(abs_disp, votes, count_threshold=5):
    '''Returns the foreground mask defined as regions that voted
    for bins with more than count_threshold votes.'''

    out_shape = abs_disp.shape[:-1]
    abs_disp = abs_disp.reshape(-1, 4)
    fg_mask = np.zeros(abs_disp.shape[:-1], dtype=np.bool_)

    for i in range(abs_disp.shape[0]):
        z, x, y, t = abs_disp[i, 0], abs_disp[i, 1], abs_disp[i, 2], abs_disp[
            i, 3]

        # verify that the displacement is within the image bounds
        if z >= 0 and z < votes.shape[0] and x >= 0 and x < votes.shape[
                1] and y >= 0 and y < votes.shape[
                    2] and t >= 0 and t < votes.shape[3]:
            fg_mask[i] = votes[z, x, y, t] > count_threshold

    return fg_mask.reshape(out_shape)


def fg_from_votes(abs_disp, votes, count_threshold=5):

    if abs_disp.shape[-1] == 3:
        fg_mask = fg_from_votes_3D(abs_disp,
                                   votes,
                                   count_threshold=count_threshold)
    elif abs_disp.shape[-1] == 4:
        fg_mask = fg_from_votes_4D(abs_disp,
                                   votes,
                                   count_threshold=count_threshold)
    elif abs_disp.shape[-1] == 2:
        fg_mask = fg_from_votes_2D(abs_disp,
                                   votes,
                                   count_threshold=count_threshold)
    else:
        raise NotImplementedError(
            "numba 'fg_from_votes' not implemented for ndim = {}".format(
                abs_disp.shape[-1]))

    return fg_mask


def relative_to_absolute_displacement(rel_displacement):
    '''Converts a relative displacement field to absolute 
    coordinates by summing the image coordinates.
    
    Notes:
    The output can be treated as embeddings of dimension 
    equal to image.ndim'''

    abs_disp = np.stack(np.meshgrid(
        *[np.arange(s, dtype=np.float32) for s in rel_displacement.shape[:-1]],
        indexing='ij'),
                        axis=-1) + rel_displacement

    return np.round(abs_disp).astype(np.int16)


def kmeans_cluster_embeddings(embeddings,
                              seeds,
                              spacing=1,
                              mask=None,
                              refine_centers=False):
    '''Clusters embeddings where mask is True and returns the labels.
    
    refine_centers==False --> simple nearest neighbor assignement'''
    # NOTE: brute force scipy cdist runs out of memory for large images
    # Kmeans without fit does nearest neighbor assignement but seems to better handle memory usage

    spacing = np.broadcast_to(np.array(spacing), embeddings.ndim - 1)

    if mask is None:
        mask = np.ones(embeddings.shape[:-1], dtype=bool)

    labels = np.zeros(embeddings.shape[:-1], dtype=int)

    if len(seeds) > 0:
        clusterer = KMeans(n_clusters=len(seeds),
                           init=np.asarray(seeds) * spacing,
                           n_init=1)
        if refine_centers:
            clusterer.fit(embeddings[mask] * spacing)
        else:
            # hack to init clusterer
            clusterer.fit(np.asarray(seeds) * spacing)

        labels[mask] = clusterer.predict(embeddings[mask] * spacing) + 1

    return labels


def vfield_to_labels(vfield,
                     spacing=1.,
                     peak_min_distance=2,
                     min_count=5,
                     fg_mask=None,
                     fgbg_threshold=5,
                     n_instance_max=500,
                     opening_radius=None,
                     fill_holes=True,
                     clean_size_threshold=None,
                     return_score=False,
                     return_embeddings=False,
                     return_votes=False,
                     return_fgmask=False):
    '''Converts a voting vectorfield to labels.
    
    Notes:
    An instance voting score of 1.0 means that all votes fell in the same bin (i.e. same pixel/voxel)

    '''

    spacing = np.broadcast_to(np.asarray(spacing), vfield.shape[-1])

    abs_disp = relative_to_absolute_displacement(vfield)
    votes = accumulate_votes(abs_disp, fg_mask).astype(np.float32)
    if fg_mask is None:
        fg_mask = fg_from_votes(abs_disp,
                                votes,
                                count_threshold=fgbg_threshold)
        votes = accumulate_votes(abs_disp, fg_mask).astype(np.float32)

    seeds, seeds_intensities = local_max(votes,
                                         min_distance=peak_min_distance,
                                         threshold=min_count,
                                         spacing=spacing)
    seeds = seeds[:n_instance_max]
    seeds_intensities = seeds_intensities[:n_instance_max]

    labels = kmeans_cluster_embeddings(abs_disp,
                                       seeds,
                                       spacing=spacing,
                                       mask=fg_mask)
    unique_labels, label_size = np.unique(labels, return_counts=True)

    # TODO option to filter by size or 'largest'
    labels = clean_up_labels(labels,
                             fill_holes=fill_holes,
                             radius=opening_radius,
                             size_threshold=clean_size_threshold,
                             spacing=spacing)

    outputs = {}
    if return_score:
        # normalize seed score by instance area/volume
        outputs['voting_score'] = seeds_intensities / label_size[1:]
    if return_embeddings:
        outputs['embeddings'] = abs_disp
    if return_votes:
        outputs['votes'] = votes
    if return_fgmask:
        outputs['fg_mask'] = fg_mask

    outputs['labels'] = labels
    return outputs
