import warnings

from sklearn.cluster import KMeans
from numba import njit
import numpy as np

from skimage.segmentation import relabel_sequential
from scipy.ndimage import find_objects
from scipy.ndimage import label as nd_label
from improc.feature import local_max


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


# TODO replace hacky kmeans use by
# from sklearn.neighbors import NearestNeighbors
def kmeans_cluster_embeddings(embeddings,
                              centers,
                              mask=None,
                              refine_centers=False):
    '''Clusters embeddings where mask is True and returns the labels.
    
    refine_centers==False --> simple nearest neighbor assignement'''
    # NOTE: brute force scipy cdist runs out of memory for large images
    # Kmeans without fit does nearest neighbor assignement but seems to better handle memory usage

    if mask is None:
        mask = np.ones(embeddings.shape[:-1], dtype=bool)

    labels = np.zeros(embeddings.shape[:-1], dtype=int)

    if len(centers) > 0:
        clusterer = KMeans(n_clusters=len(centers),
                           init=np.asarray(centers),
                           n_init=1)
        if refine_centers:
            clusterer.fit(embeddings[mask])
        else:
            # hack to init clusterer
            clusterer.fit(np.asarray(centers))

        labels[mask] = clusterer.predict(embeddings[mask]) + 1

    return labels


# TODO update test
def embeddings_to_labels(
        embeddings,
        fg_mask,
        spacing=1.,
        peak_min_distance=2,
        min_count=5,
        n_instance_max=500,
        return_score=False,
        return_votes=False,
):
    '''Converts a voting embeddings to labels.
    
    Notes:
    embeddings obtained from semi-conv is expected to be in isotropic coords
    An instance voting score of 1.0 means that all votes fell in the same bin (i.e. same pixel/voxel)

    '''

    # TODO flexible binning (possibly independent of image size)?
    # TODO nearest nb instead of hacky use of kmeans

    spacing = np.broadcast_to(np.asarray(spacing), embeddings.shape[-1])

    # convert embeddings to pixel coords and bin --> bin size = voxel
    embeddings_px = np.round(embeddings / spacing).astype(np.int16)
    votes = accumulate_votes(embeddings_px, fg_mask).astype(np.float32)
    seeds, seeds_intensities = local_max(votes,
                                         min_distance=peak_min_distance,
                                         threshold=min_count,
                                         spacing=spacing)
    seeds = seeds[:n_instance_max]
    seeds_intensities = seeds_intensities[:n_instance_max]
    # convert seeds to isotropic for nearest neighbhor assignments
    seeds = np.asarray(seeds) * spacing[None]

    labels = kmeans_cluster_embeddings(embeddings, seeds, mask=fg_mask)

    labels, _, _ = relabel_sequential(labels)

    outputs = {}
    if return_score:
        # normalize seed score by instance area/volume
        _, label_size = np.unique(labels, return_counts=True)
        outputs['voting_score'] = seeds_intensities / label_size[1:]
    if return_votes:
        outputs['votes'] = votes

    outputs['labels'] = labels
    return outputs


# TODO write test
def seeded_embeddings_to_labels(
        embeddings,
        fg_mask,
        seeds,
):
    '''Converts a voting embeddings to labels.'''

    labels = kmeans_cluster_embeddings(embeddings,
                                       embeddings[tuple(seeds.T)],
                                       mask=fg_mask)

    # split instance mask and keep the one touching seed
    seed_map = np.zeros_like(labels)
    seed_map[tuple(seeds.T)] = list(range(1, len(seeds) + 1))

    locs = find_objects(labels)
    for idx, loc in enumerate(locs, start=1):
        if loc:
            mask = labels[loc] == idx
            split_labels, n_splits = nd_label(mask)
            for sl in range(1, n_splits + 1):
                split_mask = split_labels == sl
                labels[loc][split_mask] = seed_map[loc][split_mask].max()

    return {'labels': labels}
