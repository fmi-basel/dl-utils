import warnings

from numba import njit
import numpy as np

from skimage.segmentation import relabel_sequential
from scipy.ndimage import find_objects
from scipy.ndimage import label as nd_label
from improc.feature import local_max
from scipy.ndimage import measurements


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
        z, x, y, t = abs_disp[i, 0], abs_disp[i, 1], abs_disp[i,
                                                              2], abs_disp[i,
                                                                           3]

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


from sklearn.neighbors import NearestNeighbors


def embeddings_nearest_neighbor(
    embeddings,
    centers,
    dist_threshold=None,
    mask=None,
):
    '''Clusters embeddings where mask is True and returns the labels.'''

    if mask is None:
        mask = np.ones(embeddings.shape[:-1], dtype=bool)

    labels = np.zeros(embeddings.shape[:-1], dtype=int)

    if len(centers) > 0:
        neigh = NearestNeighbors(n_neighbors=1, radius=0.4)
        neigh.fit(np.asarray(centers))

        dist, nn = neigh.kneighbors(embeddings[mask], return_distance=True)

        labels[mask] = nn.squeeze() + 1

        if dist_threshold:
            # set background where embeddings are too far from their center
            labels[mask] *= (dist.squeeze() < dist_threshold)

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

    labels = embeddings_nearest_neighbor(embeddings,
                                         seeds,
                                         mask=fg_mask,
                                         dist_threshold=6)
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
    dist_threshold=None,
    n_iter=1,
):
    '''Converts a voting embeddings to labels.
    
    dist_threshold: maximum distance from embeddings to its center. usefull to segment with partial seeds'''

    # TODO refine embeddigns instead of iterative seeds refinement
    # TODO test n_iter on nuclei seg
    # useful to refine instance center when initial seed is away from the center of mass (e.g. seed from nuclei in large cell at early timepoints)

    for i in range(n_iter):
        if i == 0:
            centers = embeddings[tuple(seeds.T)]

        else:
            unique_l = np.unique(labels)
            unique_l = unique_l[unique_l > 0]
            centers = [
                measurements.mean(embeddings[..., dim], labels, unique_l)
                for dim in range(embeddings.shape[-1])
            ]
            centers = np.stack(centers, axis=-1)

        labels = embeddings_nearest_neighbor(embeddings,
                                             centers,
                                             dist_threshold=dist_threshold,
                                             mask=fg_mask)

    return {'labels': labels}


def refine_embeddings(embeddings, spacing=1, n_iter=2):
    '''Iteratively resamples embeddings at voted location'''

    emb_ndim = embeddings.shape[-1]
    spacing = np.broadcast_to(np.asarray(spacing), emb_ndim)

    embeddings = np.round(embeddings / spacing).astype(np.int16)
    for dim in range(emb_ndim):
        embeddings[..., dim] = embeddings[...,
                                          dim].clip(0,
                                                    embeddings.shape[dim] - 1)

    for i in range(n_iter):
        embeddings[:] = embeddings[tuple(embeddings.reshape(
            -1, emb_ndim).T)].reshape(embeddings.shape)

    return embeddings * spacing
