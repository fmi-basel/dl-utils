from dlutils.postprocessing.voting import embeddings_to_labels
from dlutils.metrics.instance_segmentation import instance_intersection_over_union, _instance_precision_recall

from skimage.io import imread
import pytest


@pytest.mark.xfail
def test_embeddings_to_labels():
    '''Test 3D vector field converting to labels.
    
    Notes:
    This test only checks that the pipeline is not broken 
    by reapplying it to a visually validated sample.
    '''

    labels_target = imread('tests/data/labels.tif')
    vfield = imread('tests/data/vfield.tif')

    labels_pred = embeddings_to_labels(vfield,
                                       fg_mask=labels_target > 0,
                                       spacing=(2, 0.26, 0.26),
                                       peak_min_distance=2,
                                       min_count=5,
                                       n_instance_max=500,
                                       return_score=False,
                                       return_votes=False)['labels']

    iou_matrix = instance_intersection_over_union(labels_target, labels_pred)
    precision, recall = _instance_precision_recall(iou_matrix, threshold=.99)

    assert precision == pytest.approx(1., abs=0.001)
    assert recall == pytest.approx(1., abs=0.001)
