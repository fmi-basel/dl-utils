from dlutils.postprocessing.embedding_clustering import embeddings_to_labels
from dlutils.preprocessing.normalization import min_max_scaling

from skimage.external.tifffile import imread, imsave
import numpy as np
import pytest


from dlutils.metrics.instance_segmentation import instance_intersection_over_union, _instance_precision_recall

def test_simple_clustering():
    '''Test embeddings clustering with color labels (3 channels embeddings)
    '''
    
    labels_target = imread('tests/data/labels.tif')
    embeddings = imread('tests/data/embeddings.tif')
    embeddings = min_max_scaling(embeddings, -1, 1)
    fg_mask = labels_target>0
    
    labels_pred = embeddings_to_labels(embeddings, fg_mask, coordinate_weight=0.001, sampling=(2,0.26,0.26), size_threshold=300, sliced=False)
    
    iou_matrix = instance_intersection_over_union(labels_target, labels_pred)
    precision, recall = _instance_precision_recall(iou_matrix, threshold=.99)
    
    assert precision == pytest.approx(1., abs=0.001)
    assert recall == pytest.approx(1., abs=0.001)
