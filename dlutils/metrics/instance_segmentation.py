from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range

import numpy as np

from sklearn.metrics import confusion_matrix
from skimage.segmentation import relabel_sequential


def instance_intersection_over_union(y_true, y_pred, bg_label=0):
    '''calculate instance precision and recall as in:

    [1] Harihan et al. Simultaneous detection and segmentation, ECCV 2014
    [2] Cordts et al. The cityscapes dataset for semantic urban scene
        understanding, CVPR 2016

    Predicted and annotated instances are matched as "True Positive (TP)"
    if and only if their intersection over union is larger than the
    give ratio `overlap`.

    '''

    y_true, _, _ = relabel_sequential(y_true)
    y_pred, _, _ = relabel_sequential(y_pred)
    y_true_sum = np.bincount(y_true.flat)
    y_pred_sum = np.bincount(y_pred.flat)

    # TODO handle empty annotations.

    # calculate union and intersection
    union = y_true_sum[..., None] + y_pred_sum
    intersection = confusion_matrix(y_true.flat, y_pred.flat)
    intersection = intersection[:union.shape[0], :union.shape[1]]
    union = union - intersection

    # remove bg_label rows/columns
    for axis in range(2):
        union = np.delete(union, bg_label, axis)
        intersection = np.delete(intersection, bg_label, axis)

    return intersection.astype(float) / union


def _instance_precision_recall(iou_matrix, threshold):
    '''calculates instance precision and recall at given threshold.

    '''
    matches = iou_matrix > threshold
    true_positives = min(matches.max(axis=axis).sum() for axis in range(2))
    precision = true_positives / float(matches.shape[1])
    recall = true_positives / float(matches.shape[0])
    return precision, recall


def instance_prauc(y_true, y_pred, overlap_thresholds=None):
    '''
    '''
    if overlap_thresholds is None:
        overlap_thresholds = np.linspace(0.05, 0.95, 19)[::-1]
    else:
        overlap_thresholds = sorted(overlap_thresholds)[::-1]

    iou = instance_intersection_over_union(y_true, y_pred)

    if iou.shape[0] == 0:
        if iou.shape[1] == 0:
            return 1.  # perfect.
        else:
            return 0.  # total disaster.

    recall = [
        0,
    ]
    precision = []
    for overlap in overlap_thresholds:
        matches = iou > overlap
        true_positives = min(matches.max(axis=axis).sum() for axis in range(2))
        precision.append(true_positives / float(matches.shape[1]))
        recall.append(true_positives / float(matches.shape[0]))

    average_precision = np.sum(np.diff(recall) * np.asarray(precision))
    return average_precision
