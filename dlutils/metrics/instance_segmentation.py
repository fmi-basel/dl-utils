from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment

# TODO
# double check when background no present


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


def symmetric_best_dice(y_true, y_pred, bg_label=0):
    '''Compute the symetric average best dice score as defined in:
    
    Scharr, Hanno, et al. "Leaf segmentation in 
    plant phenotyping: a collation study." Machine 
    vision and applications 27.4 (2016): 585-606.
    
    Dice score: 2 TP / (2 TP + FP + FN)

    '''

    y_true, _, _ = relabel_sequential(y_true)
    y_pred, _, _ = relabel_sequential(y_pred)
    y_true_sum = np.bincount(y_true.flat)
    y_pred_sum = np.bincount(y_pred.flat)

    # TODO handle empty annotations.

    # calculate sum and intersection
    area_sum = y_true_sum[..., None] + y_pred_sum
    intersection = confusion_matrix(y_true.flat, y_pred.flat)
    intersection = intersection[:area_sum.shape[0], :area_sum.shape[1]]

    # remove bg_label rows/columns
    for axis in range(2):
        area_sum = np.delete(area_sum, bg_label, axis)
        intersection = np.delete(intersection, bg_label, axis)

    dice_matrix = 2 * intersection.astype(float) / area_sum

    return np.minimum(
        dice_matrix.max(axis=0).mean(),
        dice_matrix.max(axis=1).mean())


def aggregated_jaccard_index(y_true, y_pred, bg_label=0):
    '''Aggregated Jaccard Index (AJI)

    '''

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not (y_true == bg_label).any() or not (y_pred == bg_label).any():
        raise ValueError('case not handled: no background label present')

    y_true, _, _ = relabel_sequential(y_true)
    y_pred, _, _ = relabel_sequential(y_pred)
    y_true_sum = np.bincount(y_true.flat)
    y_pred_sum = np.bincount(y_pred.flat)

    # calculate union and intersection
    union = y_true_sum[..., None] + y_pred_sum
    intersection = confusion_matrix(y_true.flat, y_pred.flat)
    intersection = intersection[:union.shape[0], :union.shape[1]]
    union = union - intersection

    # remove bg_label rows/columns
    for axis in range(2):
        union = np.delete(union, bg_label, axis)
        intersection = np.delete(intersection, bg_label, axis)

    # TODO use delete?
    y_true_sum = y_true_sum[[
        i for i in range(len(y_true_sum)) if i != bg_label
    ]]
    y_pred_sum = y_pred_sum[[
        i for i in range(len(y_pred_sum)) if i != bg_label
    ]]

    # index of best match for each groundtruth object
    IoU = intersection.astype(float) / union
    gt_best_idx = np.argmax(IoU, axis=1)

    # index of segmented objects that aren't best match for any objects
    # ""best match" with zero itnersection are considered not matched
    unused_idx = list(
        set(range(IoU.shape[1])) - set(gt_best_idx[np.max(IoU, axis=1) > 0]))

    aggregated_intesection = np.take_along_axis(intersection,
                                                gt_best_idx[:, None],
                                                axis=1)
    # take union of best if overlap > 0, else take cardinality of gt
    aggregated_union = np.where(
        aggregated_intesection <= 0, y_true_sum[:, None],
        np.take_along_axis(union, gt_best_idx[:, None], axis=1))

    aggregated_intesection = aggregated_intesection.sum()
    aggregated_union = aggregated_union.sum()
    aggregated_union += y_pred_sum[unused_idx].sum()

    return aggregated_intesection / aggregated_union


def measure_image_scores(y_true, y_pred, overlap_thresholds=None):
    '''Measures various metrics for a single pair of grountruth, prediction images.
    '''

    if overlap_thresholds is None:
        overlap_thresholds = np.linspace(0.05, 0.95, 19)[::-1]
    else:
        overlap_thresholds = sorted(overlap_thresholds)[::-1]

    iou_matrix = instance_intersection_over_union(y_true, y_pred)
    dice_matrix = symmetric_best_dice(y_true, y_pred)

    SBD = symmetric_best_dice(y_true, y_pred)
    AJI = aggregated_jaccard_index(y_true, y_pred)

    scores = []
    for t in overlap_thresholds:
        matches = iou_matrix > t

        # bibaritite matching? slow and only makes a difference below 0.5 thresh
        # ~lap_matching = linear_sum_assignment(-iou_matrix)
        # ~tp = matches[lap_matching].sum()

        tp = min(matches.max(axis=axis).sum() for axis in range(2))
        fp = float(matches.shape[1]) - tp
        fn = float(matches.shape[0]) - tp

        sc = {
            'o_thresh': t,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / (tp + fp) if tp > 0 else 0,
            'recall': tp / (tp + fn) if tp > 0 else 0,
            'stardistAP': tp / (tp + fp + fn) if tp > 0 else 0,
            'SBD': SBD,
            'AJI': AJI
        }  # threshold independent, repeated for each threshold value, consider handling separetly if more like it

        sc['f1'] = (2 * sc['precision'] * sc['recall'] /
                    (sc['precision'] + sc['recall'])) if tp > 0 else 0

        scores.append(sc)

    return pd.DataFrame(scores)
