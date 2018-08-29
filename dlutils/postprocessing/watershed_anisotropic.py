from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SimpleITK as itk

import numpy as np


def segment_nuclei(nuclei_pred,
                   border_pred,
                   threshold,
                   upper_threshold=None,
                   smoothness=1,
                   watershed_line=True,
                   spacing=None,
                   return_numpy=True):
    '''watershed based segmentation of nuclei.

    '''
    if isinstance(nuclei_pred, np.ndarray):
        if spacing is None:
            raise ValueError('spacing=None is not allowed when nuclei_pred is '
                             'not an itk::Image object')
        nuclei_pred = itk.Cast(
            itk.GetImageFromArray(nuclei_pred), itk.sitkFloat32)
        nuclei_pred.SetSpacing(spacing)

    if isinstance(border_pred, np.ndarray):
        if spacing is None:
            raise ValueError('spacing=None is not allowed when border_pred is '
                             'not an itk::Image object')
        border_pred = itk.Cast(
            itk.GetImageFromArray(border_pred), itk.sitkFloat32)
        border_pred.SetSpacing(spacing)

    # NOTE this is slightly different from the scipy version:
    # we smooth only the nuclei_pred
    combined = itk.SmoothingRecursiveGaussian(nuclei_pred, smoothness, True)
    combined = combined * (1.0 - border_pred)

    # maxima are going to be the seeds.
    maxima = itk.Cast(
        itk.RegionalMaxima(nuclei_pred, flatIsMaxima=False), itk.sitkUInt8)
    maxima = itk.And(combined >= threshold, maxima)

    # Filter out maxima
    if upper_threshold is not None:
        maxima = itk.Or(combined >= upper_threshold, maxima)
    markers = itk.ConnectedComponent(maxima)

    ws = itk.MorphologicalWatershedFromMarkers(
        -combined, markers, markWatershedLine=True, fullyConnected=True)
    segmentation = itk.Mask(
        ws, itk.Cast(nuclei_pred >= threshold, ws.GetPixelID()))
    if return_numpy:
        return itk.GetArrayFromImage(segmentation)
    return segmentation
