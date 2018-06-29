from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range

from dlutils.training.generator import LazyTrainingHandle
from dlutils.preprocessing.normalization import standardize

import os
import numpy as np
from glob import glob


class ImageHandle(LazyTrainingHandle):
    def get_input_keys(self):
        '''returns a list of input keys
        '''
        return [
            'input',
        ]

    def get_output_keys(self):
        '''returns a list of output keys
        '''
        return ['cell_pred', 'border_pred']

    def __init__(self, img_path, segm_path, patch_size):
        '''
        '''
        self['img_path'] = img_path
        self['segm_path'] = segm_path
        self.patch_size = patch_size
        self.check()

    def check(self):
        '''make sure the filenames of image and segmentation have matching
        signatures.

        '''
        raise NotImplementedError()

    def load(self):
        '''
        '''
        raise NotImplementedError()
        self['input'] = standardize(self['input'])

    def clear(self):
        '''
        '''
        for key in self.get_input_keys() + self.get_output_keys():
            self[key] = None


def find_annotation(path, suffix):
    '''find the associated annotation image for a given image path.

    '''
    raise NotImplementedError()
    segm_path = path.replace('/Single/', '/labelMasks/')
    segm_path, ext = os.path.splitext(segm_path)
    segm_path = segm_path + suffix + ext
    if not os.path.exists(segm_path):
        return None
    return segm_path


def split(handles, ratio, seed=13):
    '''split images into training and validation.

    NOTE Sets random.seed to have a stable training/validation split.
    '''
    # shuffle
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(handles)

    split_idx = int((1. - ratio) * len(handles))
    return handles[:split_idx], handles[split_idx:]


def collect_handles(basedir, patch_size, split_ratio=0.2):
    '''gather all handles and split them into training and validation
    Default split is 20%.

    '''
    raise NotImplementedError()
    handles = [
        ImageHandle(img_path=img, segm_path=segm, patch_size=patch_size)
        for (img, segm) in ((
            path, find_annotation(path, suffix='_label')) for path in glob(
                os.path.join(basedir, 'Day*', 'obj*', 'Single', '*.tif')))
        if segm is not None
    ]

    return split(handles, ratio=split_ratio)
