from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range

import logging
import numpy as np

from dlutils.training.generator import LazyTrainingHandle
from dlutils.preprocessing.normalization import standardize
from dlutils.training.augmentations import ImageDataAugmentation
from dlutils.training.generator import TrainingGenerator

from skimage.external.tifffile import imread


def split(samples, ratio, seed=13):
    '''split images into training and validation.

    '''
    assert 0 <= ratio <= 1.
    # shuffle
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(samples)
    split_idx = int((1. - ratio) * len(samples))
    return samples[:split_idx], samples[split_idx:]


class BinarySegmentationHandle(LazyTrainingHandle):
    def get_input_keys(self):
        '''returns a list of input keys
        '''
        return ['input']

    def get_output_keys(self):
        '''returns a list of output keys
        '''
        return ['fg_pred']

    def __init__(self, img_path, segm_path, patch_size):
        '''initializes handle with source paths and patch_size for sampling.

        '''
        self['img_path'] = img_path
        self['segm_path'] = segm_path
        self.patch_size = patch_size

    def load(self):
        '''actually loads data.

        '''
        if self.is_loaded():
            return

        # load image
        self['input'] = imread(self['img_path'])
        self['input'] = standardize(
            self['input'],
            min_scale=10)  # NOTE consider adjusting min_scale to your dataset

        # load segmentation and make sure it's binary
        self['fg_pred'] = imread(self['segm_path']) >= 1

        # add flat channel if needed
        for key in self.get_input_keys() + self.get_output_keys():
            if self[key].ndim == len(self.patch_size):
                self[key] = self[key][..., None]

    def clear(self):
        '''discard loaded data.

        '''
        for key in self.get_input_keys() + self.get_output_keys():
            self[key] = None

    def is_loaded(self):
        '''
        '''
        return all(
            self.get(key, None) is not None
            for key in self.get_input_keys() + self.get_output_keys())


def prepare_dataset(path_pairs,
                    task_type,
                    patch_size,
                    split_ratio=0.2,
                    augmentation_params=None,
                    **config_params):
    '''create a generator for training samples and validation samples.

    '''
    if task_type == 'binary_segmentation':
        Handle = BinarySegmentationHandle
    elif task_type == 'instance_segmentation':
        Handle = InstanceSegmentationHandleWithSeparator
    else:
        raise ValueError('Unknown task_type: {}'.format(task_type))

    train_handles, validation_handles = split(
        [Handle(*paths, patch_size=patch_size) for paths in path_pairs],
        split_ratio)

    logger = logging.getLogger(__name__)
    logger.info('Training samples: %i', len(train_handles))
    logger.info('Validation samples: %i', len(validation_handles))

    if config_params['buffer']:
        # preload.
        for handle in train_handles + validation_handles:
            handle.load()

    dataset = dict()
    dataset['training'] = TrainingGenerator(
        train_handles, patch_size=patch_size, **config_params)

    if augmentation_params is not None:
        dataset['training'].augmentator = ImageDataAugmentation(
            **augmentation_params)

    dataset['validation'] = TrainingGenerator(
        validation_handles, patch_size=patch_size, **config_params)
    return dataset
