from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging

from dlutils.training.generator import LazyTrainingHandle
from dlutils.preprocessing.normalization import standardize
from dlutils.training.augmentations import ImageDataAugmentation
from dlutils.training.generator import TrainingGenerator
from dlutils.training.split import split

from skimage.external.tifffile import imread


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

    Parameters
    ----------
    path_pairs : list of tuples
        list of pairs of (path_to_input_img, path_to_target_img) or
        three-tuples of (path_to_input_img, path_to_target_img, category).
        For the latter, the categories are used to stratify the training
        and validation split.
    '''
    if task_type == 'binary_segmentation':
        Handle = BinarySegmentationHandle
    elif task_type == 'instance_segmentation':
        Handle = InstanceSegmentationHandleWithSeparator
    else:
        raise ValueError('Unknown task_type: {}'.format(task_type))

    # prepare categories for split stratification if available.
    if all(len(paths) == 3 for paths in path_pairs):
        stratify = [label for _, _, label in path_pairs]
        path_pairs = [(input_path, target_path)
                      for input_path, target_path, _ in path_pairs]
    else:
        stratify = None

    train_handles, validation_handles = split(
        [Handle(*paths, patch_size=patch_size) for paths in path_pairs],
        split_ratio,
        stratify=stratify)

    logger = logging.getLogger(__name__)
    logger.info('Training samples: %i', len(train_handles))
    logger.info('Validation samples: %i', len(validation_handles))

    if config_params.get('buffer', False):
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
