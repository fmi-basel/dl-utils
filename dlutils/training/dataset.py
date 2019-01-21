from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging
import numpy as np

from dlutils.training.generator import LazyTrainingHandle
from dlutils.preprocessing.normalization import standardize
from dlutils.preprocessing.crop import crop_object
from dlutils.training.augmentations import ImageDataAugmentation
from dlutils.training.generator import TrainingGenerator
from dlutils.training.split import split
from dlutils.training.targets import generate_separator_map
from dlutils.training.targets import generate_distance_transform

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

    def __init__(self, img_path, segm_path, patch_size, *args, **kwargs):
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
            min_scale=50)  # NOTE consider adjusting min_scale to your dataset
            
        # load segmentation and make sure it's binary
        self['fg_pred'] = imread(self['segm_path']) >= 1
        
        # add flat channel if needed
        for key in self.get_input_keys() + self.get_output_keys():
            if self[key].ndim == len(self.patch_size):
                self[key] = self[key][..., None]


class InstanceSegmentationHandleWithSeparator(LazyTrainingHandle):
    def get_input_keys(self):
        '''returns a list of input keys.

        '''
        return ['input']

    def get_output_keys(self):
        '''returns a list of output keys.

        '''
        return ['fg_pred', 'separator_pred']

    def __init__(self, img_path, segm_path, patch_size, *args, **kwargs):
        '''initializes handle with source paths and patch_size for sampling.

        '''
        self['img_path'] = img_path
        self['segm_path'] = segm_path
        self.patch_size = patch_size

    def load(self):
        '''
        '''
        if self.is_loaded():
            return

        self['input'] = imread(self['img_path'])
        self['input'] = standardize(self['input'], min_scale=50)

        segm = imread(self['segm_path'])
        self['fg_pred'] = segm >= 1
        if segm.max() <= 0:
            self['separator_pred'] = np.zeros_like(segm)
        else:
            self['separator_pred'] = generate_separator_map(segm, reach=15)


class InstanceSegmentationHandleWithSeparatorMultislice(
        InstanceSegmentationHandleWithSeparator):
    def load(self):
        '''
        '''
        super(InstanceSegmentationHandleWithSeparatorMultislice, self).load()

        # move Z axis to last position and
        # we probably have to remove the flat dimension at the end.
        for key in self.get_input_keys() + self.get_output_keys():
            if self[key].shape[-1] == 1:
                self[key] = np.squeeze(self[key], axis=-1)
            self[key] = np.moveaxis(self[key], 0, -1)

    def get_random_patch(self, patch_size, *args, **kwargs):
        '''
        '''
        patches = super(InstanceSegmentationHandleWithSeparatorMultislice,
                        self).get_random_patch(patch_size, *args, **kwargs)

        # subselect middle plane of outputs
        for key in self.get_output_keys():
            patches[key] = patches[key][..., patch_size[-1] // 2][..., None]
        return patches


class InstanceSegmentationHandleWithDistanceMap(LazyTrainingHandle):
    def get_input_keys(self):
        '''returns a list of input keys.

        '''
        return ['input']

    def get_output_keys(self):
        '''returns a list of output keys.

        '''
        return ['fg_pred', 'transform_pred']

    def __init__(self, img_path, segm_path, patch_size, crop_margins, sampling):
        '''initializes handle with source paths and patch_size for sampling.

        '''
        self['img_path'] = img_path
        self['segm_path'] = segm_path
        self.patch_size = patch_size
        self.sampling = sampling
        self.crop_margins = crop_margins

    def load(self):
        '''
        '''
        if self.is_loaded():
            return

        self['input'] = imread(self['img_path'])
        self['input'] = standardize(self['input'], min_scale=50)

        segm = imread(self['segm_path']).astype(np.int, copy=False)
        self['fg_pred'] = segm >= 1
        
        self['transform_pred'] = generate_distance_transform(segm, 
                                                sampling=self.sampling,
                                                sigma=0.5)
        if self.crop_margins is not None:    
            self['input'], self['fg_pred'], self['transform_pred'] \
                                    = crop_object([self['input'], 
                                                   self['fg_pred'],
                                                   self['transform_pred']],
                                                   self['fg_pred'],
                                                   margins=self.crop_margins)
                                                                      
        # add flat channel if needed
        for key in self.get_input_keys() + self.get_output_keys():
            if self[key].ndim == len(self.patch_size):
                self[key] = self[key][..., None]

def prepare_dataset(path_pairs,
                    task_type,
                    patch_size,
                    split_ratio=0.2,
                    augmentation_params=None,
                    crop_margins=None,
                    sampling=None,
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
    elif task_type == 'instance_segmentation_distance_map':
        Handle = InstanceSegmentationHandleWithDistanceMap
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
        [Handle(*paths, patch_size=patch_size,
                crop_margins=crop_margins,
                sampling=sampling) for paths in path_pairs],
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
