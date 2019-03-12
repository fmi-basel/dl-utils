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
from dlutils.training.targets.distance_transform import generate_distance_transform, shrink_labels
from dlutils.training.targets.seeds import generate_seed_map
from dlutils.training.targets.location_map import generate_locationmap, generate_locationmap_target
from dlutils.training.targets.normalization_mask import generate_normalization_mask

from skimage.external.tifffile import imread

from keras.utils.np_utils import to_categorical

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
        self.patch_size = patch_sizse

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
        if self.is_loaded():
            return
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


class InstanceSegmentationHandleWithLocationMap(LazyTrainingHandle):
    def get_input_keys(self):
        '''returns a list of input keys.

        '''
        return ['input']

    def get_output_keys(self):
        '''returns a list of output keys.

        '''
        return ['fg_pred', 'segmentation']

    def __init__(self, img_path, segm_path, patch_size, sampling, locationmap_params, *args, **kwargs):
        '''initializes handle with source paths and patch_size for sampling.

        '''
        self['img_path'] = img_path
        self['segm_path'] = segm_path
        self.patch_size = patch_size
        self.sampling = sampling

        ndim = len(patch_size)
        self.period_bounds = np.broadcast_to(np.asarray(locationmap_params['period_bounds']), (ndim,2))
        self.offset_bounds = np.broadcast_to(np.asarray(locationmap_params['offset_bounds']), (ndim,2))

    def load(self):
        '''
        '''
        if self.is_loaded():
            return

        self['input'] = imread(self['img_path']).astype(np.float32)
        self['input'] = standardize(self['input'], min_scale=50)
        location_map = generate_locationmap(self['input'].shape) # will be replaced during patch sampling
        self['input'] = np.expand_dims(self['input'], axis=-1)
        self['input'] = np.concatenate([self['input'], location_map], axis=-1)

        segm = imread(self['segm_path']).astype(np.int, copy=False)
        self['fg_pred'] = segm>=1
        norm_mask = generate_normalization_mask(self['fg_pred'], include_background=True)
        self['fg_pred'] = np.stack([self['fg_pred'], norm_mask], axis=-1)
        
        self['segmentation'] = segm
                                                                      
        # add flat channel if needed
        for key in self.get_input_keys() + self.get_output_keys():
            if self[key].ndim == len(self.patch_size):
                self[key] = self[key][..., None]
        
    def get_random_patch(self, patch_size, *args, **kwargs):
        '''
        '''
    
        patches = super(InstanceSegmentationHandleWithLocationMap,
                        self).get_random_patch(patch_size, *args, **kwargs)

        # compute/randomize location map and corresponding target
        period = [np.random.uniform(*bound) for bound in self.period_bounds]
        offset = [np.random.uniform(*bound) for bound in self.offset_bounds]
        
        patches['input'][...,1:] = generate_locationmap(patches['input'][...,0].shape, period=period, offset=offset)
        patches['input'] = np.ascontiguousarray(patches['input'])
        
        patches['segmentation'] = generate_locationmap_target(patches['segmentation'][...,0], patches['input'][...,1:])
        patches['segmentation'] = np.ascontiguousarray(patches['segmentation'])
            
        return patches
        

def prepare_dataset(path_pairs,
                    task_type,
                    patch_size,
                    split_ratio=0.2,
                    augmentation_params=None,
                    task_params=None,
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
    elif task_type == 'instance_segmentation_location_map':
        Handle = InstanceSegmentationHandleWithLocationMap
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
                **task_params) for paths in path_pairs],
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
