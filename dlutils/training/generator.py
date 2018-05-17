from keras.utils import Sequence

import numpy as np

from dlutils.training.sampling import get_random_patch


class LazyTrainingHandle(dict):
    '''custom dict to manage input and target maps.

    '''

    def __init__(self, **kwargs):
        raise NotImplementedError('Implement initializer for your dataset!')

    def load(self, **kwargs):
        raise NotImplementedError('Implement load(..) for your dataset!')

    def get_random_patch(self, patch_size, augmentator=None):
        '''
        '''
        return dict(
            zip(self.get_input_keys() + self.get_output_keys(),
                get_random_patch(
                    [
                        self[key] for key in
                        self.get_input_keys() + self.get_output_keys()
                    ],
                    patch_size=patch_size,
                    augmentator=augmentator
                    )))

    def get_input_keys(self):
        '''returns a list of input keys
        '''
        raise NotImplementedError('implement input keys for your dataset!')

    def get_output_keys(self):
        '''returns a list of output keys
        '''
        raise NotImplementedError('implement output keys for your dataset!')

    def clear(self):
        '''clears any input/output data attributes.

        '''
        for key in self.get_input_keys() + self.get_output_keys():
            self[key] = None


class TrainingGenerator(Sequence):
    '''generates random patches from a set of training images and
    their annotations.

    Notes
    -----
    The last batch is discarded if incomplete.
    Make sure to choose `samples_per_handle` large enough to
    allow at least one complete batch.

    '''

    def __init__(self,
                 handles,
                 patch_size,
                 batch_size,
                 seed=None,
                 buffer=False,
                 samples_per_handle=1,
                 augmentator=None):
        '''
        '''
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.handles = handles
        self.samples_per_handle = samples_per_handle
        self.buffer = buffer
        self.augmentator = augmentator

        if len(self) <= 0:
            print len(self.handles) * self.samples_per_handle, self.batch_size
            raise ValueError(
                'Generator can have zero length! Increase samples_per_handle or decrease batch_size. {} < {}'.
                format(
                    len(self.handles) * self.samples_per_handle,
                    self.batch_size))

    def __len__(self):
        return int((len(self.handles) * self.samples_per_handle) / float(
            self.batch_size))

    def __getitem__(self, idx):
        '''return idx-th batch of size batch_size

        '''
        if idx >= len(self):
            raise IndexError('Index out of bounds {} >= {}'.format(
                idx, len(self)))

        inputs = {key: [] for key in self.handles[0].get_input_keys()}
        outputs = {key: [] for key in self.handles[0].get_output_keys()}

        # indexing wraps around to support batches of larger size than
        # len(handles) when samples_per_handle > 1
        for handle in (self.handles[idx % len(self.handles)]
                       for idx in xrange(self.batch_size)):
            handle.load()  # make sure data is available.
            patches = handle.get_random_patch(self.patch_size,
                                              self.augmentator)

            for key in inputs.iterkeys():
                inputs[key].append(patches[key])
            for key in outputs.iterkeys():
                outputs[key].append(patches[key])

            if not self.buffer:
                handle.clear()

        # Turn each list into a numpy array
        for key in inputs.iterkeys():
            inputs[key] = np.asarray(inputs[key])

        for key in outputs.iterkeys():
            outputs[key] = np.asarray(outputs[key])

        return inputs, outputs

    def on_epoch_end(self):
        '''shuffle data
        '''
        np.random.shuffle(self.handles)
