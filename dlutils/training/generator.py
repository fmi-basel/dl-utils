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

    def get_random_patch(self, patch_size, **augmentation_params):
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
                    **augmentation_params)))

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
    The incomplete batch at the end is discarded.

    '''

    def __init__(self, handles, patch_size, batch_size, seed=None, buffer=False):
        '''
        '''
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.handles = handles
        self.buffer = buffer
        self.augmentation_params = dict()

    def __len__(self):
        return int(len(self.handles) / float(self.batch_size))

    def __getitem__(self, idx):
        '''return idx-th batch of size batch_size

        '''
        assert idx < len(self)

        inputs = dict()
        for key in self.handles[0].get_input_keys():
            inputs[key] = []

        outputs = dict()
        for key in self.handles[0].get_output_keys():
            outputs[key] = []

        for handle in self.handles[idx * self.batch_size:(
                idx + 1) * self.batch_size]:
            handle.load()  # make sure data is available.
            patches = handle.get_random_patch(self.patch_size,
                                              **self.augmentation_params)

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
