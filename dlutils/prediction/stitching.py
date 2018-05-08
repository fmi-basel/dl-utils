from dlutils.models.utils import get_batch_size
from dlutils.models.utils import get_patch_size
from dlutils.models.utils import get_input_channels

from keras.utils import Sequence

import numpy as np


class StitchingGenerator(Sequence):
    def __init__(self, image, batch_size, patch_size, border):
        '''
        '''
        self.image = image
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.border = border
        self.calc_corners()

    def __len__(self):
        '''determine the number of batches needed.

        '''
        return int(np.ceil(len(self.corners) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''generates the ith-batch of patches.

        '''
        # TODO integrate pre-selection of patches to only process those
        # patches that contain foreground
        if idx > len(self):
            raise IndexError('idx {} out of range {}'.format(idx, len(self)))
        batch_end = min((idx + 1) * self.batch_size, len(self.corners))
        coord_batch, img_batch = zip(*[(
            (i, j),
            self.image[i:i + self.patch_size[0], j:j + self.patch_size[1], ...]
        ) for i, j in self.corners[idx * self.batch_size:batch_end]])

        return dict(input=np.asarray(img_batch), coord=np.asarray(coord_batch))

    def calc_corners(self):
        '''
        '''
        # corners of patches.
        step_size = np.asarray(self.patch_size) - 2 * self.border
        x = range(0, self.image.shape[0] - self.patch_size[0], step_size[0])
        y = range(0, self.image.shape[1] - self.patch_size[1], step_size[1])
        x.append(self.image.shape[0] - self.patch_size[0])
        y.append(self.image.shape[1] - self.patch_size[1])
        self.corners = [(i, j) for i in x for j in y]


def predict_complete(model, image, batch_size=None, patch_size=None,
                     border=10):
    '''apply model to entire image.

    '''
    if batch_size is None:
        batch_size = get_batch_size(model)
    if patch_size is None:
        patch_size = get_patch_size(model)
    n_channels = get_input_channels(model)

    if n_channels == 1 and image.shape[-1] != 1:
        image = image[..., None]

    if border > 0:
        pad_width = [(
            border,
            border,
        ) for idx in xrange(len(patch_size))] + [
            (0, 0),
        ]
        image = np.pad(image, pad_width=pad_width, mode='symmetric')

    # predict on each patch.
    # TODO consider stitching and prediction concurrently.
    # TODO allow for prediction-time-augmentation
    generator = StitchingGenerator(
        image, patch_size=patch_size, batch_size=batch_size, border=border)

    responses = dict(
        (name, np.zeros(image.shape[:-1] + (out_shape[-1], )))
        for name, out_shape in zip(model.output_names, model.output_shape))

    for img_batch, coord_batch in (
        (batch['input'], batch['coord'])
            for batch in (generator[idx] for idx in xrange(len(generator)))):

        # predict
        pred_batch = model.predict_on_batch(img_batch)

        # re-assemble
        for idx, coord in enumerate(coord_batch):
            slices = [
                slice(x + border, x + dx - border)
                for x, dx in zip(coord, patch_size)
            ]

            for key, pred in zip(model.output_names, pred_batch):
                if len(slices) == 2:
                    # TODO implement smooth stitching.
                    responses[key][slices[0], slices[1], ...] = pred[
                        idx][border:-border,
                             border:-border]
                else:
                    raise NotImplementedError(
                        'Stitching for {}-dimensional images not implemented.'.
                        format(len(slices)))

    if border > 0:
        for key, val in responses.iteritems():
            responses[key] = val[border:-border, border:-border, ...]

    return responses
