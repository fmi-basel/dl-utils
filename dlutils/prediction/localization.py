from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range, zip, dict

from dlutils.prediction.stitching import StitchingGenerator
from dlutils.models.utils import get_batch_size, get_patch_size

from scipy.spatial.distance import cdist
import numpy as np


class Anchor(object):
    '''
    '''

    def __init__(self, x, y, x_offset, y_offset, confidence):
        self.x = x
        self.y = y
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.confidence = confidence

    def get_coordinates(self):
        return self.x + self.x_offset, self.y + self.y_offset

    def __str__(self):
        '''
        '''
        return 'Anchor at xy=({},{}), offset=({}, {})'.format(
            self.x, self.y, self.x_offset, self.y_offset)

    def __repr__(self):
        return str(self)


def localize(model, image, threshold=0.5):
    '''localize detections from fully-convolutional single-shot detection model.

    TODO Parameters
    TODO Returns

    '''
    # TODO handle non-isotropic grid_step
    grid_step = model.input_shape[1] / model.output_shape[0][1]

    def inv_transform(coords, offset):
        return ((x + 0.5) * grid_step + ox for x, ox in zip(coords, offset))

    batch_size = get_batch_size(model)
    if batch_size is None:
        batch_size = 10

    anchors = []
    generator = StitchingGenerator(
        image,
        patch_size=get_patch_size(model),
        batch_size=batch_size,
        border=int(grid_step))

    for img_batch, coord_batch in (
        (batch['input'], batch['coord'])
            for batch in (generator[idx] for idx in range(len(generator)))):
        pred_batch = dict(
            zip(model.output_names, model.predict_on_batch(img_batch)))

        for idx, coord in enumerate(coord_batch):
            mask = pred_batch['anchor_pred'][idx].squeeze() >= threshold

            if not np.any(mask):
                continue

            for x, y in zip(*np.where(mask)):
                confidence = pred_batch['anchor_pred'][idx, x, y].squeeze()
                x_offset, y_offset = pred_batch['offset_pred'][idx, x, y]
                x, y = inv_transform([x, y], coord)
                anchors.append(
                    Anchor(
                        x=x,
                        y=y,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        confidence=confidence))

    return anchors


def nms(anchors, threshold):
    '''non-maximum suppression of detection anchors.

    TODO docs

    '''
    sorted_anchors = sorted(anchors, key=lambda x: -x.confidence)
    locations = np.asarray([(anchor.x + anchor.x_offset,
                             anchor.y + anchor.y_offset)
                            for anchor in sorted_anchors])
    dist = cdist(locations, locations, metric='cityblock')

    marked = [
        False,
    ] * len(sorted_anchors)
    marked[0] = True
    for idx in range(1, len(sorted_anchors)):
        marked[idx] = not any(
            (marked[better_idx] and dist[idx, better_idx] <= threshold
             for better_idx in range(0, idx)))
    return [anchor for idx, anchor in enumerate(sorted_anchors) if marked[idx]]
