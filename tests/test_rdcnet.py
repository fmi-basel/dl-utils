import tensorflow as tf
import pytest
import numpy as np
from itertools import product

from dlutils.models.rdcnet import GenericRDCnetBase
from dlutils.models.heads import add_instance_seg_heads

# TODO complete test with parameters + markslow (if slow) similar to hourglass


def test_GenericRDCnetBase():

    input_shape = (16, 16, 3)
    batch_size = 2

    model = GenericRDCnetBase(input_shape,
                              downsampling_factor=2,
                              n_downsampling_channels=4,
                              n_output_channels=7)

    model = add_instance_seg_heads(model, n_classes=5)

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    assert pred[0].shape == (batch_size, ) + input_shape[:-1] + (2, )
    assert pred[1].shape == (batch_size, ) + input_shape[:-1] + (5, )


if __name__ == '__main__':

    pass
