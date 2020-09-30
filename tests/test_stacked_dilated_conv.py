import itertools

import numpy as np
import tensorflow as tf
import pytest

from dlutils.layers.stacked_dilated_conv import StackedDilatedConv
from tensorflow.keras.layers import LeakyReLU

# TODO complete similar to semi conv test with save/load


def test_instantiation():

    sd_conv = StackedDilatedConv(rank=2,
                                 filters=4,
                                 kernel_size=3,
                                 dilation_rates=(1, 2, 4),
                                 groups=1,
                                 activation=LeakyReLU())


def test__concat_interleaved_groups():

    # group id: 1, 2, 3
    # dilation rate id: 10, 20

    o = np.ones((3, 4, 1), dtype=np.int32)
    dilated_outs = [
        np.stack([o + 10, 2 * o + 10, 3 * o + 10], axis=-1),
        np.stack([o + 20, 2 * o + 20, 3 * o + 20], axis=-1)
    ]

    sd_conv = StackedDilatedConv(rank=2,
                                 filters=4,
                                 kernel_size=3,
                                 dilation_rates=(1, 2, 4),
                                 groups=3)

    first_px_res = sd_conv._concat_interleaved_groups(
        dilated_outs)[0, 0].numpy().squeeze()
    assert all(first_px_res == [11, 21, 12, 22, 13, 23])
