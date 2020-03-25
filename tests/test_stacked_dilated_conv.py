import itertools

import numpy as np
import tensorflow as tf
import pytest

from dlutils.layers.stacked_dilated_conv import StackedDilatedConv

# TODO complete similar to semi conv test with save/load


def test_instantiation():

    sd_conv = StackedDilatedConv(filters=4, kernel_size=3, rank=2)
