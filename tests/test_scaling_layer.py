import numpy as np
import tensorflow as tf
import pytest

from dlutils.layers.input_scaler import ScaleAndClipLayer


@pytest.mark.parametrize('batch_size, lower, upper', [(5, -1, 1), (3, 1, 5.),
                                                      (1, -3.3, -0.1)])
def test_scale_and_clip(batch_size, lower, upper):
    '''test scaling layer
    '''

    model = tf.keras.models.Sequential(
        [ScaleAndClipLayer(lower=lower, upper=upper, input_shape=(None, 1))])
    model.compile(loss='mae')
    model.summary()

    np.random.seed(13)
    vals = np.random.randn(batch_size, 100, 1) * 5
    scaled_vals = (vals - lower) / (upper - lower)
    output = model.predict(vals)

    assert vals.shape == output.shape

    assert output.min() >= 0.
    assert output.max() <= 1.

    assert np.all(output[vals <= lower] == 0)
    assert np.all(output[vals >= upper] == 1.)

    mask = np.logical_and(vals > lower, vals < upper)

    np.testing.assert_allclose(output[mask], scaled_vals[mask], atol=1e-6)
