import pytest
import tensorflow as tf
import numpy as np
import h5py

from dlutils.models import load_model
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from dlutils.training.callbacks import ModelConfigSaver


def build_simple_model():
    '''keras-only model.

    '''
    input = tf.keras.layers.Input(batch_shape=(None, None, None, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(input)
    x = tf.keras.layers.Conv2D(1,
                               kernel_size=1,
                               activation='sigmoid',
                               name='out')(x)
    return tf.keras.models.Model(inputs=input, outputs=[x])


def build_custom_model():
    '''keras model wrapped with two custom layers from dlutils.

    '''
    input = tf.keras.layers.Input(batch_shape=(None, None, None, 1))
    x = DynamicPaddingLayer(factor=8)(input)
    model = build_simple_model()
    x = model(x)
    x = DynamicTrimmingLayer()([input, x])
    return tf.keras.models.Model(inputs=input, outputs=[x])


def compare_models(left, right):
    '''utility to compare two models.

    '''
    assert left.get_config() == right.get_config()
    for left_w, right_w in zip(left.get_weights(), right.get_weights()):
        np.testing.assert_allclose(left_w, right_w)


@pytest.mark.parametrize('model_constructor',
                         [build_simple_model, build_custom_model])
def test_default_save_load_h5(tmpdir, model_constructor):
    '''test saving/loading with default as h5.

    '''
    model = model_constructor()
    model.summary()

    # dump model.
    output_path = tmpdir / 'model.h5'
    tf.keras.models.save_model(model, str(output_path))
    assert output_path.exists()

    # test if it's really an hdf5 model
    if not h5py.is_hdf5(str(output_path)):
        raise RuntimeError('Model was not saved as hdf5!')

    # reload it.
    loaded_model = tf.keras.models.load_model(output_path,
                                              custom_objects={
                                                  'DynamicPaddingLayer':
                                                  DynamicPaddingLayer,
                                                  'DynamicTrimmingLayer':
                                                  DynamicTrimmingLayer
                                              })
    loaded_model.summary()

    # compare
    compare_models(model, loaded_model)


@pytest.mark.parametrize('model_constructor',
                         [build_simple_model, build_custom_model])
def test_model_config_saver(tmpdir, model_constructor):
    '''basic test of model IO with ModelConfigSaver

    '''
    model = model_constructor()
    model.summary()

    # dump model.
    architecture_path = tmpdir / 'model_architecture.yaml'
    weights_path = tmpdir / 'model_latest.h5'

    callback = ModelConfigSaver(str(architecture_path))
    callback.model = model  # mock patch
    callback.on_train_begin()
    assert architecture_path.exists()

    model.save_weights(str(weights_path))
    assert weights_path.exists()

    # reload it.
    loaded_model = load_model(str(weights_path))
    loaded_model.summary()

    # compare
    compare_models(model, loaded_model)
