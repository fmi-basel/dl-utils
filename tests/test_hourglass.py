import tensorflow as tf
import pytest
import numpy as np
from itertools import product

from dlutils.models.hourglass import bottleneck_conv_block, hourglass_block, single_hourglass, delta_loop, GenericRecurrentHourglassBase
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer


def test_bottleneck_block():
    '''test bottleneck block instantiation and feed forward
    '''

    b_block = bottleneck_conv_block(channels=32, spatial_dims=2, norm_groups=4)

    input_tensor = tf.random.normal((3, 128, 128, 32))
    output_tensor = b_block(input_tensor)

    assert input_tensor.shape == output_tensor.shape


def test_double_bottleneck_blocks():
    '''test that separately instantiated bottleneck blocks don't share weights'
    '''

    b_block_A = bottleneck_conv_block(channels=32,
                                      spatial_dims=2,
                                      norm_groups=4)
    b_block_B = bottleneck_conv_block(channels=32,
                                      spatial_dims=2,
                                      norm_groups=4)

    input_tensor = tf.random.normal((3, 128, 128, 32))

    output_tensor_A = b_block_A(input_tensor)
    output_tensor_A_prime = b_block_A(input_tensor)
    output_tensor_B = b_block_B(input_tensor)

    assert np.allclose(output_tensor_A.numpy(),
                       output_tensor_A_prime.numpy(),
                       rtol=1e-5)
    assert not np.allclose(
        output_tensor_A.numpy(), output_tensor_B.numpy(), rtol=1e-5)


def test_hourglass_block():
    '''test hourglass block instantiation and feed forward
    '''

    b_block = hourglass_block(n_levels=4,
                              channels=32,
                              channels_growth=2,
                              spatial_dims=2,
                              spacing=1)

    input_tensor = tf.random.normal((3, 128, 128, 32))
    output_tensor = b_block(input_tensor)

    assert input_tensor.shape == output_tensor.shape


def test_single_hourglass():
    '''test single hourglass stack instantiation and feed forward
    '''

    hglass = single_hourglass(output_channels=7,
                              n_levels=4,
                              channels=32,
                              channels_growth=2,
                              spatial_dims=2,
                              spacing=1,
                              norm_groups=4)

    input_tensor = tf.random.normal((3, 128, 128, 1))
    output_tensor = hglass(input_tensor)

    assert output_tensor.shape == (3, 128, 128, 7)


def test_delta_loop():
    '''test delta_loop instantiation and feed forward
    '''

    n_steps = 3
    hglass = single_hourglass(output_channels=1,
                              n_levels=4,
                              channels=32,
                              channels_growth=2,
                              spatial_dims=2,
                              spacing=1,
                              norm_groups=4)
    recur_block = delta_loop(output_channels=1,
                             recurrent_block=hglass,
                             default_n_steps=n_steps)

    input_tensor = tf.random.normal((3, 128, 128, 3))

    # without initial state
    output_tensor, deltas = recur_block(input_tensor)
    assert output_tensor.shape == (n_steps, 3, 128, 128, 1)

    # with initial state
    state = tf.zeros((3, 128, 128, 1), input_tensor.dtype)
    output_tensor, deltas = recur_block(input_tensor, state=state)
    assert output_tensor.shape == (n_steps, 3, 128, 128, 1)

    # with extra iterations
    output_tensor, deltas = recur_block(input_tensor, n_steps=n_steps + 1)
    assert output_tensor.shape == (n_steps + 1, 3, 128, 128, 1)


params_2d = list(
    product([(3, 122, 128, 3), (3, 43, 67, 1)], [1, 3], [2, 4], [8, 16],
            [1, 1.5, 2], [2], [1]))
params_3d = list(
    product([(3, 16, 122, 128, 3)], [1, 3], [2, 4], [8, 16], [1, 2], [3],
            [1, (7, 1, 1)]))
hglass_options = [
    pytest.param(*p, marks=pytest.mark.slow) if idx > 0 else p
    for idx, p in enumerate(params_2d + params_3d)
]


@pytest.mark.parametrize(
    "input_shape,output_channels,n_levels,channels,channels_growth,spatial_dims,spacing",
    hglass_options)
def test_GenericRecurrentHourglassBase(tmpdir, input_shape, output_channels,
                                       n_levels, channels, channels_growth,
                                       spatial_dims, spacing):
    '''test recurrent hourglass instantiation, feed forward and changing the number of iterations/initial state
    '''

    n_steps = 3
    model = GenericRecurrentHourglassBase(
        tuple(None for _ in range(spatial_dims)) + input_shape[-1:],
        output_channels=output_channels,
        external_init_state=False,
        default_n_steps=n_steps,
        n_levels=n_levels,
        channels=channels,
        channels_growth=channels_growth,
        spatial_dims=spatial_dims,
        spacing=spacing,
        norm_groups=4)
    weights_path = tmpdir / 'model_latest.h5'
    model.save_weights(str(weights_path))

    input_tensor = tf.random.normal(input_shape)
    outputs, deltas = model(input_tensor)
    assert outputs.shape == (n_steps, ) + input_shape[:-1] + (
        output_channels, )

    ####################################################################
    # with initial state (and weights from model)
    model2 = GenericRecurrentHourglassBase(
        tuple(None for _ in range(spatial_dims)) + input_shape[-1:],
        output_channels=output_channels,
        external_init_state=True,
        default_n_steps=n_steps,
        n_levels=n_levels,
        channels=channels,
        channels_growth=channels_growth,
        spatial_dims=spatial_dims,
        spacing=spacing,
        norm_groups=4)
    model2.load_weights(str(weights_path))
    state = tf.zeros(input_shape[:-1] + (output_channels, ),
                     input_tensor.dtype)
    outputs, deltas = model2([input_tensor, state])
    assert outputs.shape == (n_steps, ) + input_shape[:-1] + (
        output_channels, )

    ####################################################################
    # with different number of iter (and weights from model)
    # NOTE could be dynamic if there was a way to pass the optional inpput/parameter to keras Model
    model3 = GenericRecurrentHourglassBase(
        tuple(None for _ in range(spatial_dims)) + input_shape[-1:],
        output_channels=output_channels,
        external_init_state=False,
        default_n_steps=n_steps + 2,
        n_levels=n_levels,
        channels=channels,
        channels_growth=channels_growth,
        spatial_dims=spatial_dims,
        spacing=spacing,
        norm_groups=4)
    model3.load_weights(str(weights_path))
    outputs, deltas = model3(input_tensor)

    assert outputs.shape == (n_steps +
                             2, ) + input_shape[:-1] + (output_channels, )


def get_dummy_dataset(n_samples, batch_size, repeats=None):
    '''Creates a dummy tensorflow dataset with random noise as input
    and a mask where input>0 as target.'''
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def gen():
        for i in range(n_samples):
            yield tf.random.normal((17, 23, 1))

    return (tf.data.Dataset.from_generator(
        gen, (tf.float32), output_shapes=(17, 23, 1)).map(
            lambda img: (img, tf.math.greater(img, 0.)),
            num_parallel_calls=AUTOTUNE).repeat(repeats).batch(batch_size))


def supervision_loss(y_true, y_preds):
    '''minimal supervision loss: average of intermediate outputs' losses'''

    # NOTE keras model.fit somehow squeezes the last dimension when outputs has an extra dim???
    y_preds = y_preds[..., None]

    def partial_loss_fct(y_pred):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true,
                                                                    y_pred)

    loss = tf.map_fn(partial_loss_fct, y_preds, tf.float32)

    return tf.reduce_mean(loss)


@pytest.mark.slow
def test_training_GenericRecurrentHourglassBase(tmpdir):
    '''tests recurrent hourglass training and saving'''

    input_tensor = tf.random.normal((4, 122, 128, 1))
    target = tf.math.greater(input_tensor, 0.)

    model = GenericRecurrentHourglassBase((None, None, 1),
                                          output_channels=1,
                                          external_init_state=False,
                                          default_n_steps=3,
                                          n_levels=4,
                                          channels=32,
                                          channels_growth=2,
                                          spatial_dims=2,
                                          spacing=1,
                                          norm_groups=4)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=[supervision_loss, None],
    )

    outputs_init, deltas = model(input_tensor)
    loss_init = supervision_loss(target, outputs_init)
    assert outputs_init.shape == (3, 4, 122, 128, 1)

    # train model
    model.fit(
        get_dummy_dataset(16, batch_size=4),
        validation_data=get_dummy_dataset(4, batch_size=4),
        epochs=10,
        steps_per_epoch=4,
        validation_steps=1,
    )

    outputs_trained, deltas = model(input_tensor)
    loss_trained = supervision_loss(target, outputs_trained)
    assert outputs_trained.shape == (3, 4, 122, 128, 1)

    # save and reload model
    output_path = tmpdir / 'model.h5'
    tf.keras.models.save_model(model, str(output_path))
    loaded_model = tf.keras.models.load_model(output_path,
                                              custom_objects={
                                                  'DynamicPaddingLayer':
                                                  DynamicPaddingLayer,
                                                  'DynamicTrimmingLayer':
                                                  DynamicTrimmingLayer,
                                                  'supervision_loss':
                                                  supervision_loss
                                              })

    outputs_reloaded, deltas = loaded_model(input_tensor)
    loss_reloaded = supervision_loss(target, outputs_reloaded)
    assert outputs_reloaded.shape == (3, 4, 122, 128, 1)

    assert not np.allclose(
        outputs_init.numpy(), outputs_trained.numpy(), rtol=1e-5)
    assert np.allclose(outputs_trained.numpy(),
                       outputs_reloaded.numpy(),
                       rtol=1e-5)

    assert loss_trained.numpy() + 0.1 < loss_init.numpy()
    np.testing.assert_almost_equal(loss_trained.numpy(), loss_reloaded.numpy())


if __name__ == '__main__':

    pass
