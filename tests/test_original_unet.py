from dlutils.models.unet import UnetBase
from dlutils.models.utils import add_fcn_output_layers

import numpy as np
import pytest


@pytest.yield_fixture(autouse=True)
def cleanup():
    '''
    '''
    # make sure models are gone after each test.
    from keras.backend import clear_session
    clear_session()


@pytest.mark.parametrize("input_shape", [
    (250, 250, 1), (250, 271, 1)])
def test_unet_setup(input_shape):
    '''
    '''
    batch_size = 3

    model = UnetBase(input_shape=input_shape, with_bn=True, dropout=0.5)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    model.summary()

    # make sure the feed forward path works.
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    for name, pred in zip(pred_names, pred):
        # TODO include check for proper dimensionality.
        print name, pred.shape


if __name__ == '__main__':
    main()
