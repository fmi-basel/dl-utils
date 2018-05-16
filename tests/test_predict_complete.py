from dlutils.prediction import predict_complete

from dlutils.models.fcn_resnet import ResnetBase
from dlutils.models.utils import add_fcn_output_layers

import numpy as np
import cv2


def main():

    input_shape = (350, 350, 1)
    batch_size = 5
    model = ResnetBase(input_shape=input_shape)

    pred_names = ['pred_cell', 'pred_border']
    model = add_fcn_output_layers(model, pred_names, [1, 1])

    model.load_weights('/tmp/model_latest.h5')

    model.compile(
        optimizer='adam',
        loss={
            'pred_cell': 'binary_crossentropy',
            'pred_border': 'mean_absolute_error'
        })

    #image = np.random.randn(384, 384, 1)
    image = cv2.imread('/tmp/141007_SMsiRNA_C04_T0001F001L01A03Z01C01.png', -1)

    prediction = predict_complete(model, image, batch_size=5, border=30)

    for key, val in prediction.iteritems():
        print key, val.shape

    import matplotlib.pyplot as plt
    _, axarr = plt.subplots(1, 3)
    for ax, img in zip(axarr, [
            image,
    ] + prediction.values()):
        ax.imshow(img.squeeze())
    plt.show()


if __name__ == '__main__':
    main()
