from keras import backend as K
import tensorflow as tf

from tensorflow import where, gather_nd, unique
from keras.layers import multiply, add
from keras.layers import Lambda


def laplacian_loss(min_curvature=0.0, sampling=None):
    '''Draft of loss based on laplacian

    '''

    def laplace2D(x):
        '''Compute the 2D laplacian of an array

        Notes
        -----
        see https://www.tensorflow.org/tutorials/non-ml/pdes
        '''

        laplace_kernel = K.constant([[0.5, 1.0, 0.5],
                                     [1.0, -6., 1.0],
                                     [0.5, 1.0, 0.5]],
                                    shape=[3, 3, 1, 1])
        y = K.depthwise_conv2d(x, laplace_kernel, (1, 1), padding='same')

        return y

    def laplace3D(x, sampling):
        '''Compute the 3D laplacian of an array

        Notes
        -----
        depthwise_conv3d not available --> emulate with tf.scan
        https://github.com/tensorflow/tensorflow/issues/7278
        -----
        '''
        # TODO verify with separate test

        if sampling is None:
            sampling = (1.0, 1.0, 1.0)
        else:
            scaling = sampling[-1]
            sampling = tuple(x / scaling for x in sampling)

        w = tuple(1. / x**2 for x in sampling)
        cen = -2 * sum(w)

        laplace_kernel = K.constant([[[0.0, 0.0, 0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0, 0.0]],

                                     [[0.0, w[1], 0.0],
                                      [w[2], cen, w[2]],
                                      [0.0, w[1], 0.0]],

                                     [[0.0, 0.0, 0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0, 0.0]]],

                                    shape=[3, 3, 3, 1, 1])

        def flat_conv(_, elem):
            x = elem[0]
            kernel = elem[1]

            y = tf.nn.conv3d(x, kernel, [1, 1, 1, 1, 1], "SAME")
            return y

        kernel_scan = tf.transpose(laplace_kernel, [3, 0, 1, 2, 4])
        kernel_scan = tf.expand_dims(kernel_scan, axis=4)
        x_scan = tf.transpose(x, [4, 0, 1, 2, 3])
        x_scan = tf.expand_dims(x_scan, axis=-1)

        y = tf.scan(flat_conv, (x_scan, kernel_scan),
                    initializer=tf.zeros_like(x))

        y = tf.transpose(y, [1, 2, 3, 4, 0, 5])
        y = tf.squeeze(y, -1)

        return y

    def loss(y_true, y_pred):
        '''
        '''

        # extract pre-computed weight layer
        weights_fg = y_true[..., 1:]
        y_true = y_true[..., :1]

        # foreground > 1.0
        loss_fg = K.clip(1. - y_pred, 0., None)  # K.abs(1.-y_pred)
        loss_fg = K.sum(loss_fg * weights_fg)

        # lapl < min_curvature
        loss_fg_lapl = K.clip(lapl + min_curvature, 0., None)
        loss_fg_lapl = K.sum(loss_fg_lapl * weights_fg)

        return loss_fg + loss_fg_lapl

    return loss
