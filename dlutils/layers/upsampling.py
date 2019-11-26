'''Bilinear upsampling layer from deeplabv3 implementation with Keras:

https://github.com/qqgeogor/keras-segmentation-networks/blob/master/deeplabv3.py

'''
from tensorflow.keras import backend as K
from tensorflow.keras.engine import InputSpec
from tensorflow.keras.engine.topology import Layer
import tensorflow as tf


class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self,
                 target_shape=None,
                 factor=None,
                 data_format=None,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0], self.target_size[1],
                    input_shape[3])
        else:
            return (input_shape[0], input_shape[1], self.target_size[0],
                    self.target_size[1])

    def call(self, inputs):
        return tf.image.resize_images(inputs, size=self.target_size[:2])

    def get_config(self):
        config = {
            'target_shape': self.target_shape,
            'data_format': self.data_format
        }
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
