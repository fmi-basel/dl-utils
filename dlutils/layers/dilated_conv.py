from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers.convolutional import _Conv as KerasConvBase

from tensorflow.keras import backend as K


class DilatedConv2D(KerasConvBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'rank' in kwargs.keys():
            kwargs.pop('rank')
        super(DilatedConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        '''
            '''
        output = tf.nn.atrous_conv2d(
            inputs,
            self.kernel,
            #strides=(1, ) + self.strides + (1, ),
            padding=self.padding.upper(),
            rate=self.dilation_rate)

        if self.use_bias:
            output = K.bias_add(
                output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)
        return output

    def get_config(self):
        '''
        '''
        config = super(DilatedConv2D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config
