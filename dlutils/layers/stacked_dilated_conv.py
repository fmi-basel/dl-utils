import tensorflow as tf

from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import LeakyReLU


# TODO instantiate conv ops once (and tf keras normalize dilation tuples) once in init and only reassign in call
# TODO 2D/3D + add to get_nd_layer
# TODO add to list for yaml import
# TODO unittest
class StackedDilatedConv(Conv):
    '''Applies the same filters with different dilation rates,
    concatenates the outputs and reduce it back to "filters" channels.
    '''
    def __init__(self, *args, dilation_rates=(1, 2, 4, 8, 16), **kwargs):
        '''
        '''
        super().__init__(*args, **kwargs)

        self.dilation_rates = dilation_rates

        self.reduce_ch_conv = Conv(self.rank, self.filters, 1)

    def call(self, inputs):
        '''
        '''

        outs = []
        for dilation in self.dilation_rates:
            # replace conv op with current dilation; the same self.weights are used
            self.dilation_rate = conv_utils.normalize_tuple(
                dilation, self.rank, 'dilation_rate')
            self._convolution_op = nn_ops.Convolution(
                inputs.get_shape(),
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(
                    self.data_format, self.rank + 2))

            outs.append(super().call(inputs))

        out = tf.concat(outs, axis=-1)
        out = LeakyReLU()(out)
        out = self.reduce_ch_conv(out)

        return out

    def get_config(self):
        '''
        '''

        return super().get_config()
        config['dilation_rates'] = self.dilation_rates
        return config
