import tensorflow as tf


class ScaleAndClipLayer(tf.keras.layers.Layer):
    '''statically rescales the input tensors values from [lower, upper] to [0, 1.]
    and clips everything outside [0,1].

    '''

    def __init__(self, lower, upper, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
        if not upper > lower:
            raise ValueError(
                'lower bound must be smaller than upper bound: {} !< {}'.
                format(lower, upper))

        self.lower = lower
        self.delta = upper - lower

    def call(self, x):
        '''
        '''
        return tf.clip_by_value((x - self.lower) / self.delta, 0., 1.)

    def compute_output_shape(self, input_shape):
        '''
        '''
        return input_shape

    def get_config(self):
        '''
        '''
        config = super().get_config()
        config['lower'] = self.lower
        config['upper'] = self.lower + self.delta
        return config
