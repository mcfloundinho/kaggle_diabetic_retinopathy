import numpy as np

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils


class MaxFeaturePooling(Layer):
    '''Layer for pooling over feature axis.'''

    def __init__(self, pool_size=2, data_format=None, **kwargs):
        super(MaxFeaturePooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.axis = 1 if self.data_format == 'channels_first' else 3
        self.input_spec = InputSpec(ndim=4)

    def assert_input_compatibility(self, inputs):
        super(MaxFeaturePooling, self).assert_shape_compatiblity(self, inputs)
        input = inputs[0]
        if input.shape[self.axis] % self.pool_size:
            raise ValueError('Input is incompatible with layer ' +
                             self.name + ': expected axis ' +
                             str(self.axis) + ' of input shape to be multiple '
                             'of ' + str(self.pool_size) +
                             ' but got shape ' + str(input.shape))

    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + \
            (input_shape[self.axis] / self.pool_size,) + \
            input_shape[self.axis + 1, :]

    def call(self, inputs):
        input = inputs[0]
        input_shape = tuple(input.shape)
        pool_shape = input_shape[:self.axis] + \
            (input_shape[self.axis] / self.pool_size, self.pool_size) + \
            input_shape[self.axis + 1, :]
        pool_axis = self.axis + 1
        input_reshaped = K.reshape(input, pool_shape)
        return K.max(input_reshaped, axis=pool_axis)


class MergeBatch(Layer):
    '''Layer for merging adjacent samples in the batch.'''

    def __init__(self, coef=2, **kwargs):
        super(MergeBatch, self).__init__(**kwargs)
        self.coef = coef
        self.input_spec = InputSpec(min_ndim=2)

    def assert_input_compatibility(self, inputs):
        super(MergeBatch, self).assert_shape_compatiblity(self, inputs)
        input = inputs[0]
        if input.shape[0] % self.coef:
            raise ValueError('Input is incompatible with layer ' +
                             self.name + ': expected axis 0 of input shape'
                             'to be multiple of ' + str(self.coef) +
                             ' but got shape ' + str(input.shape))

    def compute_output_shape(self, input_shape):
        batches = input_shape[0] / self.coef
        return (batches, np.product(input_shape) / batches)

    def call(self, inputs):
        input = inputs[0]
        input_shape = tuple(input.shape)
        return K.reshape(input, (input_shape[0] / self.coef, -1))


class SplitBatch(Layer):
    '''Layer for splitting a sample into multiple samples in the batch.'''

    def __init__(self, coef=2, **kwargs):
        super(SplitBatch, self).__init__(**kwargs)
        self.coef = coef
        self.input_spec = InputSpec(min_ndim=2)

    def assert_input_compatibility(self, inputs):
        super(SplitBatch, self).assert_shape_compatiblity(self, inputs)
        input = inputs[0]
        input_shape = tuple(input.shape)
        if np.product(input_shape[1:]) % self.coef:
            raise ValueError('Input is incompatible with layer ' +
                             self.name + ': expected product of axes [1:] of '
                             'input shape to be multiple of ' + str(self.coef) +
                             ' but got shape ' + str(input.shape))

    def compute_output_shape(self, input_shape):
        batches = input_shape[0] * self.coef
        return (batches, np.product(input_shape) / batches)

    def call(self, inputs):
        input = inputs[0]
        input_shape = tuple(input.shape)
        return K.reshape(input, (input_shape[0] * self.coef, -1))
