import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.kerase.models import Model 
from tensorflow.keras.losses import Loss


class CustomRelu(Layer):

    def __init__(self, name="CustomRelu"):
        super().__init__(name=name)

    def call(self, inputs):
        return tf.maximum(inputs, 0.0)
    
    def get_config(self):
        return super().get_config()
    

class CustomDense(Layer):

    def __init__(self, units, name="CustomDense"):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):

        in_features = input_shape[-1]

        self.W = self.add_weight(name="WEIGHTS", shape=(in_features, self.units), initializer="he_normal", trainable=True)

        self.B = self.add_weight(name="BIAS", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):

        output = tf.matmul(inputs, self.W) + self.B
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units" : self.units})
        return config


class CustomConv2D(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding="SAME", name="CustomConv2D"):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()

    def build(self, input_shape):

        in_channel = input_shape[-1]
        kh, kw = self.kernel_size

        self.kernel = self.add_weight(name="kernel", shape=(kh, kw, in_channel, self.filters), initializer="he_normal", trainable=True)

        self.B = self.add_weight(name="BIAS", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):

        conv_output = tf.nn.conv2d(inputs, self.kernel, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding)
        output = tf.nn.bias_add(conv_output, self.B)

        return output
    
    def get_config(self):
        config = super().get_config()
        config = config.update({"filters":self.filters, "kernel_size":self.kernel_size, "strides":self.strides, "padding":self.padding})
        return config
    
class CustomFlatten(Layer):

    def __init__(self, name="CustomFlatten"):
        super().__init__(name=name)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, (batch_size, -1))
    
    def get_config(self):
        return super().get_config()
    
class CustomConcat(Layer):

    def __init__(self, axis=-1, name="CustomConcat"):
        super().__init__(name=name)
        self.axis = axis

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis":self.axis})
        return config
    
