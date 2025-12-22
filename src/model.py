import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.kerase.models import Model 
from tensorflow.keras.losses import Loss


class CustomRelu(Layer):

    def __init__(self, name="CustomRelu"):
        super().__init__(name=name)

    def call(self, inputs):
        return tf.maximum(inputs, 0.0)
    

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


