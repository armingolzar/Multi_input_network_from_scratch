import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.kerase.models import Model 
from tensorflow.keras.losses import Loss


class CUSTOMRELU(Layer):

    def __init__(self, name="CUSTOMRELU"):
        super().__init__(name=name)

    def call(self, inputs):
        return tf.maximum(inputs, 0.0)
    
    


