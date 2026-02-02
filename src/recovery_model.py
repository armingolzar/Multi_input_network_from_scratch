import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer 
from tensorflow.keras.losses import Loss

class CustomRelu(Layer):

    def __init__(self, name="CustomRelu", delta=0):
        super().__init__(name=name)
        self.delta = delta

    def call(self, inputs):
        return tf.maximum(inputs, self.delta)
    
    def get_config(self):
        config = super().get_config()
        config.update({"delta" : self.delta})
        return config

class CustomDense(Layer):

    def __init__(self, units, name="CustomDense"):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):

        last_dim = input_shape[-1]
        self.w = tf.add_weight(name="weights", shape=(last_dim, self.units), initializer="he_normal", trainable=True)
        self.b = tf.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):

        output = tf.matmul(inputs, self.w) + self.b
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units" : self.units})
        return config
    
class CustomConv2D(Layer):

    def __init__(self, filters, kernel_size, stride=(1, 1), padding="VALID", name="CustomConv2D"):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding.upper()

    def build(self, input_shape):

        last_dim = input_shape[-1]
        kh, kw = self.kernel_size
        self.k = tf.add_weight(name="kernel", shape=(kh, kw, last_dim, self.filters), initializer="he_norm", trainable=True)
        self.b = tf.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):

        conv_output = tf.nn.conv2d(inputs, self.k, strides=[1, self.stride[0], self.stride[1], 1], padding=self.padding)
        result = tf.nn.bias_add(conv_output, self.b)
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters" : self.filters, "kernel_size" : self.kernel_size, "stride" : self.stride, "padding" : self.padding})
        return config


class CustomMaxpooling2D(Layer):

    def __init__(self, pool_size=(2, 2), stride=None, padding="VALID", name="CustomMaxpooling2D"):
        super().__init__(name=name)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding.upper()

    def call(self, inputs):
        return tf.nn.max_pool2d(inputs, ksize=[1, self.pool_size[0], self.pool_size[1], 1], strides=[1, self.stride[0], self.stride[1], 1], padding=self.padding)
    
    def get_config(self):
        config = super().get_config()
        config.update({"pool_size" : self.pool_size, "stride" : self.stride, "padding" : self.padding})
        return config
    
    
class CustomFlatten(Layer):
    
    def __init__(self, name="CustomFlatten", feature_dim=None):
        super().__init__(name=name)
        self.feature_dim = feature_dim

    def build(self, input_shape):

        feature_dim = 1
        for dim in input_shape[1:]:
            if dim is None:
                raise ValueError("CustomFlatten requires static spatial dimensions")
            feature_dim *= dim

        self.feature_dim = feature_dim

    def call(self, inputs):
        return tf.reshape(inputs, (-1, self.feature_dim))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.feature_dim)
    
    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim" : self.feature_dim})
        return config

class CustomConcat(Layer):

    def __init__(self, axis=-1, name="CustomConcat"):
        super().__init__(name=name)
        self.axis = axis

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis" : self.axis})
        return config
    
class ImageEncoder(Layer):
    
    def __init__(self, name="ImageEncoder"):
        super().__init__(name=name)

        self.conv1 = CustomConv2D(32, (3, 3))
        self.relu1 = CustomRelu()
        self.conv2 = CustomConv2D(32, (3, 3))
        self.relu2 = CustomRelu()
        self.maxpool1 = CustomMaxpooling2D()

        self.conv3 = CustomConv2D(64, (3, 3))
        self.relu3 = CustomRelu()
        self.conv4 = CustomConv2D(64, (3, 3))
        self.relu4 = CustomRelu()
        self.maxpool2 = CustomMaxpooling2D()

        self.flatten = CustomFlatten()
        self.dense = CustomDense(128)

    def call(self, inputs):

        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.conv2(x))
        x = self.maxpool1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


class TabularEncoder(Layer):
    def __init__(self, name="TabularEncoder"):
        super().__init__(name=name)

        self.dense1 = CustomDense(16)
        self.relu1 = CustomRelu()
        self.dense2 = CustomDense(32)
        self.relu2 = CustomRelu()
        self.dense3 = CustomDense(64)

    def call(self, inputs):

        x = self.relu1(self.dense1(inputs))
        x = self.relu2(self.dense2(x))
        x = self.dense3(x)

        return x

    
class HousePriceModel(Model):

    def __init__(self, name="HousePriceModel"):
        super().__init__(name=name)

        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder()

        self.concat = CustomConcat()
        self.dense1 = CustomDense(64)
        self.relu1 = CustomRelu()
        self.dense2 = CustomDense(32)
        self.relu2 = CustomRelu()
        self.dense3 = CustomDense(1)

    def call(self, inputs):

        image, tabular = inputs
        image_featuremap = self.image_encoder(image)
        tabular_featuremap = self.tabular_encoder(tabular)

        featuremap = self.concat([image_featuremap, tabular_featuremap])
        x = self.relu1(self.dense1(featuremap))
        x = self.relu2(self.dense2(x))
        x = self.dense3(x)

        return x







    

