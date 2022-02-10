"""Utilities for creating VQC models"""

import tensorflow as tf
from tensorflow import keras

class Scale(keras.layers.Layer):

    def __init__(self, name=None):
        super(Scale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return self.factor * inputs

class SingleScale(keras.layers.Layer):

    def __init__(self, name=None):
        super(SingleScale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(1),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return self.factor * inputs


class ConstantScale(keras.layers.Layer):

    def __init__(self, factor, name=None):
        super(ConstantScale, self).__init__(name=name)
        self.factor = factor

    def call(self, inputs):
        return self.factor * inputs

class ExpScale(keras.layers.Layer):

    def __init__(self, name=None):
        super(ExpScale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return tf.exp(self.factor) * inputs
