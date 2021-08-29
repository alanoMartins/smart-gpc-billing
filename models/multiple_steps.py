import tensorflow as tf
from tensorflow import keras


class Baseline(tf.keras.Model):

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class MLP(keras.models):

    def __init__(self, output_size, feature_size):
        super(MLP, self).__init__()
        self.output_size = output_size
        self.feature_size = feature_size
        self.lambda1 = tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.output_size*self.feature_size, kernel_initializer=tf.initializers.zeros()),
        self.output = tf.keras.layers.Reshape([self.output_size, self.feature_size])

    def call(self, inputs):
        x = self.lambda1(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output(x)


class LSTM(tf.keras.models):

    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.dense1 = tf.keras.layers.Dense(self.output_size * self.feature_size,
                                            kernel_initializer=tf.initializers.zeros()),
        self.output = tf.keras.layers.Reshape([self.output_size, self.feature_size])

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dense1(x)
        return self.output(x)

