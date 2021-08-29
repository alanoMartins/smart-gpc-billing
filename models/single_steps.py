import tensorflow as tf
from tensorflow import keras


class Baseline(keras.Model):

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class MLP(keras.Model):

    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.output_model = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_model(x)


class LSTM(keras.Model):

    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.output_model = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return self.output_model(x)

