from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class Attention(tf.keras.Model):
    def __init__(self, units, history):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.history = history

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(
            tf.nn.tanh(
                self.W1(values) + self.W2(hidden_with_time_axis)
            )
        )
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.squeeze(tf.gather(context_vector, [self.history - 1], axis=1), axis=1)
        return context_vector, attention_weights