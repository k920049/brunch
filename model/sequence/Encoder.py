from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, embedding_mx):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_mx],
                                                   trainable=False)
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.CuDNNLSTM(self.enc_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        )
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, x):
        x = self.embedding(x)
        x = self.bi_lstm(x)
        output = x[0]
        state = self.concat(x[1:])
        return output, state


