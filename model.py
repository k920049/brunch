from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os

from model.sequence.Encoder import Encoder
from model.sequence.Attention import Attention
from handler.dataset import generate_dataset, generate_embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from fire import Fire

class Recommender(object):

    def __init__(self, epoch=10,
                 batch_size = 512,
                 evaluation_ratio = 0.1,
                 encoder_units = 256,
                 history_length=15):

        self.epoch = epoch
        self.batch_size = batch_size
        self.evaluation_ratio = evaluation_ratio
        self.encoder_units = encoder_units
        self.history_length = history_length

        self._build_model()

    def _build_model(self):

        self.embedding_mx = generate_embedding(path_to_dictionary="./data/dictionary.json",
                                               path_to_embedding="./data/embedding-2000.npy")

        self.vocab_size = self.embedding_mx.shape[0]
        self.embedding_dim = self.embedding_mx.shape[1]

        with tf.name_scope("data"):
            user_input = tf.keras.Input(shape=(self.history_length,))
            item_input = tf.keras.Input(shape=(1,))

        with tf.name_scope("model"):
            # Sequencial Model
            encoder = Encoder(vocab_size=self.vocab_size,
                              embedding_dim=self.embedding_dim,
                              enc_units=self.encoder_units,
                              batch_size=self.batch_size,
                              embedding_mx=self.embedding_mx)
            sample_output, sample_hidden = encoder(user_input)
            # Attention layer
            attention_layer = Attention(units=64, history=self.history_length)
            attension_result, attention_weights = attention_layer(sample_hidden, sample_output)
            # user dense layer
            user = tf.keras.layers.Dense(units=256, activation="relu")(attension_result)
            user = tf.keras.layers.Dense(units=128, activation="relu")(user)
            user = tf.keras.layers.Dense(units=64, activation="relu")(user)
            # item dense layer
            item = encoder.embedding(item_input)
            item = tf.keras.backend.squeeze(item, axis=1)
            item = tf.keras.layers.Dense(units=256, activation="relu")(item)
            item = tf.keras.layers.Dense(units=128, activation="relu")(item)
            item = tf.keras.layers.Dense(units=64, activation="relu")(item)
            # dot product
            logit = tf.keras.layers.Dot(axes=1)([user, item])
            pred = tf.keras.layers.Activation(activation='sigmoid')(logit)

        with tf.name_scope("train"):
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=pred)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        data, label = generate_dataset("./data/history_stripped.parquet")
        filepath = "./data/checkpoints-original/model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     save_weights_only=True,
                                     save_best_only=True,
                                     mode="max",
                                     verbose=1)
        self.model.fit(x = data,
                       y = label,
                       batch_size=self.batch_size,
                       validation_split=self.evaluation_ratio,
                       shuffle=True,
                       epochs=self.epoch,
                       callbacks=[checkpoint])

    def test(self):
        pass


if __name__ == "__main__":
    Fire(Recommender)



