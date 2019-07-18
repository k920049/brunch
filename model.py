from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from model.sequence.Encoder import Encoder
from model.sequence.Attention import Attention

epoch=10

history_length = 10
vocab_size=1234
embedding_dim=300
units=256

BATCH_SIZE=64

embedding_mx = np.zeros(shape=(vocab_size, embedding_dim))

with tf.name_scope("data"):
    user = tf.placeholder(dtype=tf.int32,
                          shape=(None, history_length))
    item = tf.placeholder(dtype=tf.int32,
                          shape=(None, 1))
    label = tf.placeholder(dtype=tf.int32,
                           shape=(None, 1))

with tf.name_scope("model"):
    # Sequencial Model
    encoder = Encoder(vocab_size=vocab_size,
                      embedding_dim=embedding_dim,
                      enc_units=units,
                      batch_size=BATCH_SIZE,
                      embedding_mx=embedding_mx)
    sample_output, sample_hidden = encoder(user)
    # Attention layer
    attention_layer = Attention(units=10, history=history_length)
    attension_result, attention_weights = attention_layer(sample_hidden, sample_output)
    print("done")
    # user dense layer
    user = tf.keras.layers.Dense(units=128, activation="relu")(attension_result)
    user = tf.keras.layers.Dense(units=64, activation="relu")(user)
    # item dense layer
    item = encoder.embedding(item)
    item = tf.keras.backend.squeeze(item, axis=1)
    item = tf.keras.layers.Dense(units=128, activation="relu")(item)
    item = tf.keras.layers.Dense(units=64, activation="relu")(item)
    # dot product
    logit = tf.keras.layers.Dot(axes=1)([user, item])

with tf.name_scope("train"):
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(label, logit)

