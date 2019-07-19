from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from model.sequence.Encoder import Encoder
from model.sequence.Attention import Attention
from handler.dataset import generate_dataset
from keras.callbacks import ModelCheckpoint

epoch=10

history_length = 15
vocab_size=1234
embedding_dim=300
units=256

BATCH_SIZE=64

embedding_mx = np.zeros(shape=(vocab_size, embedding_dim))

with tf.name_scope("data"):
    user_input = tf.keras.Input(shape=(history_length,))
    item_input = tf.keras.Input(shape=(1,))

with tf.name_scope("model"):
    # Sequencial Model
    encoder = Encoder(vocab_size=vocab_size,
                      embedding_dim=embedding_dim,
                      enc_units=units,
                      batch_size=BATCH_SIZE,
                      embedding_mx=embedding_mx)
    sample_output, sample_hidden = encoder(user_input)
    # Attention layer
    attention_layer = Attention(units=10, history=history_length)
    attension_result, attention_weights = attention_layer(sample_hidden, sample_output)
    print("done")
    # user dense layer
    user = tf.keras.layers.Dense(units=128, activation="relu")(attension_result)
    user = tf.keras.layers.Dense(units=64, activation="relu")(user)
    # item dense layer
    item = encoder.embedding(item_input)
    item = tf.keras.backend.squeeze(item, axis=1)
    item = tf.keras.layers.Dense(units=128, activation="relu")(item)
    item = tf.keras.layers.Dense(units=64, activation="relu")(item)
    # dot product
    logit = tf.keras.layers.Dot(axes=1)([user, item])
    pred = tf.keras.layers.Activation(activation='sigmoid')(logit)

with tf.name_scope("train"):
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=pred)
    filepath = "../data/checkpoints/model-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  batch_size=10,
                  validation_split=0.1,
                  callbacks=[checkpoint],
                  verbose=0)

dataset = generate_dataset()
model.fit(dataset, epochs=epoch)


