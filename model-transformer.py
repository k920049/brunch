from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os

# from model.sequence.Encoder import Encoder
# from model.sequence.Attention import Attention
from model.transformer.Encoder import Encoder
from model.transformer.Schedule import CustomSchedule
from handler.dataset import generate_dataset, generate_embedding
from tensorflow.keras.callbacks import ModelCheckpoint

print(os.getcwd())

embedding_mx = generate_embedding(path_to_dictionary="./data/dictionary.json",
                                  path_to_embedding="./data/embedding-2000.npy")
epoch = 10
history_length = 15
vocab_size = embedding_mx.shape[0]
embedding_dim = embedding_mx.shape[1] - 1
units = 256

BATCH_SIZE=512
evaluation_ratio = 0.9

with tf.name_scope("data"):
    user_input = tf.keras.Input(shape=(history_length,))
    item_input = tf.keras.Input(shape=(1,))

with tf.name_scope("model"):
    # Sequencial Model
    # encoder = Encoder(vocab_size=vocab_size,
    #                   embedding_dim=embedding_dim,
    #                   enc_units=units,
    #                   batch_size=BATCH_SIZE,
    #                   embedding_mx=embedding_mx)
    encoder = Encoder(num_layers=4,
                      d_model=embedding_dim,
                      num_heads=5,
                      dff=512,
                      input_vocab_size=vocab_size,
                      embedding_mx=embedding_mx)
    output = encoder(user_input, training=True, mask=None)
    output = tf.squeeze(tf.gather(output, [history_length - 1], axis = 1), axis = 1)
    # Attention layer
    # attention_layer = Attention(units=64, history=history_length)
    # attension_result, attention_weights = attention_layer(sample_hidden, sample_output)
    # user dense layer
    user = tf.keras.layers.Dense(units=256, activation="relu")(output)
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
    filepath = "./data/checkpoints-original/model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode="max",
                                 verbose=1)
    learning_rate = CustomSchedule(embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1 = 0.9,
                                         beta_2 = 0.98,
                                         epsilon = 1e-9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

data, label = generate_dataset("./data/history_stripped.parquet")
model.fit(x = data,
          y = label,
          batch_size=BATCH_SIZE,
          validation_split=0.1,
          shuffle=True,
          epochs=epoch,
          callbacks=[checkpoint])


