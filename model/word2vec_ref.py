# Word2Vec: CBOW Model (Continuous Bag of Words)
# ---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit the CBOW model of
#  the Word2Vec Algorithm
import tensorflow as tf
import numpy as np
import os

from handler.text_handler import TextHandler
from tensorflow.python.framework import ops

ops.reset_default_graph()
# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 1796381
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 2 * 3  # How many words to consider left and right.
# Add checkpoints to training
save_embeddings_every = 5000
print_loss_every = 100

print("Loading data")
with tf.name_scope("data") as scope:
    handler = TextHandler(batch_size=batch_size,
                          window_size=window_size)
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])

print('Creating Model')
with tf.name_scope("model") as scope:
    # Define Embeddings:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Lookup the word embedding
    embed = tf.zeros([batch_size, embedding_size])
    for element in range(2 * window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

with tf.name_scope("train") as scope:
    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=y_target,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size,
                                         partition_strategy="div"))

    # Create optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=model_learning_rate).minimize(loss)

with tf.name_scope("miscellaneous") as scope:
    # Create model saving operation
    saver = tf.train.Saver({"embeddings": embeddings})
    # Add variable initializer.
    init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    print('Starting Training')
    sess.run(init)
    loss_vec = []
    loss_x_vec = []

    for i in range(generations):
        batch_inputs, batch_labels = handler.generate_batch_data()
        feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
        # Run the train step
        sess.run(optimizer, feed_dict=feed_dict)

        # Return the loss
        if (i + 1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i + 1)
            print('Loss at step {} : {}'.format(i + 1, loss_val))

        # Save dictionary + embeddings
        if (i + 1) % save_embeddings_every == 0:
            # Save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(), "../data/cbow_movie_embeddings.ckpt")
            save_path = saver.save(sess, model_checkpoint_path)
            print('Model saved in file: {}'.format(save_path))
