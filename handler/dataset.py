import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf

def generate_dataset(path_to_weight="../data/embedding.npz.npy",
                      path_to_history="../data/history_stripped.parquet"):

    cores = multiprocessing.cpu_count()
    doc_vec = np.load(path_to_weight)
    vocab_size = doc_vec.shape[0]
    df = pd.read_parquet(path_to_history)

    def generate_batch(frame):
        frame = frame.sort_values(by="timestamp")
        values = frame["pos"].values
        # user batch
        user = [values] * (2 * len(values))
        user = np.array(user)
        # item batch
        sampled = np.random.randint(low=0, high=vocab_size, size=15)
        item = np.concatenate([values, sampled])
        item = np.reshape(item, (-1, 1))
        # label batch
        pos = [1] * 15
        neg = [0] * 15
        label = np.concatenate([pos, neg])
        label = np.reshape(label, (-1, 1))
        return user, item, label

    df_group = df.groupby("id")
    df_group = [frame for name, frame in df_group]

    with multiprocessing.Pool(cores) as p:
        print("Generating a dataset of this epoch")
        dataset = p.map(generate_batch, df_group)
        user = np.vstack([elem[0] for elem in dataset])
        item = np.vstack([elem[1] for elem in dataset])
        label = np.vstack([elem[2] for elem in dataset])

        data = tf.data.Dataset.from_tensor_slices((user, item))
        label = tf.data.Dataset.from_tensor_slices(label)
        dataset = tf.data.Dataset.zip((data, label))
        return dataset
