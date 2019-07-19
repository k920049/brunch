import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf

vocab_size=642190
sample_ratio = 5

def generate_batch(frame):
    frame = frame.sort_values(by="timestamp")
    values = frame["pos"].values
    # user batch
    user = [values] * (sample_ratio * len(values))
    user = np.array(user)
    # item batch
    sampled = np.random.randint(low=0, high=vocab_size, size=(sample_ratio - 1) * len(values))
    item = np.concatenate([values, sampled])
    item = np.reshape(item, (-1, 1))
    # label batch
    pos = [1] * len(values)
    neg = [0] * ((sample_ratio - 1) * len(values))
    label = np.concatenate([pos, neg])
    label = np.reshape(label, (-1, 1))
    return user, item, label


def generate_dataset(path_to_history="./data/history_stripped.parquet",
                     batch_size=256):
    print("Reading a pandas dataframe")
    cores = multiprocessing.cpu_count()
    df = pd.read_parquet(path_to_history)

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
        dataset = tf.data.Dataset.zip((data, label)).shuffle(1000).batch(batch_size)
        return dataset
