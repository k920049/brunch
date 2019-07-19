import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import math

from tqdm import tqdm

vocab_size=642190
sample_ratio = 2
ratio = 0.1

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


def generate_dataset(path_to_history="./data/history_stripped.parquet"):
    print("Reading a pandas dataframe")
    cores = multiprocessing.cpu_count()
    df = pd.read_parquet(path_to_history)

    df_group = df.groupby("id")
    df_group = [frame for name, frame in tqdm(df_group)]

    with multiprocessing.Pool(cores) as p:
        print("Generating a dataset of this epoch")
        dataset = list(tqdm(p.imap(generate_batch, df_group), total=len(df_group)))
        user = np.vstack([elem[0] for elem in dataset]).astype(np.float32)
        item = np.vstack([elem[1] for elem in dataset]).astype(np.float32)
        label = np.vstack([elem[2] for elem in dataset]).astype(np.float32)

        train_length = math.floor(user.shape[0])
        user_train = user[0:train_length]
        item_train = item[0:train_length]
        data_train = [user_train, item_train]
        label_train = label[0:train_length]

        user_eval = user[train_length:]
        item_eval = item[train_length:]
        data_eval = [user_eval, item_eval]
        label_eval = label[train_length:]

        #train_data = tf.data.Dataset.from_tensor_slices((user_train, item_train))
        #train_label = tf.data.Dataset.from_tensor_slices(label_train)
        #train_dataset = tf.data.Dataset.zip((train_data, train_label)).shuffle(1000)

        #eval_data = tf.data.Dataset.from_tensor_slices((user_eval, item_eval))
        #eval_label = tf.data.Dataset.from_tensor_slices(label_eval)
        #eval_dataset = tf.data.Dataset.zip((eval_data, eval_label)).shuffle(1000)
        return data_train, label_train, data_eval, label_eval
