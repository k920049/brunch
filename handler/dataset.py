import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import math

from tqdm import tqdm

vocab_size=642190
sample_ratio = 3
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

        return [user, item], label

def generate_embedding(path_to_dictionary="./data/positional_dictionary.json",
                       path_to_embedding="./data/embedding.npy"):
    print("Loading doc2vec embeddings")
    embedding = np.load(path_to_embedding)
    df = pd.read_json(path_to_dictionary)

    df = df.sort_values("pos")
    date = df["date"].values
    date_array = np.array(date)
    date_array = (date_array - np.min(date_array))
    date_array = date_array / np.max(date_array)
    date_array = np.expand_dims(date_array, axis=1)
    return np.concatenate((embedding, date_array), axis=1)
