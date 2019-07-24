import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import math

from tqdm import tqdm

vocab_size=642190
sample_ratio = 4
ratio = 0.1
rec_list = np.load("./data/test_45000.npy")

# def generate_batch(frame):
#     frame = frame.sort_values(by="timestamp")
#     values = frame["pos"].values
#     # user batch
#     user = [values] * (sample_ratio * len(values))
#     user = np.array(user)
#     # item batch
#     sampled = np.random.randint(low=0, high=vocab_size, size=(sample_ratio - 1) * len(values))
#     item = np.concatenate([values, sampled])
#     item = np.reshape(item, (-1, 1))
#     # label batch
#     pos = [1] * len(values)
#     neg = [0] * ((sample_ratio - 1) * len(values))
#     label = np.concatenate([pos, neg])
#     label = np.reshape(label, (-1, 1))
#     return user, item, label

def generate_batch(frame):

    user = list(frame["train"].values)
    item = frame["eval"].values[0]
    item = np.unique(item)
    pos = [1] * len(item)
    item_length = len(item)

    sampled = np.random.randint(low=0, high=vocab_size, size=(sample_ratio * item_length))
    #sampled = np.random.choice(rec_list, size=(sample_ratio * item_length))
    item = np.append(item, sampled)
    item = np.reshape(item, (-1, 1))

    neg = [0] * (sample_ratio * item_length)
    label = np.concatenate([pos, neg])
    label = np.reshape(label, (-1, 1))

    user = user * ((sample_ratio + 1) * item_length)
    user = np.array(user)

    assert len(user) == len(item)

    return user, item, label


def generate_train_dataset(path_to_history="./data/train.parquet"):
    print("Reading a pandas dataframe")
    cores = multiprocessing.cpu_count()
    df = pd.read_parquet(path_to_history)

    df = df[df["length"] > 5]
    df_group = df.groupby("id")
    df_group = [frame for name, frame in tqdm(df_group)]

    print(rec_list.shape)

    with multiprocessing.Pool(cores) as p:
        print("Generating a dataset of this epoch")
        dataset = list(tqdm(p.imap(generate_batch, df_group), total=len(df_group)))
        user = np.vstack([elem[0] for elem in dataset]).astype(np.float32)
        item = np.vstack([elem[1] for elem in dataset]).astype(np.float32)
        label = np.vstack([elem[2] for elem in dataset]).astype(np.float32)

        return [user, item], label

def generate_evaluation_dataset(df, id, numbers):
    sample = df.loc[id]
    train = np.array([sample["train"]] * numbers)

    return train

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
    embedding = np.concatenate((embedding, date_array), axis=1)
    zero = np.zeros((1, 301))
    return np.concatenate((embedding, zero), axis=0)
