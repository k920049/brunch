from fire import Fire
import pyarrow.parquet as pq
import pandas as pd

dataset = "../data/documents"

class Doc2Vec(object):

    def __init__(self):
        pass

    def read(self):
        table = pq.ParquetDataset(dataset).read()
        df = table.to_pandas()
        df.head()

if __name__ == "__main__":
    Fire(Doc2Vec)