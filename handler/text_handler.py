import json
import multiprocessing
import numpy as np
import fire

from multiprocessing import Pool

class TextHandler(object):

    def __init__(self, batch_size, window_size):

        self.batch_size = batch_size
        self.window_size = window_size
        self.sentences = self._read_sentences_from_json("./data/test.json")

    def retrieve_batch_from_bq(self, idx):
        pass

    def generate_batch_data(self):

        batch_data = []
        label_data = []

        while len(batch_data) < self.batch_size:
            # select random sentence to start
            rand_sentence_ix = int(np.random.choice(len(self.sentences), size=1))
            rand_sentence = self.sentences[rand_sentence_ix]
            # Generate consecutive windows to look at
            window_sequences = [rand_sentence[max((ix - self.window_size), 0):(ix + self.window_size + 1)] for ix, x in enumerate(rand_sentence)]
            # Denote which element of each window is the center word of interest
            label_indices = [ix if ix < self.window_size else self.window_size for ix, x in enumerate(window_sequences)]
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * self.window_size]

            if len(batch_and_labels) == 0:
                continue

            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # extract batch and labels
            batch_data.extend(batch[:self.batch_size])
            label_data.extend(labels[:self.batch_size])
        # Trim batch and label at the end
        batch_data = batch_data[:self.batch_size]
        label_data = label_data[:self.batch_size]

        # Convert to numpy array
        batch_data = np.array(batch_data)
        label_data = np.transpose(np.array([label_data]))

        return batch_data, label_data

    @staticmethod
    def _read_sentences_from_json(file_name):
        documents = None
        with open(file_name) as fp:
            lines = [line for line in fp]

            num_cores = multiprocessing.cpu_count()
            with Pool(num_cores) as p:
                documents = p.map(json.loads, lines)

        return [doc["sentences"] for doc in documents]


if __name__ == "__main__":
    fire.Fire(TextHandler)

