import os
from tqdm import tqdm

from fire import Fire
from datetime import datetime
from google.cloud import bigquery
import matplotlib.pyplot as plt


PROJECT = "gcp-tensorflow-222205"
REGION = "asia-east1"
dataset_id = "brunch"
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../gcp-tensorflow-222205-2db292399699.json"


class HistoryHandler(object):

    def __init__(self):
        self.client = self.open()
        self.dataset = self.client.dataset('brunch')
        self.table = self.client.get_table(self.dataset.table('history'))

    def open(self):
        return bigquery.Client()

    def read(self):

        file_path = "/Users/jeasungpark/Repository/brunch/data/read"
        timestamp = os.listdir(file_path)

        for elem in tqdm(timestamp):
            tokens = elem.split("_")
            try:
                time_from = datetime.strptime(tokens[0], "%Y%m%d%H")
                time_to = datetime.strptime(tokens[1], "%Y%m%d%H")
            except Exception as e:
                print(e)
            else:
                with open(os.path.join(file_path, elem)) as fp:
                    history = [line for line in fp]
                    to_store = []
                    for each_log in history:
                        tokens = each_log.split(" ")
                        id = tokens[0]
                        visited = tokens[1:-1]
                        visited = list(set(visited))
                        visited = [(id, time_from, vis) for vis in visited]
                        to_store.extend(visited)

                    try:
                        job = self.client.insert_rows(self.table, to_store)
                    except Exception as e:
                        print(e)
                    else:
                        assert job == []

    def write_history(self):

        query_string = """
        select
          id, timestamp, document
        from
          `gcp-tensorflow-222205.brunch.history`
        """
        df = self.client.query(query_string).to_dataframe()
        df.to_parquet("../data/history.parquet")

    def plot_top_n(self, n = 10, bins = 10):
        """
        Plot a histogram of popluar news
        :param n: Top n news article to show
        :return: Nothing
        """
        query_string = """
        select
            document, count(id) as count
        from
            `gcp-tensorflow-222205.brunch.history`
        group by
            document
        order by
            count desc
        """

        if n != -1:
            query_string = query_string + """
            limit {}
            """.format(n)

        df = self.client.query(query_string).to_dataframe()
        hist = df.plot.line(x="document", y="count")
        plt.show()



if __name__ == "__main__":
    Fire(HistoryHandler)
