import os
from multiprocessing import Pool

import fire
import tqdm
import json
import ast
import sqlite3

from google.cloud import bigquery

PROJECT = "gcp-tensorflow-222205"
REGION = "asia-east1"
dataset_id = "brunch"
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../gcp-tensorflow-222205-b11b68d4c01c.json"


class Handler(object):

    contents_file_number = 7

    def __init__(self, read_batch_size = None):
        self.read_batch_size = read_batch_size
        # self.conn, self.cursor = self._open()
        self.client = self._open()

    def _open(self):
        return bigquery.Client(project=PROJECT)

    def write_contents_to_bq(self):

        dataset_ref = self.client.dataset(dataset_id)
        job_config = bigquery.LoadJobConfig()
        job_config.autodetect = True
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        uri = "gs://dataproc-7e10897a-5391-4ea0-b815-f6e72cf284f7-asia-east1/data/contents/data_modified.json/part-00000-3873cd9e-dd6c-4e9f-a94c-6362a265d946-c000.json"

        try:
            load_job = self.client.load_table_from_uri(
                uri,
                dataset_ref.table("contents"),
                job_config=job_config
            )  # API request
        except Exception as e:
            print(e)
        finally:
            print(load_job.result())
            # load_job.result()  # Waits for table load to complete.
            print("Job finished.")

    def write_sentences_to_bq(self):

        dataset_ref = self.client.dataset(dataset_id)
        job_config = bigquery.LoadJobConfig()
        job_config.autodetect = True
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        uri = "gs://dataproc-7e10897a-5391-4ea0-b815-f6e72cf284f7-asia-east1/data/contents/sentences/part-00000-a04d02d9-794e-4546-a9af-38dbb086452f-c000.json"

        try:
            load_job = self.client.load_table_from_uri(
                uri,
                dataset_ref.table("sentences"),
                job_config=job_config
            )  # API request
        except Exception as e:
            print(e)
        finally:
            print(load_job.result())
            # load_job.result()  # Waits for table load to complete.
            print("Job finished.")


if __name__ == "__main__":
    fire.Fire(Handler)
