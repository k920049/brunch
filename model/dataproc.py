from pyspark.sql import SparkSession

from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, collect_list, struct
from pyspark.ml.feature import Word2Vec, Word2VecModel

def Main():
    spark = SparkSession.builder\
        .appName("Word2Vec")\
        .config("spark.driver.cores", "8")\
        .config("spark.driver.maxResultSize", "13312m")\
        .config("spark.driver.memory", "26624m")\
        .config("spark.executor.cores", "8")\
        .config("spark.executor.memory", "37237m")\
        .getOrCreate()


    FILE_NO = 7
    total_df = []
    for idx in range(FILE_NO):
        each_df = spark.read.format("json") \
            .option("mode", "FAILFAST") \
            .option("inferSchema", "true") \
            .load("gs://dataproc-7e10897a-5391-4ea0-b815-f6e72cf284f7-asia-east1/data/contents/data.{}".format(idx))
        total_df.append(each_df)

    df = reduce(DataFrame.unionAll, total_df)
    df = df.select(df.id, explode(df.morphs).alias("words"))

    word2vec = Word2Vec(vectorSize=300,
                        minCount=0,
                        windowSize=2,
                        numPartitions=10,
                        inputCol="words",
                        outputCol="vector")
    model = word2vec.fit(df)
    df = model.transform(df)
    final = df.groupBy("id") \
        .agg(collect_list(struct("vector")).alias("matrix"))

    final.show(1)

if __name__ == "__main__":
    Main()