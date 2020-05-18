import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,countDistinct
from pyspark.sql import functions as F
from pyspark.sql import Window

def process(spark, input_file, target_path):
    # TODO Ваш код
    click = spark.read.parquet(input_file)
    #define window
    window = Window.partitionBy("ad_id")
    # aggregations for day_count column,queries for is_CPM and is_CPC columns and query for CTR
    aggregations = [F.size(F.collect_set("date").over(window)).alias('day_count').cast('integer'),\
                when(col('ad_cost_type')=='CPM',1).otherwise(0).cast('integer').alias('is_CPM'),\
                when(col('ad_cost_type')=='CPC',1).otherwise(0).cast('integer').alias('is_CPC'),\
                (F.count(when (col("event")=='click',1)).over(window)/F.count(when (col("event")=='view',1)).over(window)).alias("CTR").cast("double")]
    #select all necessary data
    filtered_data = click.select(col('ad_id').cast('integer'),col('target_audience_count').cast('decimal'), \
    col('has_video').cast('integer'),col('ad_cost').cast('double'),*aggregations).dropDuplicates()

    
    #train_test_val split
    splits = filtered_data.randomSplit([0.5,0.25,0.25], seed = 77)
    ##save data
    splits[0].write.parquet(target_path + '/train')
    splits[1].write.parquet(target_path + '/test')
    splits[2].write.parquet(target_path + '/validate')
    


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
