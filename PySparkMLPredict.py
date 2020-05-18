import io
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


LR_MODEL = 'lr_model'


def process(spark, input_file, output_file):
    #input_file - data for ctr prediction
    #output_file - path to final predictions [ads_id, prediction]
    from pyspark.ml.pipeline import PipelineModel

    validation = spark.read.parquet(input_file)

    model = PipelineModel.load(LR_MODEL)
    preds = model.transform(validation)

    preds.select('ad_id','prediction').write.option('header', 'true').csv(output_file)


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
