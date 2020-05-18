import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession


LR_MODEL = 'lr_model'


def process(spark, train_data, test_data):
    #train_data - path to train data
    #test_data - path to test data
    train = spark.read.parquet(train_data)
    test = spark.read.parquet(test_data)
    cols = train.drop('ad_id','ctr').columns

    assembler = VectorAssembler(inputCols=cols, outputCol='features')

    lr = LinearRegression(maxIter = 40, regParam=0.4, elasticNetParam=0.8, labelCol= 'ctr') 

    pipeline = Pipeline(stages=[assembler, lr])

    model = pipeline.fit(train)
    model.write().overwrite().save(LR_MODEL)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='ctr')
    met = evaluator.evaluate(predictions , {evaluator.metricName :  'rmse'})
    print(met)
    return (met)

def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
