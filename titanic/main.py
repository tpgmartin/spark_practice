import csv
import numpy as np
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
import time

# Not sure if need to specify os config
conf = pyspark.SparkConf()
sc = pyspark.SparkContext(conf=conf)
conf.getAll()

sc.setLogLevel("ERROR")

# PassengerId, Survived, Pclass, Name                     , Sex , Age, SibSp, Parch, Ticket   , Fare, Cabin, Embarked
# 1          , 0       , 3     , "Braund, Mr. Owen Harris", male, 22 , 1    , 0    , A/5 21171, 7.25,      , S

# Extract data from CSV file
spark = SparkSession.builder.appName("Titanic Trainset").master("local").getOrCreate()
trainTitanic = spark.read.csv("./titanic/data/train.csv", inferSchema=True, header=True)

print("Before --------------------")
print(trainTitanic.show(5))
print("Initial DataFrame size", trainTitanic.count())
print()

# Transform & filter 
trainTitanic = trainTitanic.withColumn("Sex", when(trainTitanic["Sex"] == "male", 0).otherwise(1))
# trainTitanic = trainTitanic.filter("Name == ''").filter("Sex == ''")

# Different to example script, feature columns may in different order
print("After --------------------")
print(trainTitanic.show(5))
print("Transformed DataFrame size", trainTitanic.count())
print()

# Not sure what columns to filter, as columns appear not to be in same order as tutorial

# Creating "labeled point" rdds, something fro MLlib
# Target label should be "Survived"
# Features should by "Pclass", "Sex", "Age", "SibSp", "Parch"

# trainTitanicLP=trainTitanic.map(lambda line: LabeledPoint(line["Survived"],[line[["Pclass", "Sex", "Age", "SibSp", "Parch"]]]))
# print(trainTitanicLP.first())

# Shouldn't need additional transformations as all features are numerical
assembler = VectorAssembler(
    inputCols=["Pclass", "Sex", "Age", "SibSp", "Parch"],
    outputCol="features")

transformedTrain = assembler.transform(trainTitanic)

(transformedTrain.select(col("Survived").alias("label"), col("features"))
  .rdd
  .map(lambda row: LabeledPoint(row.label, row.features)))

(trainData, testData) = transformedTrain.randomSplit([0.7, 0.3])

print(testData)