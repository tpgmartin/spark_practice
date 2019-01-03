import csv
import numpy as np
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
import time

# Not sure if need to specify os config
conf = pyspark.SparkConf()
sc = pyspark.SparkContext(conf=conf)
conf.getAll()

sc.setLogLevel("WARN")

# PassengerId, Survived, Pclass, Name                     , Sex , Age, SibSp, Parch, Ticket   , Fare, Cabin, Embarked
# 1          , 0       , 3     , "Braund, Mr. Owen Harris", male, 22 , 1    , 0    , A/5 21171, 7.25,      , S

# Extract data from CSV file
spark = SparkSession.builder.appName("Titanic Trainset").master("local").getOrCreate()
trainTitanic = spark.read.csv("./titanic/data/train.csv", inferSchema=True, header=True)

print("Before --------------------")
print(trainTitanic.show(5))

# Transform
trainTitanic = trainTitanic.withColumn("Sex", when(trainTitanic["Sex"] == "male", 0).otherwise(1))

print("After --------------------")
print(trainTitanic.show(5))