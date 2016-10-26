from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LassoWithSGD
from numpy import array
from pyspark import SparkContext, SparkConf
import numpy as np

conf = SparkConf().setAppName("log-reg")
sc = SparkContext(conf=conf)
data = [LabeledPoint(0.0, [0.0]),LabeledPoint(1.0, [1.0]),LabeledPoint(3.0, [2.0]),LabeledPoint(2.0, [3.0])]
lrm = LassoWithSGD.train(sc.parallelize(data), initialWeights=array([1.0]))
print abs(lrm.predict(np.array([0.0])) - 0) < 0.5
