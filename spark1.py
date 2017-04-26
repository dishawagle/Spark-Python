from pyspark import SparkConf, SparkContext
from pyspark.sql import *
import numpy as np
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.ml.feature import MinMaxScaler
from pyspark.mllib.linalg import Vectors
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

def parse_line(line):
  split=line.split(",")
  label,features=int(split[0]),split[1:]
  return LabeledPoint(int(label),features)

def getrmse(v):
  return np.sqrt(v.map(lambda (prediction, label): (label - prediction) ** 2).mean())
  

def main():
  conf = SparkConf().setMaster("local").setAppName("Assignment 1")
  sc = SparkContext(conf=conf)
  sqlContext=SQLContext(sc)
  sc.setLogLevel("ERROR")
  #part 1
  data = sc.textFile('/home/disha/Downloads/MSD.txt',2)
  dc=data.count()
  #print data.count()
  #print data.take(40)
  sdata=data.take(40)
  #part 2
  lp=[parse_line(p) for p in sdata] 
  #part 3
  x1=list(lp[i].features[3] for i in range(40))
  x2=list(lp[i].features[4] for i in range(40))
  dataFrame = sqlContext.createDataFrame([(Vectors.dense(x1),),(Vectors.dense(x2),)], ["features"])
  scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
  scalerModel = scaler.fit(dataFrame)
  scaledData = scalerModel.transform(dataFrame)
  x=scaledData.select("scaledFeatures").map(list).collect()
  xdf= pd.DataFrame({'1':x[0][0],'2':x[1][0]})
  '''
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(xdf, cmap=plt.cm.Greys, alpha=0.8)
  fig = plt.gcf()
  fig.set_size_inches(8, 11)
  ax.set_frame_on(False)
  ax.invert_yaxis()
  ax.xaxis.tick_top()'''
  #plt.show()
  #part 4
  onlyLabels = data.map(parse_line).map(lambda point: int(point.label)).collect()
  minYear = min(onlyLabels)
  maxYear = max(onlyLabels)
  print maxYear, minYear
  lp_rdd=data.map(parse_line).map(lambda l: LabeledPoint(int(l.label)-minYear, l.features))
  #print lp_rdd.take(10)
  #part 5
  train,test=lp_rdd.randomSplit([0.8,0.2])
  model = LogisticRegressionWithLBFGS.train(train, iterations=10,numClasses=maxYear-minYear+1)
  vp = test.map(lambda p: (model.predict(p.features),p.label))
  rmse=getrmse(vp)
  print rmse
  a1=test.map(lambda p: model.predict(p.features)).collect()
  a2=test.map(lambda p: int(p.label)).collect()
  plt.scatter(a1,a2)
  plt.show()
  


  
  

  


  
if __name__ == "__main__":
    main()


  

