# encoding=utf-8
import KNN  # 导入Knn模块
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


# group,labels=Classification_KNN.createDataSet()  # 创建了变量
# Classification_KNN.classfy0([0,0],group,labels,3)
# reload(Classification_KNN)
# datingDataMat,datingLabels = KNN.file2matrix('datingTestSet2.txt')

# KNN.Graph(datingDataMat,datingLabels)
# normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
# print(normMat, ranges, minVals)

KNN.datingClassTest('datingTestSet2.txt')