# encoding=utf-8
# 基本思想：选择K个最相似数据中出现次数最多的分类


from numpy import *
import operator  # 操作符函数
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  # 创建数组-
    labels= ['A','A','B','B']
    return group,labels
'''
KNN计算过程:
1)计算已知类别的数据集中的点与当前点之间的距离
2）按照距离递增次序排序
3）选取与当前点距离最小的K个点
4）确定前K个点所在类别的出现频率
5）返回前K个点出现频率最高的类别作为当前点的预测分类
'''
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # 读取矩阵的长度 shape[0]:第一维度矩阵的长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # tile(A,reps):生成array，tile((1,2,3),2)=array([1,2,3,1,2,3])
    # b=[1,3,5], tile(b,[2,3])=array([[1,3,5,1,3,5,1,3,5],[1,3,5,1,3,5,1,3,5]])
    sqDiffMat = diffMat**2 # 幂运算
    sqDistances = sqDiffMat.sum(axis=1) # .sum(axis=0):普通的相加；.sum(axis=1)：将矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() # 数组值从小到大的索引值
    classCount={} #
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    #  dict.get(key,x):查找键为key的value,  operator.itemgetter(1)：返回对象的第i+1个元素，相当于匿名函数
    # 数组中每个元素的字节大小 获取对象第一域的值，key为函数，制定取待排序元素的哪一项进行排序，True：降序排列
    return sortedClassCount[0][0]

'''
实例:在约会网站上使用K-邻近算法
1) 收集数据：提供文本文件
2）准备数据：使用Python解析文本文件
3）分析数据：使用Matplotlib画二维扩散图
4）训练算法：此步骤不适用于KNN
5）测试算法：使用海伦提供的部分数据作为测试样本：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误
6）使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型
'''

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() # 得到文件行数
    numberOLines = len(arrayOLines)
    returnMat = zeros((numberOLines,3)) # 创建返回的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:  # 解析文件数据到列表
        line = line.strip()  # 截取掉所有回车字符，然后使用tab字符\t，将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector

'''画图'''
def Graph(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

''' 准备数据：归一化数值'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    # normDataSet = linalg.solve(normDataSet,tile(ranges, (m, 1)))
    return normDataSet,ranges,minVals

''' 测试正确率'''
def datingClassTest(filename):
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix(filename) ## 读取文件
    normMat,ranges,minVals = autoNorm(datingDataMat)  ## 数据归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount +=1.0
    print ("the total error rate is : %f" % (errorCount/float(numTestVecs)))

''' 测试正确率'''
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats= float()


