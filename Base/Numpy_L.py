# encoding=utf-8
from numpy import *
''' Numpy的主要对象是同种元素的多维数组，这是一个所有的元素都是一种类型，通过一个正整数元祖索引的元素表格（通常的元素是数字）。
    在Numpy中维度（dimensions）叫做轴（axes），轴的个数叫做秩（rank）
    例如，在3D空间一个点的坐标[1,2,3]是一个秩为1的数组，因为它只有一个轴，那个轴的长度为3，
    例如，[[1.,0.,0.],[0.,1.,2.]] 的秩为2，第一个维度长度为2，第二个维度长度为3'''

''' Numpy的数组类被称作ndarry，通常被称作数组，注意numpy.array和标准python库类array.array并不相同，后者只处理一维数组和提供少量功能'''

A = array([[1.,0.,0.],[0.,1.,2.]])
# A = arange(15).reshape(3, 5)

print (A.ndim) #数组轴的个数
print (A.shape)# 数组的维度，这是一个指定数组在每个维度上大小的整数元祖。例如一个n排m列的矩阵，它的shape属性将是（2,3）
print (A.size) # 数据元素的总个数
print (A.dtype) # 描述数组中元素类型的对象
print (A.itemsize) #数组中每个元素的字节大小，一个元素类型为float64的数组itemsize属性值为8=（64/8）
print (A.data) #包含数据数组元素的缓冲区，通常我们不需要使用这个属性，因为我们总是用过索引来使用数组中的元素

'''创建数组'''
A = array([[1.,0.,0.],[0.,1.,2.]])
print (A)

B = array ([(1.5,2,3),(4,5,6)]) # 数组将序列包含序列转化成二维的数组
C = array([[1,2],[2,3]],dtype = complex) # 数字类型可以在创建时显示指定
D = zeros((3,4)) # 创建一个全是0的数组，最小化了扩展数组的需求和高昂的运算代价
E = ones((4,5))
F = empty((2,3)) # 创建一个内容随机的数组，类型为 float64
G = arange(5) # array([0, 1, 2, 3，5])
H = arange(10,30,5) # 返回array([10, 15, 20, 25])

B = arange(15).reshape(3, 5) # 打印2d array
C = arange(24).reshape(2, 3,4) # 打印3d array
set_printoptions(threshold='nan') # 设置参数来完成打印

'''基本运算'''

## **2 平方

A = array( [[1,1], [0,1]] )
B = array( [[2,0], [3,4]] )
A*B #元素相称
dot(A,B)  # 矩阵相乘
D *= 3  # 更改数组中的元素
A += B
B.sum(axis=0) # 每一列的总和 指定axis参数你可以把运算应用到数组置顶的轴上
B.min(axis=1) # 每一行的最小值