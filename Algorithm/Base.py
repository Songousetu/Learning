# encoding=utf-8
from numpy import *

content_v1= random.rand(4,4) # 构造了一个4*4的随机数组
content = mat(content_v1) # 将数组转化成矩阵
# content.I 矩阵的逆运算，eye(4) 单位矩阵
print (content)