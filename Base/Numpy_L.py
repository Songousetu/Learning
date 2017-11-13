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
B.cumsum(axis=1) # 每行的累加 cumulative sum along each row
exp(B) # 各个元素以e为底的指数函数
sqrt(B) # 各个元素的平方
add(A,B) # 加法
'''更多函数alltrue, any, apply along axis, argmax, argmin, argsort, average,
bincount, ceil, clip, conj, conjugate, corrcoef, cov, cross, cumprod,
cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median,
 min, minimum, nonzero, outer, prod, re, round, sometrue, sort, std,
 sum, trace, transpose, var, vdot, vectorize, where
 https://docs.scipy.org/doc/numpy/reference/routines.html'''

'''索引，切片和迭代'''
a = arange(10)**3 #array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
a[2] # 8
a[:6:2] = -1000 #equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000,
                #array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
a[: : -1] # reversed a,倒过来
# array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
# for i in a :
 #   print (i**(1/3.)) #nan 1.0 nan 3.0 nan 5.0 6.0 7.0 8.0 9.0 开三次方，结果为浮点型

'''多维数组可以每个轴有一个索引。这些索引由一个逗号分割的元组给出'''

def f(x,y):
    return 10*x+y
b = fromfunction(f,(5,4),dtype=int)
'''
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
'''
b[2,3] # 23 第三行的第四列
b[0:5,1] #each row in the second column of b, 1到5行的第二列（不包括第六行）
b[:, 1]  # 所有行的第二列
b[1:3, : ] # 第2，3行 的所有列
b[-1]   # the last row. Equivalent to b[-1,:],最后一行
#当少于轴数的索引被提供时，确失的索引被认为是整个切片：

'''b[i]中括号中的表达式被当作i和一系列:，来代表剩下的轴。NumPy也允许你使用“点”像b[i,...]。
点(…)代表许多产生一个完整的索引元组必要的分号。如果x是秩为5的数组(即它有5个轴)，那么:

x[1,2,…] 等同于 x[1,2,:,:,:],
x[…,3] 等同于 x[:,:,:,:,3]
x[4,…,5,:] 等同 x[4,:,:,5,:].
'''
for element in b.flat:
    print (element)
     # 想对每个数组中元素进行运算，我们可以使用flat属性，该属性是数组元素的一个迭代器:
'''newaxis,添加新维度的
   ndenumerate,enumerate函数用于遍历序列中的元素以及它们的下标
   indices,
   index，监测是否存在，定位
'''

'''更改数组的形状'''
a = floor(10*random.random((3,4)))
'''   array([[ 7.,  5.,  9.,  3.],
       [ 7.,  2.,  7.,  8.],
       [ 6.,  8.,  3.,  2.]])
'''
a.shape # (3,4)
a.ravel() # array([ 7.,  5.,  9.,  3.,  7.,  2.,  7.,  8.,  6.,  8.,  3.,  2.])# flatten the array
a.shape = (6, 2) #改变参数的形状并返回它，
a.transpose() # array([[ 7.,  9.,  7.,  7.,  6.,  3.],[ 5.,  3.,  2.,  8.,  8.,  2.]])
a.reshape((2,6))#改变参数的形状并返回它，
a.resize((2,6)) #改变数组自身，
'''组合(stack)不同的数组'''
a = floor(10*random.random((2,2))) #array([[ 1.,  1.],[ 5.,  8.]])
b = floor(10*random.random((2,2))) # array([[ 3.,  3.],[ 6.,  0.]])
vstack((a,b)) # 纵向相加
'''array([[ 1.,  1.],
       [ 5.,  8.],
       [ 3.,  3.],
       [ 6.,  0.]])'''
hstack((a,b)) # 横向相加,column_stack((a,b)) 也是如此，with 2D arrays
'''array([[ 1.,  1.,  3.,  3.],
       [ 5.,  8.,  6.,  0.]]) '''
''''''
a=array([4.,2.])
b=array([2.,8.])
a[:,newaxis]  # This allows to have a 2D columns vector, array([[ 4.],[ 2.]])
column_stack((a[:,newaxis],b[:,newaxis])) #array([[ 4.,  2.],[ 2.,  8.]])
vstack((a[:,newaxis],b[:,newaxis])) # The behavior of vstack is different,array([[ 4.],[ 2.],[ 2.],[ 8.]])
#对那些维度比二维更高的数组，hstack沿着第二个轴组合，vstack沿着第一个轴组合,concatenate允许可选参数给出组合时沿着的轴
#在复杂情况下，r_[]和c_[]对创建沿着一个方向组合的数很有用，它们允许范围符号(“:”):
r_[1:4,0,4] # array([1, 2, 3, 0, 4]),合并从1到4的数，再加上0和4
#当使用数组作为参数时，r_和c_的默认行为和vstack和hstack很像，但是允许可选的参数给出组合所沿着的轴的代号。

'''将一个数组分割(split)成几个小数组,
使用hsplit你能将数组沿着它的水平轴分割，或者指定返回相同形状数组的个数，或者指定在哪些列后发生分割:'''
a = floor(10*random.random((2,12)))
hsplit(a,3) # # Split a into 3,
'''[array([[ 8.,  8.,  3.,  9.],
       [ 0.,  3.,  2.,  9.]]), array([[ 0.,  4.,  3.,  0.],
       [ 6.,  0.,  4.,  5.]]), array([[ 0.,  6.,  4.,  4.],
       [ 7.,  5.,  1.,  4.]])]'''
hsplit(a,(3,4))   # Split a after the third and the fourth column
'''[array([[ 8.,  8.,  3.],
       [ 0.,  3.,  2.]]), array([[ 9.],
       [ 9.]]), array([[ 0.,  4.,  3.,  0.,  0.,  6.,  4.,  4.],
       [ 6.,  0.,  4.,  5.,  7.,  5.,  1.,  4.]])] '''
# vsplit沿着纵向的轴分割，array split允许指定沿哪个轴分割。

'''复制和视图，有三种情况'''
# 完全不拷贝，简单的赋值不拷贝数组对象或他们的数据
a = arange(12)
b = a            # no new object is created
b is a           #调用 True  a and b are two names for the same ndarray object
b.shape = 3,4    # changes the shape of a
a.shape  #(3, 4)
# Python 传递不定对象作为参考4，所以函数调用不拷贝数组。
#视图(view)和浅复制，不同的数组对象分享同一个数据，视图方式创建一个新的数组对象指向同一数据
c = a.view()
c is a #False
c.base is a       #True                 # c is a view of the data owned by a
c.flags.owndata #False

c.shape = 2,6                      # a's shape doesn't change
a.shape  # (3, 4)
c[0,4] = 1234                      # a's data changes
print (a)
'''array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
'''
# 切片数组返回它的一个视图：
s = a[ : , 1:3] # spaces added for clarity; could also be written "s = a[:,1:3]"
s[:] = 10 # s[:] is a view of s. Note the difference between s=10 and s[:]=10
print (a)
'''array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
'''
# 深复制，完全复制数组和它的数据
d = a.copy()  # a new array object with new data is created
d is a # False
d.base is a         # False     d doesn't share anything with a
d[0, 0] = 9999 # 不会改变a
'''array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
'''

'''各种索引'''
#数组索引
a = arange(12)**2                          # the first 12 square numbers
i = array( [ 1,1,3,8,5 ] )                 # an array of indices
a[i]  # array([ 1,  1,  9, 64, 25])
j = array( [ [ 3, 4], [ 9, 7 ] ] )         # a bidimensional array of indices
a[j]         # the same shape as j
# array([[ 9, 16],
#       [81, 49]])
# 以下示例通过将图片标签用调色版转换成色彩图像展示了这种行为。
palette = array( [ [0,0,0],                # black
        [255,0,0],              # red
        [0,255,0],              # green
        [0,0,255],              # blue
        [255,255,255] ] )       # white

image = array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
               [ 0, 3, 4, 0 ]  ] )
palette[image]
# 结果：生成多维数组如下，
array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],
       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])
# 我们也可以给出不止一维的索引，每一维的索引数组必须有相同的形状
a = arange(12).reshape(3,4)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
i = array( [ [0,1],  [1,2] ] )         # indices for the first dim of a
j = array( [ [2,1],  [3,3] ] )                    # indices for the second dim
a[i,j]  # i and j must have equal shape
array([[ 2,  5],
       [ 7, 11]])
a[i,2]
array([[ 2,  6],
       [ 6, 10]])

a[:,j]  # 每一行 i.e., a[ : , j]
# 结果：
array([[[ 2,  1],
        [ 3,  3]],
       [[ 6,  5],
        [ 7,  7]],
       [[10,  9],
        [11, 11]]])