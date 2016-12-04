
### Nupmy基础
## 创建adarray
import numpy as np
data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2.ndim
arr2.shape
arr1.dtype
np.zeros(10)
np.zeros((10,2))
np.zeros([10,2])
np.empty((2,3))
# 切片索引
arr[1:6]
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[:,2]
arr2d[1]
# 布尔型索引
names = np.array(['Bob','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
names == 'Bob'
data[names == 'Bob']
data[names == 'Bob',2:]
mask = (names == 'Bob') | (names == 'Will')
# python中的and和or无效，只能使用|和&之类的
# 花式索引
arr = np.empty((8,4))
for i in range(8):
	arr[i] = i
arr[[4,3,0,6]]
# 使用负数索引会从末尾开始选行
arr[[1,5,7,2],[0,3,1,2]]
arr[[1,5,7,2]][:,[0,3,1,2]]
