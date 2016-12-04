## --------------------------------------------------------------------------------------------------------------------------
## Pandas 入门
## --------------------------------------------------------------------------------------------------------------------------
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
##
## pandas 的数据结构介绍
##
## Series
obj = Series([4,7,-5,3])
obj2 = Series([4,7,-5,3],index=['b','d','a','c'])
sdata = {'Ohio':35000, 'Texas':71000,'Oregon':16000, 'Utah':5000}
obj3 = Series(sdata)
states = ['California','Ohio','Oregon','Texas']
obj4 = Series(sdata, index=states)
# 如果某个索引值在字典中没有，那么值就为NaN
## 数组形式
obj.values
## 索引对象
obj2.index
obj2['a']
obj2[['c','a','d']]
obj2[obj2>0]
## 运算
obj2*2
np.exp(obj2)
'b' in obj2
## 缺失值
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()
## 自动对齐，对于索引的元素运算
obj3 + obj4
## Series对象本身和索引都有一个name属性
obj4.name = 'population'
obj4.index.name = 'state'

## DataFrame
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
'year':[2000,2001,2002,2001,2002],
'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = DataFrame(data)
pop = {'Nevada':{2001:2.4,2002:2.9},
'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3 = DataFrame(pop)
# 指定列就按照指定列排序
DataFrame(data, columns=['year','state','pop'])
DataFrame(data, columns=['year','state'])
frame2 = DataFrame(data, columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
DataFrame(pop, index=[2001,2002,2003])
frame2.columns
frame2['state']
frame2.year
frame2.ix['three']
# 使用Series复制，会精确匹配DataFrame
val = Series([-1.2,-1.5,-1.7], index=['two','four','five'])
frame2['debt'] = val
# 转置
frame3.T
# index和columns的名称
frame3.index.name
frame3.columns.name
# ndarray形式
frame3.values

## 索引对象
obj = Series(range(3), index=['a','b','c'])
index = obj.index
# index对象是不可以修改的
obj2 = Series([1.5, -2.5, 0], index=index)

## 重新索引
obj = Series([4.5,7.2,-5.3,3.6], index=['d','b','a','c'])
obj2 = obj.reindex(['a','b','c','d','e'])
obj2 = obj.reindex(['a','b','c','e'])
# 如果某个索引值不存在，那么默认是空缺值
# 设置填充值
obj2 = obj.reindex(['a','b','c','d','e'], fill_value=0)
# 设置前向值填充，默认填充前一个
obj3 = Series(['blue','purple','yellow'],index=[0,2,4])
obj3.reindex(range(6), method='ffill')
# reindex默认重新索引行，但是也可以重新索引列
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'],
	columns=['Ohio','Texas','California'])
states = ['Texas','Utah','California']
frame2 = frame.reindex(columns=states)

## 丢掉指定轴上的项
obj = Series(np.arange(5.), index=['a','b','c','d','e'])
obj.drop('c')
obj.drop(['d','c'])
# 丢弃列
data = DataFrame(np.arange(16).reshape(4,4),
	index = ['Ohio','Colorado','Utah','New York'],
	columns = ['one','two','three','four'])
data.drop(['one','four'], axis=1)

## 索引、选取和过滤
data.ix['Colorado',['two','three']]
data.ix['Colorado',[3,0,1]]
data.ix['Colorado']
data['two']
data.two

## 算数运算对齐
s1 = Series([7.3,-2.5,3.4,1.5], index=['a','b','d','e'])
s2 = Series([-2.1,3.6,-1.5,4,3.1], index=['a','c','e','f','g'])
s1 + s2
df1 = DataFrame(np.arange(9).reshape((3,3)), columns=list('bcd'), index=['Ohio','Texas','Colorado'])
df2 = DataFrame(np.arange(12).reshape((4,3)),columns=list('bde'), index=['Utah','Ohio','Texas','Oregon'])
df1 + df2
# 所谓的对齐，就是索引相同的值做运算
# 填充值，可以给对不上的对象填充一个特殊值
df1.add(df2, fill_value=0)
# 这个只会填充df2中没有的对象。

## DataFrame和Series之间的运算
frame = DataFrame(np.arange(12).reshape((4,3)), columns=list('bde'), index=['Utah','Utahs','Texas','Oregon'])
series = frame.ix[0]
frame - series
# 每列都会减对应元素，这种被称为沿着行广播，如果想要沿着列广播，可以如下操作
series3 = frame['d']
series3 = frame.ix['d',:]
frame.sub(series3, axis=0)

## 函数的应用和映射
frame = DataFrame(np.random.randn(4,3), columns=list('bde'),
	index=['Utah','Ohio','Texas','Oregon'])
f = lambda x :x.max()-x.min()
# 每列使用函数f
frame.apply(f)
# 每行使用函数f
frame.apply(f, axis=1)
# 返回多个值的函数
def f(x):
	return Series([x.min(), x.max()], index=['min','max'])
frame.apply(f)
# python的元素级的函数
format = lambda x: '%.2f' % x
frame.applymap(format)

## 排序和排名
obj = Series(range(4),index=['d','a','b','c'])
obj. ()
frame = DataFrame(np.arange(8).reshape((2,4)), index=['three','one'],
	columns=['d','a','b','c'])
frame.sort_index()
# 按列名排序
frame.sort_index(axis=1)
# 降序排
frame.sort_index(axis=1, ascending=False)
# 对值进行排序，这个只能对Series使用
obj = Series([4,7,-3,2])
obj.order()
# 排序时缺失值都会被放在末尾
# 对多列进行排序
frame = DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
frame.sort_index(by=['a','b'])
frame.order(by=['a','b'])
# 排名
obj = Series([7,-5,7,4,2,0,4])
obj.rank()
# 对于相同值，按照出现次序排
obj.rank(method='first')
# 降序
obj.rank(ascending=False,method='max')
# 对列计算排名
frame = DataFrame({'b':[4.3,7,-3,2],'a':[0,1,0,1],'c':[-2,5,8,-2.5]})
frame.rank(axis=1)

## 带有重复值的轴索引
obj = Series(range(5), index=['a','a','b','b','c'])
# 检验是否唯一
obj.index.is_unique
# 一个索引有多个值，那么该索引就会返回多个值。
obj['a']

## 汇总和计算描述统计
df = DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],
	index=['a','b','c','d'], columns=['one','two'])
# 对列
df.sum()
# 对行
df.sum(axis=1)
# 默认会排除NA，但是可以通过skipna禁用该功能
df.mean(axis=1,skipna=False)
# 返回最大值的索引
df.idxmax()
# 累加
df.cumsum()
df.describe()
# 相关系数
returns.MSFT.corr(returns.IBM)
returns.corr()
returns.cov()
returns.corrwith(returns.IBM)

## 唯一值，值计数以及成员资格
obj = Series(['c','a','d','a','a','b','b','c','c'])
uniques = obj.unique()
# 统计个数
obj.value_counts()
# 统计个数后默认排序，也可以不排序
pd.value_counts(obj.values, sort=False)
# 判断是否存在
mask = obj.isin(['b','c'])
obj[mask]
# 分别对每列统计频数
data = DataFrame({'Qu1':[1,3,4,3,4],
	'Qu2':[2,3,1,2,3],
	'Qu3':[2,3,1,2,3]})
data.apply(pd.value_counts)

## 处理缺失数据
string_data = Series(['aardvark','artichoke',np.nan,'avocado'])
# Python中的空值会被处理为NA
string_data[0] = None
string_data.isnull()
# 滤除缺失数据
from numpy import nan as NA
data = Series([1,NA,3.5,NA,7])
data.dropna()
data[data.notnull()]
# 对于DataFrame
data = D ataFrame([[1.,6.5,3.],[1.,NA,NA],
	[NA,NA,NA],[NA,6.5,3.]])
cleaned = data.dropna()
# 丢掉全为NA的行
data.dropna(how='all')
# 丢掉全为NA的列
data.dropna(axis=1,show='all')
# 填充缺失数据
df.fillna(0# 不同列填充不同的值
)
df.fillna({'one':0.5,'two':-1})
# 对现有对象进行修改
_ = df.fillna({'one':0.5,'two':-1}, inplace=True)
# 前向填充
df.fillna(method='ffill')
# 使用均值填充
data.fillna(data.mean())

## --------------------------------------------------------------------------------------------------------------------------
## 数据规整化：整理、转换、合并、重塑
## --------------------------------------------------------------------------------------------------------------------------
## 数据库风格的DataFrame合并
df1 = DataFrame({'key':['b','b','a','c','a','a','b'],
	'data1':range(7)})
df2 = DataFrame({'key':['a','b','d'], 'data2':range(3)})
pd.merge(df1, df2)
# 指定key
pd.merge(df1,df2,on='key')
df3 = DataFrame({'lkey':['b','b','a','c','a','a','b'],
	'data1':range(7)})
df4 = DataFrame({'rkey':['a','b','d'],
	'data2':range(3)})
# key的名称不一样
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# 全连接
pd.merge(df3, df4, left_on='lkey', right_on='rkey', how='outer')
# 左连接
pd.merge(df3, df4, left_on='lkey', right_on='rkey', how='left')
# 右连接
pd.merge(df3, df4, left_on='lkey', right_on='rkey', how='right')
# 多个键合并
left = DataFrame({'key1':['foo','foo','bar'],
	'key2':['one','two','one'],
	'lval':[1,2,3]})
right = DataFrame({'key1':['foo','foo','bar','bar'],
	'key2':['one','one','one','two'],
	'rval':[4,5,6,7]})
pd.merge(left, right, on=['key1','key2'],how='outer')
# 索引上的合并
left1 = DataFrame({'key':['a','b','a','a','b','c'],
	'value':range(6)})
right1 = DataFrame({'group_val':[3.5,7]}, index=['a','b'])
pd.merge(left1, right1, left_on='key',right_index=True)
pd.merge(left1, right1, left_on='key', right_index=True,how='outer')

## 轴向连接
s1 = Series([0,1],index=['a','b'])
s2 = Series([2,3,4],index=['c','d','e'])
s3 = Series([5,6],index=['f','g'])
pd.concat([s1,s2,s3])
## 横向拼接
pd.concat([s1, s2, s3], axis=1)
## 只要交集
s4 = pd.concat([s1*5, s3])
pd.concat([s1,s4], axis=1, join='inner')
## 指定拼接后索引
pd.concat([s1,s4], axis=1, join_axes=[['a','c','b','e']])
## 给拼接的各个部分起名字
pd.concat([s1,s1,s3],keys=['one','two','three'])
pd.concat([s1,s1,s3],axis=1,keys=['one','two','three'])
df1 = DataFrame(np.arange(6).reshape(3,2),index=['a','b','c'],columns=['one','two'])
df2 = DataFrame(5+np.arange(4).reshape(2,2),index=['a','c'],columns=['one','four'])
pd.concat([df1,df2])
## 合并重叠数据
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
	index=['f','e','d','c','b','a'])
b = Series(np.arange(len(a), dtype=np.float64),
	index=['f','e','d','c','b','a'])
b[-1] = np.nan
# np中实现ifelse语句，a中空值位置用b替代
np.where(pd.isnull(a),b,a)
# pd中类似函数，b中控制用a替代
b[:-2].combine_first(a[2:])
# DataFrame中使用
df1 = DataFrame({'a':[1.,np.nan,5.,np.nan],
	'b':[np.nan,2.,np.nan, 6.],
	'c':range(2,18,4)})
df2 = DataFrame({'a':[5.,4.,np.nan,3.,7.],
	'b':[np.nan,3.,4,6.,8.]})
df1.combine_first(df2)
## 移除重复数据
data = DataFrame({'k1':['one']*3+['two']*4,
	'k2':[1,1,2,3,3,4,4]})
data.duplicated()
# 去除重复值，默认留第一个
data.drop_duplicates()
# 根据某一列去除重复值
data['v1'] = range(7)
data.drop_duplicates(['k1'])
# 保留最后一个
data.drop_duplicates(['k1','k2'], take_last=True)

## 利用函数或映射进行数据转换
data = DataFrame({'food':['bacon','pulled pork','bacon','Pastrami',
	'corned beef','Bacon','pastrami','honey ham','nova lox'],
	'ounces':[4,3,12,6,7.5,8,3,5,6]})
meat_to_animal = {
	'bacon':'pig',
	'pulled pork':'pig',
	'pastrami':'cow',
	'corned beef':'cow',
	'honey ham':'pig',
	'nova lox':'salmon'
}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data['food'].map(lambda x: meat_to_animal[x.lower()])
## 替换值
data = Series([1.,-999.,2.,-999.,-1000.,3.])
data.replace(-999, np.nan)
data.replace([-999,-1000],np.nan)
# 替换为不同值
data.replace([-999,-1000],[np.nan,0])
data.replace({-999:np.nan,-1000:0})

## 离散化和面元划分
ages = [20,22,25,27,21,23,37,31,61,45,41,32]
ages = Series([20,22,25,27,21,23,37,31,61,45,41,32])
bins = [18,25,35,60,100]
cats = pd.cut(ages, bins)
cats.labels
cats.levels
# 默认左开右闭，也可以左闭右开
pd.cut(ages, [18,26,36,61,100],right=False)
# 自定义标签
group_names = ['Youth','YoungAdult','MiddleAged','Senior']
pd.cut(ages, bins, labels=group_names)
# 等段切割
data = np.random.rand(20)
pd.cut(data,4,precision=2)
# 分位数
data = np.random.randn(1000)
pd.qcut(data,4)
# 自定义切割
pd.qcut(data,[0,0.1,0.5,0.9,1.])

## 哑变量
df = DataFrame({'key':['b','b','a','c','a','a'],
	'data1':range(6)})
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'],prefix='key')
df[['data1']].join(dummies)

## --------------------------------------------------------------------------------------------------------------------------
## 数据聚合与分组运算
## --------------------------------------------------------------------------------------------------------------------------
# 分组
df = DataFrame({'key1':['a','a','b','b','a'],
	'key2':['one','two','one','two','one'],
	'data1':np.random.randn(5),
	'data2':np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
grouped.mean()
# 多个key分组
means = df['data1'].groupby([df['key1'],df['key2']]).mean()
means.unstack()
# 使用数组作为键
states = np.array(['Ohio','California','California','Ohio','Ohio'])
years = np.array([2005,2005,2006,2005,2006])
df['data1'].groupby([states, years]).mean()
# 使用其他聚合函数
df.groupby(['key1','key2']).size()

## 对分组进行迭代
for name, group in df.groupby('key1'):
	print name
	print group
# 多键分组的迭代
for (key1,key2), group in df.groupby(['key1','key2']):
	print k1,k2
## 选取一个或一组列
df.groupby('key1')['data1'].mean()
df.groupby('key1')[['data1']].mean()
## 通过字典或Series进行分组
people = DataFrame(np.random.randn(5,5),
	columns=['a','b','c','d','e'],
	index=['Joe','Steve','Wes','Jim','Travis'])
people.ix[2:3, ['b','c']] = np.nan
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
by_columns = people.groupby(mapping, axis=1)
by_columns.sum()

## 数据聚合
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)
def peak_to_peak(arr):
	return arr.max() -arr.min()
grouped.agg(peak_to_peak)
grouped.describe()  

tips = pd.read_csv('ch08/tips.csv')
tips.tip_pct = tips.tip / tips.total_bill
# 这样没法创建新列
tips['tip_pct'] = tips.tip / tips.total_bill
group_pct = tips.groupby(['sex','smoker'])['tip_pct']
group_pct.agg(['mean','std',peak_to_peak])
# 设置列名
group_pct.agg([('foo','mean'),('bar','std')])
## 以无索引的形式返回聚合数据
tips.groupby(['sex','smoker'], as_index=False).mean()
## 给聚合后变量加前缀
df.groupby('key1').mean().add_prefix('mean_')
## transform
key = ['one','two','one','two','one']
people.groupby(key).transform(np.mean)

## apply
def top(df, n=5, column='tip_pct'):
	return df.sort_index(by=column)[-n:]
top(tips, n=6)
tips.groupby('smoker').apply(top)
tips.groupby(['smoker','day']).apply(top, n=1, column='total_bill')


## 示例：随机采样和排列
suits = ['H','S','C','D']
card_val = (range(1,11)+[10]*3)*4
base_names = ['A']+range(2,11)+['J','K','Q']
cards=[]
for suit in ['H','S','C','D']:
  cards.extend(str(num)+suit for num in base_names)
deck = Series(card_val, index=cards)
# 随机抽5张
def draw(deck, n=5):
	return deck.take(np.random.permutation(len(deck))[:n])
draw(deck)
# 每个花色随机抽2张
get_suit = lambda card: card[-1]
deck.groupby(get_suit).apply(draw, n=2)
deck.groupby(get_suit,group_keys=False).apply(draw,n=2)

## 示例：分组加权平均数和相关系数
df = DataFrame({'category':['a','a','a','a','b','b','b','b'],
	'data':np.random.randn(8),
	'weights':np.random.rand(8)})
grouped = df.groupby('category')
get_wavg = lambda g:np.average(g['data'],weights=g['weights'])
grouped.apply(get_wavg)



