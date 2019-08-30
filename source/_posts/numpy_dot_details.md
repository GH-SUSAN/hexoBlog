---
title: numpy dot 详解
---
&emsp;&emsp;用numpy编程实现LinearRegression和LogisticRegression算法时，常用到dot方法计算cost函数和gradient，所以对dot需要准确的认识，才能够保证计算过程不出问题。
&emsp;&emsp;numpy的dot方法，并不是传统意义上的点积或者说不仅仅是点积，如果你学过matlab，你会发现numpy的dot和matlab也有区别。因此，为了避免混淆这里给出numpy的dot方法常用计算和意义分析，帮助理解记忆。

# 1. numpy的向量和矩阵表示

&emsp;&emsp;numpy在向量和矩阵表示上做的不好，不如Matlab明确。
&emsp;&emsp;numpy的一维向量，其本质是一维数组，numpy默认将其作为行向量处理。
&emsp;&emsp;要表示列向量，可通过reshape转换得到，本质上是一个二维数组，numpy把他当作一个矩阵处理。
如下所示：
``` python
>> a = np.array([1,2,3])
>> a
   array([1, 2, 3])
>> a.shape
   (3,) ### 3表示向量三个元素
>> a.T
   array([1, 2, 3])  ### 转置后依然是行向量


>> b = a.reshape(-1,  1)
>> b
   array([[1],
          [2],
          [3]])   ### 这种方式得到的列向量，其实已经是二维的矩阵，所以不能叫做1维的向量
>> b.shape
   (3, 1) 


>> c = b.T
>> c
   array([[1, 2, 3]])  ### 列向量转置得到了行向量，本质上是二维的矩阵
>> c.shape
   (1, 3)
```

# 2. 一维和一维数组的dot

1. 行向量和行向量的点积
``` python
>> vec1 = np.array([1, 2, 3])
>> vec2 = np.array([1, 1, 1])
>> vec1.dot(vec2)
   6  ### 得到结果是一个标量，符合线性代数的向量点积定义
```

2. 行向量和列向量点积
``` python
>> vec1 = np.array([1, 2, 3])
>> vec2 = np.array([1, 1, 1]).reshape(-1, 1)
>> vec1.dot(vec2)
   array([6])   ### 得到结果是一维数组(行向量)，不再是标量，说明这个dot不再是线性代数意义上的点积，而是矩阵乘法
>> vec2.dot(vec1)
   ValueError: shapes (3,1) and (3,) not aligned: 1 (dim 1) != 3 (dim 0) ###果然报错了，因为矩阵乘法必须保证维度相容vec2 (3,1) 无法和vec1 (3,)相乘
```

综上，在numpy计算中，只有一维数组dot一维数组才符合数学定义，满足线性代数的向量点积。

# 3. 一维数组和二维矩阵的dot
行向量/列向量和矩阵dot，代码如下：
``` python
>> vec1 = np.array([1, 2, 3])
>> vec1
   array([1, 2, 3])

>> vec2 = vec1.reshape(-1, 1)
>> vec2
   array([[1],
          [2],
       	  [3]])

>> mat1 = np.array([[1, 2, 3],[1, 1, 1]])
>> mat1
   array([[1, 2, 3],
          [1, 1, 1]]) ### （2，3）矩阵

>> mat1.dot(vec1)
   array([14,  6])  ### (2,3)矩阵dot行向量，得到了一维行向量, 相当于(2,3) * (3,) = (2,) 

>> mat1.dot(vec2)
   array([[14],
          [ 6]])   ### (2,3)矩阵dot列向量，得到了二维列向量，相当于(2,3) * (3,1) = (2,1) 

>> vec1.dot(mat1) 
   ValueError: shapes (3,) and (2,3) not aligned: 3 (dim 0) != 2 (dim 0) ### (3,)*(2,3)不符合维度相容，报错

>> vec3 = np.array([1, 3])
   array([1, 3])
>> vec3.dot(mat1)
   array([4, 5, 6])  ### (2,) * (2,3) = (3,) 这种情况下也符合维度相容，需要注意
```

综上，一维和二维进行np.dot相当于矩阵乘法，必须满足维度相容。

# 4. 二维和二维矩阵的dot
矩阵和矩阵的dot运算，代码如下：
``` python 
>> mat1 = np.array([[1, 2, 3],[1, 1, 1]])
>> mat1
   array([[1, 2, 3],
          [1, 1, 1]]) ### （2，3）矩阵

>> mat2 = np.array([[1,0],[1,1]])
>> mat2
   array([[1, 0],
          [1, 1]])

>> mat1.dot(mat2)
   ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)  ### 报错，维度不相容

>> mat1.T.dot(mat2)
   array([[2, 1],
          [3, 1],
          [4, 1]])  ### 本质上是矩阵相乘 (3,2) * (2,2) = (2,2)
   
```

综上，二维情况下，np.dot依然是矩阵乘法，必须满足维度相容。

# 5. 总结
&emsp;&emsp;np.dot的运算规则：
1. 只有一维数组(n,)之间的dot运算，是线性代数意义上的点积，计算结果维标量。
2. 一维和多维以及多维和多维数组的dot运算，按照矩阵乘法进行运算，必须符合维度相容原则，计算结果为数组。

