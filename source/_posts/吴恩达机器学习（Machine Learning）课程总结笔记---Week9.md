---
title: 参考markdown文档编辑范本
---
# 概述
&emsp;&emsp;我们前边几章学习的基本上都属于机器学习算法范畴，本章将就两个机器学习应用实例来进行学习：**异常检测和推荐系统**。
&emsp;&emsp;二者都是机器学习的典型应用，让我们开始吧！
# 课程大纲
<!-- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190709102722176.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70) -->
<img src="https://img-blog.csdnimg.cn/20190709102722176.png" width="400" align=center />

# 课程内容
## 异常检测
### 什么是异常检测
&emsp;&emsp;异常检测是机器学习算法的一个常见应用。这种算法的一个有趣之处在于：它虽然主要用于非监督学习问题，但从某些角度看，它又类似于一些监督学习问题。
&emsp;&emsp;比如，判断是一个飞机引擎出厂时候是否正常，异常检测问题就是：要知道这个新的飞机引擎$x_{test}$ 是否有某种异常，或者希望判断这个引擎是否需要进一步测试，希望知道新的数据$x_{test}$ 是不是异常的，即这个测试数据不属于该组数据的几率如何。
<!-- 如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708083229622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70) -->
<img src="https://img-blog.csdnimg.cn/20190708083229622.png" width="400" align=center />

蓝圈内的数据属于该组数据的可能性较高，越是偏远的数据，属于该组数据的可能性就越低。

&emsp;&emsp;这种方法称为密度估计，表达如下：
​	$$if \quad p(x)\left\{\begin{array}{ll}{<\varepsilon} & {\text { anomaly }} \\ {>=\varepsilon} & {\text { normal }}\end{array}\right.$$

**典型的应用还有**，
1. 识别欺骗
例如在线采集而来的有关用户的数据，一个特征向量中可能会包含如：用户多久登录一次，访问过的页面，在论坛发布的帖子数量、甚至是打字速度等。尝试根据这些特征构建一个模型，可以用这个模型来识别那些不符合该模式的用户

2. 检测一个数据中心
特征可能包含：内存使用情况，被访问的磁盘数量，CPU 的负载，网络的通信量等。根据这些特征可以构建一个模型，用来判断某些计算机是不是有可能出错。

### 高斯分布
&emsp;&emsp;上一节我们知道，异常检测是通过所谓密度估计来判断一个新的样本是否是正常的。那么按什么分布来进行密度估计，就是一个关键问题。大部分情况下，我们可以假设样本符合高斯分布，然后按照高斯分布来处理新的检测样本，进行判断。
高斯分布，也叫作正态分布，假如 x 服从高斯分布，那么我们将表示为： $x\sim N(\mu,\sigma^2)$ 。其分布概率为：
$$p(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
其中 $\mu$ 为期望值（均值）， $\sigma^2$ 为方差。
1. 期望值：$\mu=\frac{1}{m}\sum_{i=1}^{m}{x^{(i)}}$

2. 方差： $\sigma^2=\frac{1}{m}\sum_{i=1}^{m}{(x^{(i)}-\mu)}^2$
这里方差除以m而不是m-1，是因为虽然在统计学上两者的理论特性有所不同，但是在机器学习中两者区别很小，不影响最终结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708084718999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
可以看到，
1.  $\mu$决定了高斯分布的中心点位置
2. $\sigma$决定了高斯分布的峰值和宽度。

### 高斯分布算法
&emsp;&emsp;假如我们有一组 m 个无标签训练集，其中每个训练数据又有 n 个特征，那么这个训练集应该是 m 个 n 维向量构成的样本矩阵。

**step 1** ：在概率论中，对有限个样本进行参数估计
$$\mu_j = \frac{1}{m} \sum_{i=1}^{m}x_j^{(i)}\;\;\;,\;\;\; \delta^2_j = \frac{1}{m} \sum_{i=1}^{m}(x_j^{(i)}-\mu_j)^2$$
这里对参数 $\mu$ 和参数 $\delta^2$ 的估计就是二者的极大似然估计。

**step 2**： 假定每一个特征 $x_{1}$ 到 $x_{n}$ 均服从正态分布，则其模型的概率为：
$\begin{aligned} p(x) &=p\left(x_{1} ; \mu_{1}, \sigma_{1}^{2}\right) p\left(x_{2} ; \mu_{2}, \sigma_{2}^{2}\right) \cdots p\left(x_{n} ; \mu_{n}, \sigma_{n}^{2}\right) \\ &=\prod_{j=1}^{n} p\left(x_{j} ; \mu_{j}, \sigma_{j}^{2}\right) \\ &=\prod_{j=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_{j}} e x p\left(-\frac{\left(x_{j}-\mu_{j}\right)^{2}}{2 \sigma_{j}^{2}}\right) \end{aligned}$
当$p(x)<\varepsilon$时，$x$位为异常样本。

&emsp;&emsp;假定我们有两个特征 $x_1$ 、 $x_2$ ，它们都服从于高斯分布，并且通过参数估计，我们知道了分布参数：![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708085909657.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)

&emsp;&emsp;则模型 $p(x)$ 能由如下的热力图反映，热力图越热的地方，是正常样本的概率越高，参数 $\varepsilon$ 描述了一个截断高度，当概率落到了截断高度以下（下图紫色区域所示），则为异常样本：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708090024272.png)
&emsp;&emsp;将 $p(x)$ 投影到特征 $x_1$ 、$x_2$ 所在平面，下图紫色曲线就反映了 $\varepsilon$ 的投影，它是一条截断曲线，落在截断曲线以外的样本，都会被认为是异常样本：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708090034888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
### 评估系统
&emsp;&emsp;当开发一个异常检测系统时，从带标记（异常或正常）的数据着手，选择一部分正常数据用于构建训练集，用剩下的正常数据和异常数据混合构成交叉验证集和测试集。
例如：有 10000 台正常引擎的数据，有 20 台异常引擎的数据。 分配如下：
&emsp;&emsp;6000 台正常引擎的数据作为训练集
&emsp;&emsp;2000 台正常引擎和 10 台异常引擎的数据作为交叉检验集
&emsp;&emsp;2000 台正常引擎和 10 台异常引擎的数据作为测试集
&emsp;&emsp;由于异常样本是非常少的，所以整个数据集是非常偏斜的，不能单纯的用预测准确率来评估算法优劣
具体的评价方法如下：
1. 根据测试集数据，估计特征的平均值和方差并构建$p(x)$函数
2. 对交叉验证集，尝试使用不同的$\epsilon$值作为阀值，并预测数据是否异常，根据$F1$值或者查准率与召回率的比例来选择$\epsilon$
3. 选出$\epsilon$后，针对测试集进行预测，计算异常检验系统的$F1$值，或者查准率与召回率之比

### 异常检测和监督学习对比
表格一是两者在数据特点上的对比。

|有监督学习  |  异常检测|
|--|--|
| 数据分布均匀 | 数据非常偏斜，异常样本数目远小于正常样本数目 |
|可以根据对正样本的拟合来知道正样本的形态，从而预测新来的样本是否是正样本|异常的类型不一，很难根据对现有的异常样本（即正样本）的拟合来判断出异常样本的形态|
表格二，是二者在应用场景上的对比。

|有监督学习|异常检测|
|--|--|
|垃圾邮件检测|	故障检测|
|天气预测（预测雨天、晴天、或是多云天气）|某数据中心对于机器设备的监控|
|癌症的分类	|制造业判断一个零部件是否异常|

&emsp;&emsp;如果异常样本非常少，特征也不一样完全一样（比如今天飞机引擎异常是因为原因一，明天飞机引擎异常是因为原因二，谁也不知道哪天出现异常是什么原因），这种情况下就应该采用异常检测。
&emsp;&emsp;如果异常样本多，特征比较稳定有限，这种情况就应该采用监督学习。
### 特征选择
&emsp;&emsp;如果数据的分布不是高斯分布，异常检测算法也能够工作，但是最好还是将数据转换成高斯分布，例如使用对数函数：

$x = log(x+c)$，其中 $c$为非负常数； 或者$x = x^c$，$c$ 为 $0-1$ 之间的一个分数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708091546295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)

**误差分析：**
&emsp;&emsp;一个常见的问题是一些异常的数据可能也会有较高的p(x)p(x)值，因而被算法认为是正常的。这种情况下误差分析能够分析那些被算法错误预测为正常的数据，观察能否找出一些问题。可能能从问题中发现需要增加一些新的特征，增加这些新特征后获得的新算法能够更好地进行异常检测异常检测误差分析：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708091603410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;**通常可以通过将一些相关的特征进行组合**，来获得一些新的更好的特征（异常数据的该特征值异常地大或小），在检测数据中心的计算机状况的例子中，用 CPU负载与网络通信量的比例作为一个新的特征，如果该值异常地大，便有可能意味着该服务器是陷入了一些问题中。

### 多元高斯分布
#### 多元高斯分布模型
如图所示：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708180418334.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
  $x_1$ 是CPU的负载，$x_2$ 是内存的使用量。
 &emsp;&emsp;其正常样本如左图红色点所示。假如我们有一个异常的样本（图中左上角绿色点），在图中看很明显它并不是正常样本所在的范围。但是在计算概率 $p(x)$ 的时候，因为它在 $x_1$ 和 $x_2$ 的高斯分布都属于正常范围，所以该点并不会被判断为异常点。
&emsp;&emsp;这是因为在高斯分布中，它并不能察觉在蓝色椭圆处才是正常样本概率高的范围，其概率是通过圆圈逐渐向外减小。所以在同一个圆圈内，虽然在计算中概率是一样的，但是在实际上却往往有很大偏差。
所以我们开发了一种改良版的异常检测算法：多元高斯分布。
&emsp;&emsp;在一般的高斯分布模型中，计算$p(x)$的方法是： 通过分别计算每个特征对应的几率然后将其累乘起来，在多元高斯分布模型中，将构建特征的协方差矩阵，用所有的特征一起来计算$p(x)$其概率模型为：
$$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$
（其中 $|\Sigma|$ 是 $\Sigma$ 的行列式，$\mu$ 表示样本均值，$\Sigma$ 表示样本协方差矩阵。）
#### 多元高斯分布的变化
&emsp;&emsp;多元高斯分布模型也同样遵循概率分布，曲线下方的积分等于1。多元高斯分布相当于体积为1 。这样就可以通过 $\mu$ 和 $\Sigma$ (这里是协方差矩阵，原来是 $\sigma$ )的关系来判断图形的大致形状。
##### 改变$\Sigma$ 
&emsp;&emsp;$\Sigma$ 是一个协方差矩阵，所以它衡量的是方差。减小 $\Sigma$ 其宽度也随之减少，增大反之。
1. 改变$\Sigma$主对角线的数值可以进行不同方向的宽度拉伸：
$\Sigma$ 中第一个数字是衡量 $x_1$ 的，假如减少第一个数字，则可从图中观察到 $x_1$ 的范围也随之被压缩，变成了一个椭圆。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019070818094939.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
2. 改变$\Sigma$副对角线的数值可以旋转分布图像：
改变副对角线上的数据，则其图像会根据 $y=x$ 这条直线上进行高斯分布。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708181020522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708181029596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)

##### 改变$\mu$
 &emsp;&emsp;改变 $\mu$可以对分布图像进行位移：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708181151278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)

#### 算法流程
&emsp;&emsp;采用了多元高斯分布的异常检测算法流程如下：

1. 选择一些足够反映异常样本的特征 $x_j$ 。
2. 对各个样本进行参数估计：
 $$\mu=\frac{1}{m}\sum_{i=1}^{m}{x^{(i)}}$$
 $$\Sigma=\frac{1}{m}\sum_{i=1}^{m}{(x^{(i)}-\mu)(x^{(i)}-\mu)^T}$$
3. 当新的样本 x 到来时，计算 $p(x)$ ：
$$p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$
如果$p(x)<\varepsilon$ ，则认为样本 x 是异常样本。

#### 多元高斯分布模型与一般高斯分布模型的差异
&emsp;&emsp;**一般的高斯分布模型**只是多元高斯分布模型的一个约束，它将多元高斯分布的等高线约束到了如下所示同轴分布（概率密度的等高线是沿着轴向的）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708182350779.png)
&emsp;&emsp;**一般的多元高斯模型**的轮廓（等高线）总是轴对齐的，也就是 $\Sigma$ 除对角线以外的部分都是 0，当对角线以外的部分不为 0 的时候，等高线会出现斜着的，与两个轴产生一定的斜率。

&emsp;&emsp;当：$\Sigma=\left[\begin{array}{cccc}{\sigma_{1}^{2}} & {} & {} & {} \\ {} & {\sigma_{2}^{2}} & {} & {} \\ {} & {} & {} & {\sigma_{n}^{2}}\end{array}\right]$的时候，此时的多元高斯分布即是原来的多元高斯分布。（因为只有主对角线方差，并没有其它斜率的变化）
##### 模型定义
**一般高斯模型**：
$\begin{aligned} p(x) &=p\left(x_{1} ; \mu_{1}, \sigma_{1}^{2}\right) p\left(x_{2} ; \mu_{2}, \sigma_{2}^{2}\right) \cdots p\left(x_{n} ; \mu_{n}, \sigma_{n}^{2}\right) \\ &=\prod_{j=1}^{n} p\left(x_{j} ; \mu_{j}, \sigma_{j}^{2}\right) \\ &=\prod_{j=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_{j}} e x p\left(-\frac{\left(x_{j}-\mu_{j}\right)^{2}}{2 \sigma_{j}^{2}}\right) \end{aligned}$
**多元高斯模型**：
$p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$
##### 对比
|一般高斯模型|	多元高斯模型|
|--|--|
|需要手动创建一些特征来描述某些特征的相关性|	利用协方差矩阵$\Sigma$ 获得了各个特征相关性|
|计算复杂度低，适用于高维特征|计算复杂|
|在样本数目 mm 较小时也工作良好    |   需要 $\Sigma$ 可逆，亦即需要$m>n$(通常会考虑 $m>10 n$，确保有足够多的数据去拟合这些变量，更好的去评估协方差矩阵 $\Sigma$ ) 各个特征不能线性相关，如不能存在 $x_2=3x_1$ 或者 $x_3=x_1+2x_2$ |
​
**结论**：基于多元高斯分布模型的异常检测应用十分有限。

## 推荐系统
&emsp;&emsp;对机器学习来说，特征是很重要的，选择的特征，将对学习算法的性能有很大的影响。
&emsp;&emsp;如下图，每个人对电影的评分，我们需要预测某个用户对未看过的电影的可能评分。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708184136618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
以预测第3部电影第1个用户可能评的分数为例子。

&emsp;&emsp;首先我们用 $x_1$ 表示爱情浪漫电影类型， $x_2$ 表示动作片类型。上图左表右侧则为每部电影对于这两个分类的相关程度。我们默认 $x_0=1$ 。则第一部电影与两个类型的相关程度可以这样表示： $x^{(3)}=\left[ \begin{array}{ccc}1 \\0.99 \\0 \end{array} \right]$ 。然后用 $\theta^{(j)}$ 表示第 j 个用户对于该种类电影的评分。这里我们假设已经知道（详情下面再讲） $\theta^{(1)}=\left[ \begin{array}{ccc}0 \\5 \\0 \end{array} \right]$ ，那么我们用 $(\theta^{(j)})^Tx^{(i)}$ 即可计算出测第3部电影第1个用户可能评的分数。这里计算出是4.95。
### 假设函数和优化目标
**假设函数**：
&emsp;&emsp;假设对每一个用户，都训练一个线性回归模型，如下：
$$y^{(i, j)}=\left(\theta^{(j)}\right)^{T} x^{(i)}$$
&emsp;&emsp;其中，$\theta^{(j)}$是用户$j$的参数向量；$x^{(i)}$是电影$i$的特征向量。
**优化目标**：
&emsp;&emsp;针对用户 j 打分状况作出预测，我们需要：
$$\min_{(\theta^{(j)})}=\frac{1}{2}\sum_{i:r(i,j)=1}^{}{((\theta^{(j)})^T(x^{(i)})-y^{(i,j)})^2}+\frac{\lambda}{2}\sum_{k=1}^{n}{(\theta_k^{(j)})^2}$$
&emsp;&emsp;其中，$i : r(i, j)=1$，表示只计算用户$j$评价过的电影。

&emsp;&emsp;在一般的线性回归模型中，误差项和正则想都应该乘以$\frac{1}{2 m}$，在这里将$m$去掉，并且不对$\theta_{0}^{(j)}$进行正则化处理，对所有用户$1,2, \ldots, n_{u}$计算代价函数：
$$J(\theta^{(1)},\cdots,\theta^{(n_u)})=\min_{(\theta^{(1)},\cdots,\theta^{(n_u)})}=\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}^{}{((\theta^{(j)})^T(x^{(i)})-y^{(i,j)})^2}+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}{(\theta_k^{(j)})^2}$$
&emsp;&emsp;与前面所学线性回归内容的思路一致，为了计算出 $J(\theta^{(1)},\cdots,\theta^{(n_u)})$，使用梯度下降法来更新参数：
**更新偏置（插值）**：
$$\theta^{(j)}_0=\theta^{(j)}_0-\alpha \sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x^{(i)}_0$$
**更新权重**：
$$\theta^{(j)}_k=\theta^{(j)}_k-\alpha \left( \sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x^{(i)}_k+\lambda \theta^{(j)}_k \right),\;\;\; k \neq 0$$

&emsp;&emsp;其中，$n_{u}$代表用户的数量；$n_{m}$代表电影的数量；$r(i, j)$如果用户$j$给电影评过分则$r(i, j)=1$；$y^{(i, j)}$代表用户$j$给电影$i$的评分；$m^{(j)}$代表用户$j$评过分的电影的总数；
### 协同过滤
&emsp;&emsp;拥有了评价用户的$\theta$和评价商品的$x$，因此可以做到，
1. 给定$\theta$及用户对商品的评价，可以估计$x$；
2. 给定$x$及用户对商品的评价，可以估计$\theta$。

&emsp;&emsp;这样就构成了$\theta \rightarrow x \rightarrow \theta \rightarrow x . .$优化序列，构成了协同过滤算法，即能够同时优化$\theta$和$x$。
#### 协同过滤优化目标
1. 推测用户喜好，给定$x^{(1)}, \ldots, x^{\left(n_{m}\right)}$，估计$\theta^{(1)}, \ldots, \theta^{\left(n_{u}\right)}$：
$$\min _{\theta(1), \ldots, \theta^{\left(n_{u}\right)}}=\frac{1}{2} \sum_{j=1}^{n_{u}} \sum_{i : r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n}\left(\theta_{k}^{(j)}\right)^{2}$$
2. 推测商品内容，给定$\theta^{(1)}, \ldots, \theta^{\left(n_{u}\right)}$，估计$x^{(1)}, \ldots, x^{\left(n_{m}\right)}$：
$$\min _{x^{(i)}, \ldots, x^{\left(n_{m}\right)}}=\frac{1}{2} \sum_{i=1}^{n_{m}} \sum_{j : r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n}\left(x_{k}^{(i)}\right)^{2}$$
3. 协同过滤，同时优化$x^{(1)}, \ldots, x^{\left(n_{m}\right)}$及$\theta^{(1)}, \ldots, \theta^{\left(n_{u}\right)}$：
$$J\left(x^{(1)}, \ldots x^{\left(n_{n}\right)}, \theta^{(1)}, \ldots, \theta^{\left(n_{n}\right)}\right)=\frac{1}{2} \sum_{(i, j) * r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n}\left(x_{k}^{(j)}\right)^{2}+\frac{\lambda}{2} \sum_{j=1}^{n_{i}} \sum_{k=1}^{n}\left(\theta_{k}^{(j)}\right)^{2}$$

#### 算法流程
**Step 1**：初始化$x^{(1)}, \ldots, x^{\left(n_{m}\right)}, \theta^{(1)}, \ldots, \theta^{\left(n_{u}\right)}$为随机小值。
**Step 2**：使用梯度下降法最小化代价函数，
$x_{k}^{(i)} :=x_{k}^{(i)}-\alpha\left(\sum_{j : r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)} \theta_{k}^{j}+\lambda x_{k}^{(i)}\right)\right.$
$\theta_{k}^{(i)} :=\theta_{k}^{(i)}-\alpha\left(\sum_{i : r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)} x_{k}^{(i)}+\lambda \theta_{k}^{(j)}\right)\right.$
**Step 3**：如果用户偏好向量为$\theta$，而商品的特征为$x$，则可以预测用户的评价为$\theta^{T} x$。
因为协同过滤算法 $\theta$ 和 x 相互影响，因此，二者都没必要使用偏置 $\theta_0$ 和 $x_0$，即，$x \in \mathbb{R}^n$、 $\theta \in \mathbb{R}^n$ 。
### 向量化：低秩矩阵分解
将下图中，用户对电影的评价表示成为一个矩阵$Y$.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190708211903268.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
$Y=\left[\begin{array}{llll}{5} & {5} & {0} & {0} \\ {5} & {?} & {?} & {0} \\ {?} & {4} & {0} & {?} \\ {0} & {0} & {5} & {4} \\ {0} & {0} & {5} & {0}\end{array}\right]$
预测矩阵为，
$\operatorname{Predicated}=\left[\begin{array}{cccc}{\left(\theta^{(1)}\right)^{T} x^{(1)}} & {\left(\theta^{(2)}\right)^{T} x^{(1)}} & {\dots} & {\left(\theta^{\left(n_{u}\right)}\right)^{T} x^{(1)}} \\ {\left(\theta^{(1)}\right)^{T} x^{(2)}} & {\left(\theta^{(2)}\right)^{T} x^{(2)}} & {\dots} & {\left(\theta^{\left(n_{u}\right)}\right)^{T} x^{(2)}} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {\left(\theta^{(1)}\right)^{T} x^{\left(n_{m}\right)}} & {\left(\theta^{(2)}\right)^{T} x^{\left(n_{m}\right)}} & {\dots} & {\left(\theta^{\left(n_{u}\right)}\right)^{T} x^{\left(n_{m}\right)}}\end{array}\right]$
这里，令
$X=\left[\begin{array}{c}{\left(x^{(1)}\right)^{T}} \\ {\left(x^{(2)}\right)^{T}} \\ {\vdots} \\ {\left(x^{\left(n_{m}\right)}\right)^{T}}\end{array}\right], \Theta=\left[\begin{array}{c}{\left(\theta^{(1)}\right)^{T}} \\ {\left(\theta^{(2)}\right)^{T}} \\ {\vdots} \\ {\left(\theta^{\left(n_{u}\right)}\right)^{T}}\end{array}\right]$
即$X$的每一行描述了一部电影的特征，$\theta^{T}$的每一列描述了用户喜好参数，因此预测可以写为：
$$Pridicted = \Theta^{T} X$$
&emsp;&emsp;这个算法是协同过滤的向量化，也称为低秩矩阵分解(因为$\Theta$不是满秩矩阵)。
### 均值归一化
#### 为什么采用均值归一化
如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019070822291127.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;如果我们新增一个用户 Eve,并且 Eve 没有为任何电影评分,那么我们以什么为依据为 Eve 推荐电影呢?
事实上，根据最小化代价函数求解$\theta^{(5)}$：
$$J\left(x^{(1)}, \ldots x^{\left(n_{n}\right)}, \theta^{(1)}, \ldots, \theta^{\left(n_{n}\right)}\right)=\frac{1}{2} \sum_{(i, j) * r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n}\left(x_{k}^{(j)}\right)^{2}+\frac{\lambda}{2} \sum_{j=1}^{n_{i}} \sum_{k=1}^{n}\left(\theta_{k}^{(j)}\right)^{2}$$
由于$r^{(i,j)}=0$，因此真正有效的项是$\frac{\lambda}{2}\left[\left(\theta_{1}^{(5)}\right)^{2}+\left(\theta_{2}^{(5)}\right)^{2}\right]$，因此得到的$\theta^{(5)}=\left[\begin{array}{l}{0} \\ {0}\end{array}\right]$。
&emsp;&emsp;这样，就导致了计算出来的结果$y^{(i, 5)} = 0$，意味着什么电影都不推荐，这显然是不符合预想的。
&emsp;&emsp;因此，可以采用均值归一化解决这个问题。
#### 均值归一化算法
**Step 1**：先求取各个电影的平均得分$\mu$：
$\mu=\left(\begin{array}{c}{2.5} \\ {2.5} \\ {2} \\ {2.25} \\ {1.25}\end{array}\right)$
**Step 2**：对Y进行均值归一化：
$Y=Y-\mu=\left[\begin{array}{ccccc}{2.5} & {2.5} & {-2.5} & {-2.5} & {?} \\ {2.5} & {?} & {?} & {-2.5} & {?} \\ {?} & {-2} & {-2} & {?} & {?} \\ {-2.25} & {-2.25} & {2.75} & {1.75} & {?} \\ {-1.25} & {-1.25} & {3.75} & {-1.25} & {?}\end{array}\right]$
**Step 3**：求解$\theta^{(5)}=\left[\begin{array}{l}{0} \\ {0}\end{array}\right]$。
**Step 4**：计算$y^{(i, 5)}=\left(\theta^{(5)}\right)^{T} x^{(i)}+\mu_{i}=\mu_{i}$
&emsp;&emsp;也就是说在未给定任何评价的时候，我们用其他用户的平均评价来作为该用户的默认评价。
&emsp;&emsp;貌似利用均值标准化让用户的初始评价预测客观了些，但这也是盲目的，不准确的。实际环境中，如果一个电影确实没被评价过，那么它没有任何理由被推荐给用户。
# 课后编程作业
&emsp;&emsp;我将课后编程作业的参考答案上传到了github上，包括了octave版本和python版本，大家可参考使用。
https://github.com/GH-SUSAN/Machine-Learning-MarkDown/tree/master/week9
# 总结
&emsp;&emsp;本周学习了机器学习的两个重要应用，异常检测和推荐系统。推荐系统虽然学术上的关注比较少，但实际应用却很多，值得关注。
