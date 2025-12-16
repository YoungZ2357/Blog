---
title: "逻辑回归"
date: 2025-12-09
draft: false
tags: ["监督学习", "分类"]
categories: ["学习笔记"]
---

## 1.算法总体


逻辑回归通过将**线性回归的输出**映射到**概率空间**来预测事件发生给概率：
$$
P(y=1|x;\theta)=h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
该公式表示：在模型参数为$\theta$，输入特征值为$x$的情况下，分类标签$y=1$的概率为$h_{\theta}(x)$。
该模型一次**推理**的伪代码大致如下：

**算法：LogisticForward**
输入：特征值$x$
参数：向量$\theta$
输出：数据为正类别的概率值$P(y=1|x;\theta)$

1. 计算线性组合$z = \theta^T x$
2. 应用Sigmoid函数：$p=\sigma(z)=1/(1+e^{-z})$
3. 分类决策：若$p\leq 0.5$，则$\hat{y}=1$，否则$\hat{y}=0$
4. 返回$\hat{y}$

该模型通过**梯度下降**的**训练过程**伪代码大致如下：

**算法：LogisticTrainWithGD**
输入：训练集$(X, y)$，其中$X\in \mathbb{R}^{m\times n}$
超参数：学习率$\alpha$，迭代次数$T$
输出：参数向量$\theta$

1. 初始化参数$\theta$  # 初始化可以全0，也可以正态分布随机初始化等方法
2. 对于$t=1$到$T$：
		1. 初始化梯度$g\leftarrow 0$
		2. 对于$i=1$到$m$：
			1. $p_i \leftarrow$ **LogisticForward($x_i$, $\theta$)**
			2. $g \leftarrow g+(p_i-y_i)\cdot x_i$   # 计算新梯度
		3. 更新参数：$\theta \leftarrow \theta - \frac{\alpha}{m}\cdot g$  # 梯度下降更新，**详见目标函数部分**
3. 返回$\theta$

要使用其他方法更新，用对应的公式替代计算梯度和更新参数的两个步骤即可

> 实践过程中通常表现为直接修改对应物理地址的值，而非返回值
## 2.目标函数 
### 2.1 目标函数公式
给定训练数据$(x_i, y_i)$，目标函数为：
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y_ilog(h_{\theta}(x_i))+(1-y_i)log(1-h_{\theta}(x_i))]
$$
其中$h_{\theta}$是Sigmoid函数：
$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
Sigmoid函数可用于将输出值转化为二分类概率值，多分类问题则使用Softmax函数。
![Sigmoid和Softmax函数的作用](imgs/lgstr.png)


该目标函数也叫做**逻辑损失(统计学角度)** ，在二分类中等价于 **交叉熵损失(信息论角度)**，可通过Softmax扩展到多分类问题。拟合目标是**最小化目标函数**。


目标函数性质如下

| 性质名称    | 性质            | 注释                               |
| ------- | ------------- | -------------------------------- |
| **凸性**  | 凸，非强凸         | 唯一解具有理论保障，可以通过增加二次正则化项保证其强凸以加速收敛 |
| **光滑性** | 满足Lipschitz条件 | 无穷可微，可以理论分析收敛速度                  |
### 2.2 参数的梯度下降更新
> 下降方法是一种一阶优化方法，其原理详见[[梯度下降及其收敛性|一阶方法-梯度下降]]


目标函数的**梯度**如下，计算过程略：
$$
\nabla J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)
$$
一般梯度下降中，以如下的方式更新参数$\theta$：
$$
\theta \leftarrow \theta - \alpha \nabla J(\theta)
$$
其中$\alpha$是学习率，也就是更新的步长，$\alpha$越大，更新越快，但也更容易错过最优点。$\leftarrow$为赋值符号，作用等同于Python中的等号。

> 需要注意的是，许多的更新方法都是隐式地最小化目标函数，实际代码中损失值并未参与计算，而是作为一种评价指标。
### 2.3 参数拟牛顿法更新
> 拟牛顿法是一种介于梯度下降（一阶）和牛顿法（二阶）之间的优化方法，其原理详见[[牛顿法|二阶方法-牛顿法]]

参数的拟牛顿法更新方式如下：
$$
\theta_{k+1} \leftarrow \theta_{k} - \alpha_kB_k^{-1}\nabla J(\theta_k)
$$
其中$B_k^{-1}$是$\theta_k$的Hessian矩阵近似。矩阵近似有很多种计算方法，此处介绍BFGS拟牛顿法。

$$
H_{k+1} \leftarrow H_k + \frac{ss^T}{s^Ty} - \frac{H_k yy^T H_k}{y^T H_k y}
$$
其中，$s=\theta_{k+1} - \theta_k$，详细推导过程略（因为我懒得学了好麻烦）
该方法可以避免求二阶段导导致计算复杂度进一步增大

Hessian矩阵近似可以通过如下的方式初始化：
- ※※※ 单位矩阵初始化  
- 以$\beta$倍缩放后的单位矩阵初始化
- 根据特定任务自定义初始化

### 2.4 带正则化项的逻辑回归函数
> 这个例子在[[强凸性与光滑性#1.4.2逻辑回归(Logistic Regression)+L2正则化|强凸性笔记]]有所介绍，具体的强凸性证明请参照对应章节

根据强凸性定义，在目标函数后增加二次函数，并加以变换，可以使得目标函数具有强凸性，变化后的形式如下：
$$
J(w) = \frac{1}{n}\sum_{i=1}^{n}log(1+e^{(-y_iw^Tx_i)}) + \frac{\lambda}{2}\|w^2\|
$$
此时目标函数是  $\lambda$-强凸  的


## 3.算法实现
我们使用如下的方式获取用于测试的分类数据：
```python

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 生成分类数据
X, y = make_classification(n_samples=114514, n_features=5, n_class=2, n_informative=2, random_state=1919810, n_clusters_per_class=1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=.2, random_state=1919810
)

```

### 3.1 傻瓜式实现
> 实际应用中就这样实现，**不要自己搓轮子了**


**sklearn调用`LogisticRegression`实现算法**
> sklearn是相当经典的行运算库，适合研究，实现idea等小规模数据分析

```python
from sklearn.linear_model import LogisticRegression  # 直接引入逻辑回归方法
import numpy as np

model = LogisticRegression(
	penalty='None',  # 无正则化，和上文的公式一致，设置为L2可将目标变得强凸
	# C=1.0,   # 只有在启动正则化后才有效，数字越小正则化强度越大
	max_iter=100,  # 最大迭代次数
	solver='saga',  # 梯度下降求解器，模型默认实际上是L-BFGS，更高性能的BFGS法
	random_state=1919810  # 随机种子
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # 预测，输出预测标签值数组
y_prob_pred = model.predict_proba(X_test)  # 输出预测标签概率数组
```

注意：参数$C$实际上是强突定义中$\lambda$的倒数，即有：
$$
C = \frac{1}{\lambda}
$$
$C$越小，目标函数的强凸性越大，强凸中的$\frac{1}{2}$不会参与到倒数运算中，这一点会在[[逻辑回归#3.2 逐步实现(本地Python，无高级api)|逐步实现]]中得到展示



**PySpark调用`LogisticRegression`**
> PySpark是Spark的Python接口，其以列为操作单位，适合海量数据处理，不适合用于构建更加精密复杂、带有创新的分类器。编写PySpark代码时应当注重对数据的批量操作(整个DataFrame或列)

> 推荐使用PySpark3.0以上版本


假设有如下格式的DataFrame变量 `traindata`和`testdata`

| x_vec: Vector   | y_vec: Vector |
| --------------- | ------------- |
| [1, 1, 1, 1, 1] | 1             |
| [0, 1, 0, 1, 1] | 2             |
| ...             | ...           |

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder \  # 创建Spark会话以与集群资源/本地资源互动
		.appName("LRDemo") \
		.getOrGreate()
		
# 这是一个Estimator子类，能定义模型参数并通过.fit生成带训练参数的Transformer子类
lr = LogisticRegression(
	featuresCol="x_vec",  # 选中特征列
	labelCol="y_vec",  # 选中标签列
	maxIter=100,  # 最大迭代次数
	family="binomial",  # 问题是二分类
	regParam=1  # 正则化强度，作用等同于sklearn的C

)  

# 这是一个Transformer子类，它只能根据Estimator子类给出的训练参数进行推理
model = lr.fit(train_data)  # PySpark会生成新的变量，而非像sklearn那样修改原有变量

predictions = model.transform(test_data)  # 模型推理
# columns = [x_vec, y_vec, prediction]

```
### 3.2 逐步实现(本地Python，无高级api)

**目标函数（损失函数）计算**
我们只需要实现单个样本的计算方式然后求均值即可，其公式为：
$$
y_ilog(h_{\theta}(x_i)))+(1-y_i)log(1-h_{\theta}(x_i))
$$

```python

import numpy as np

# Sigmoid
def sigmoid(z):
	"""
	该函数用于实现h(x)，即Sigmoid函数

	:param z: Sigmoid函数输入值，可以是标量或向量
	:return : Sigmoid函数输出值
	"""
	# np.where有三个参数，[判断条件/布尔值，条件为真的值，条件为假的值]
	return np.where(z >= 0, 
					1 / (1+np.exp(-z)),  # 大于0时使用标准形式 
					np.exp(z)/(1+np.exp(z)))  # 当小于0时使用等级啊变换避免溢出

def logistic_loss(theta, X, y):
	"""
	计算标准的对数损失
	:param theta: 算法参数，应当为数组，尺寸为(特征数, )，这也是我们要更新的部分
	:param X: 数据的特征值，应当为数组，尺寸为(样本数, 特征数)
	:param y: 数据的标签值，应当为数组，尺寸为(样本数)，值必须为0或者1
	:return loss: 逻辑(二元交叉熵/对数)损失值
	"""


	z = X @ theta  # 运算线性组合
	h = sigmoid(z)
	epsilon=1e-15  # 极小数，避免log0
	h = np.clip(h, epsilon, 1-epsilon)  # 裁剪运算结果
	loss = -np.mean(
		y * np.log(h) + (1 - y) *np.log(1 - h)
	)
	return loss

```

带正则项的目标函数

```python
def logistic_loss_reg(theta, X, y, lambda_reg):
	"""
	计算带正则化项的损失
	:param theta: 算法参数
	:param X: 数据特征值
	:param y: 数据标签值
	:param lambda_reg: 正则化强度，直接决定目标函数强凸程度，函数变化为 lambda_reg-强凸函数，取0则不进行正则化
	:return : 带正则化项的对数损失

	"""
	base_loss = logistic_loss(theta, X, y)
	reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)  # 正则化需要去除截距项，后同
	return base_loss + reg_term
	

```

计算梯度
```python
def get_grad(theta, X, y):

	n_samples = len(y)
	z = X @ theta
	h = sigmoid(z)
	gradient = (1/n_samples) * X.T @ (h-y)  # 直接套用公式
	return gradient

def get_grad_reg(theta, X, y, lambda_reg):

	n_samples = len(y)
	gradient = get_grad(theta, X, y)
	reg_gradient[1:] = (lambda_reg / m) * theta[1:]
	return gradient + reg_gradient


```

梯度下降更新
```python
def gradient_descent(X, y, theta_init=None, learning_rate=1e-2, n_iter=100, lambda_reg=0):
	"""

	"""

	m, n = X.shape
	if theta_init is None:
		theta = np.zeros()
	else:
		theta = theta_init.copy()

	loss_values = []

	for i in range(n_iter):
		if lambda_reg > 0:
			loss = logistic_loss_reg(theta, X, y, lambda_reg)
			grad = get_grad_reg(theta, X, y, lambda_reg)
		else:
			loss = logistic_loss(theta, X, y)
			grad = get_grad(theta, X, y)
		loss_values.append(loss)
		theta = theta - learning_rate * grad
	return theta, loss_values
```

所有代码的类实现
```python

class LogisticRegression:
	def __init__(
		self,
		learning_rate=1e-2,
		n_iter=100,
		lambda_reg=0,
		fit_intercept=True
	):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.lambda_reg = lambda_reg
		self.fit_intercept = fit_intercept

		self.theta = None
		self.loss_values = None

	def _add_intercept(self, X):
		m = X.shape[0]
		return np.c_[np.ones(m), X]

	def fit(self, X, y):
		if self.fit_intersept:
			X = self._add_intercept(X)

		self.theta, self.loss_values = gradient_descent(
			X,
			y,
			learning_rate,
			n_iter,
			lambda_reg
		)
		return self

	def predict_proba(self, X):

		if self.theta is None:
			raise ValueError("No parameters fitted yet!")

		if self.fit_intercept:
			X = self._add_inetercept(X)

		return sigmoid(X @ self.theta)

	

		

```