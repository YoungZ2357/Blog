---
title: 线性回归
draft: false
math: true
created: "2026-01-20T10:46"
updated: "2026-01-22T00:59"
---



> 该笔记参考自如下内容：1) 教授Olga Vitek的课堂讲义；2)Pattern Recognition and Machine Learning - Christopher M. Bishop

## 1 预备知识
#### 1.1 概率分布和样本

统计机器学习中，我们通常划分两个层面来讨论方法：
- 概率分布 层面
- 样本/数据 层面

![](/images/lnr0.png)

“概率分布”是一个我们假设存在的一个抽象的分布，其描述了随机变量的行为规律。而样本/数据，则是“概率分布”的一次**采样**，或者“**生成**”结果。例如：

> 在鸢尾花数据集中，假设我们希望判定一个花朵是不是 $ \texttt{virginica} $ ，我们可以抽象地认为，每一朵花的特征向量 $ X $ 都是从某个潜在概率分布 $ p(X|Y=\texttt{virginica}) $ 中采样得到的，或者是从中生成的。

- 如果我们有具体的数据 $ \{(x_1, y_1), (x_2, y_2), \cdots\} $ ，我们可以使用这些数据来推断概率分布所描述的“规则”，得到一个对概率分布的**参数估计**
- 如果我们有概率分布本身，我们就可以利用它抽样或者生成更多同类型数据，也可以计算某个样本**属于该分布的概率**（这一点可以在数据为人为地、根据某种规则生成时达到）

需要注意的是，通过**推断**，我们只能得到**参数估计**，而永远无法获得**概率分布本身**。
#### 1.2 模型的类别
若以概率论角度来理解监督学习，那么我们的最终目标就是得到一个良好的条件概率分布 $ p(y|x) $ ，即在已有随机变量 $ x $ 的前提下，得到 $ y $ 的概率。

我们通常可以直接建模 $ p(y|x) $ ，但通过贝叶斯定理，我们还可以通过更底层的信息来推导出我们的目标。

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$

接下来将详细介绍两种建模思路
##### 1.2.1 生成式模型：先建模世界，再推导决策

生成式模型的目标是建模 $ p(x|y)p(y) $ ，并通过贝叶斯定理反推出目标 $ p(y|x) $ 。正如***1.1***介绍，我们通常认为实际数据是从某个潜在概率分布中采样或者生成得到的。该种思路**建模了潜在概率分布**，故而这种模型也叫做生成式模型。

典型例子：
- 朴素贝叶斯
- 线性判别分析

##### 1.2.2 判别式模型：直接建模目标
判别式模型直接通过某种方式建模 $ p(y|x) $ ，而并不关注潜在分布。

典型例子：
- 线性回归
- 树和集成树
- 支持向量机

## 2 建模

#### 2.1 一元线性回归模型
线性回归构建了一个**对参数线性**的模型，其形式简单且拥有诸多便于求解的假设和性质。一元线性回归则是只有一个特征的线性回归。

一元线性回归的模型有如下形式：

$$
y = \theta_0 + \theta_1 x + \epsilon
$$

| 符号         | 含义  | 举例                   |
| ---------- | --- | -------------------- |
| $ y $        | 观测值 | 某个具体的房价              |
| $ \theta_0 $ | 截距  | 当 $ x=0 $ 时得到的一个“基础价格”   |
| $ \theta_1 $ | 斜率  | 自变量增加一单位后，价格的变化量     |
| $ \epsilon $ | 误差  | 模型无法解释的部分，完美的线性关系不存在 |

如下为一组噪声和一个线性回归的拟合线。点和模型拟合线之间的距离为 $ \epsilon $ ，我们通常假设其服从一个**均值为0，方差未知的正态分布，且误差之间相互独立**。 $ \epsilon $ 的拆分和处理将于 **模型的误差** 一节介绍

| ![](/images/lnr1.png) | ![](/images/lnr2.png) |
| ------------- | ------------- |

线性回归对 $ \epsilon $ 的假设非常重要，我们将于 **心智模型** 这一部分详细介绍其影响与应用
#### 2.2 多元线性回归模型及其拓展
当存在多个特征时，模型的形式变化为：

$$
y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_M x_M + \epsilon
$$

为了简化后续推导，我们将其描述成矩阵的形式：

$$
\begin{align}
& \mathbf{x} = (1, x_1, x_2, \cdots, x_M)^T  \\\\
& \pmb{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_M)^T
\end{align}
$$

此时模型变化为：

$$
y = \pmb\theta^T\mathbf{x} + \epsilon
$$

由此可见，线性回归本质是参数向量与特征向量的内积。

##### 2.2.1 模型的“线性”
需要再次强调的是，线性回归是指模型对参数是线性的（**定义**），例如：
- $ M_1: \ \ y= \theta_0 + \theta_1 x + \theta_2 x^2 $ 是线性回归
- $ M_2: \ \ y = \theta_0 + \theta_1^3 x $ 不是线性回归
当符合这种“线性”的特征时，线性回归的如下性质可以被直接挪用到对应的模型：

- 最小二乘法用于估计参数，线性代数方法批量求解
- 假设检验和置信区间：t检验，F检验和 $ R^2 $ 都可以直接适用，这意味着我们可以获取回归结果的95%把握区间，或者推断某个样本是否异常。
- 完全适用的正则化方法：如LASSO，Ridge，ElasticNet

我们也可以从**最优化**的角度来**判别**一个模型是否符合线性回归的定义

假设我们通过优化平方损失来更新这两个模型，我们会得到如下形式的损失函数计算：
> 注：更严格的相关原理将于稍后介绍

对模型1：

$$
\min_{\theta_0, \theta_1, \theta_2} \sum_{i=1}^N (y_i - \theta_0 - \theta_1 x_i - \theta_2 x_i^2)^2
$$

对模型2：

$$
\min_{\theta_0, \theta_1} \sum_{i=1}^N (y_i - \theta_0 - \theta_1^2 x_i)^2
$$

此时 $ x_i $ 和 $ y_i $ 都是具体的观测值，我们的目标是调整参数，使得目标函数最小化。若通过闭式解获取最优参数，我们需要**对参数求偏导**以寻找**驻点**，例如：

对模型1的损失求偏导

$$
\frac{\partial}{\partial\theta_1}\sum_{i=1}^{n}(y_i-\theta_0-\theta_1x_i-\theta_2x_i^2)^2 = 2\sum_{i=1}^n (y_i - \theta_0 - \theta_1 x_i)(-x_i)
$$

$$
\frac{\partial}{\partial\theta_0}\sum_{i=1}^{n}(y_i-\theta_0-\theta_1x_i-\theta_2x_i^2)^2 = 2\sum_{i=1}^n (y_i - \theta_0 - \theta_1 x_i)(-1)
$$

令二者为0时可以得到一个线性方程组，可通过线性代数方法求得驻点

对模型2的损失求偏导：

$$
\frac{\partial}{\partial\theta_1}\sum_{i=1}^{n}(y_i - \theta_0 - \theta_1^2 x_i)^2 = 
2\sum_{i=1}^n (y_i - \theta_0 - \theta_1^2 x_i)(-2\theta_1 x_i)
$$

令其为0会得到一个二次等式，表明其不是线性回归。这种求解困难会丢失很多良好性质，如：
- 无法批量求解，需要配方等代数技巧，难以自动化
- 可能无解（无法求闭式解）

这种形态的模型将依赖于如梯度下降等的启发式求解法。

##### 2.2.2 带基函数的线性回归 - 非线性表达能力的来源
基于 2.2.1 中描述的对参数的线性保障，我们可以将"x"改为非线性函数来增强模型的表达能力。Bishop[^1]的书中描述了一种使用基函数的表达方法。

该种表达不直接对原始特征 $ x $ 建模，而是先通过基函数 $ \phi (\cdot) $ 进行变换，再对变换进行线性组合。其形式为：

$$
y = \theta_0\phi_0(x) + \theta_1\phi_1(x) + \theta_2\phi_2(x) + \cdots + \theta_M\phi_M(x) + \epsilon
$$

我们重新定义特征向量为：

$$
\phi(\mathbf{x}) = \begin{bmatrix}
\phi_0(\mathbf{x}), \phi_1(\mathbf{x}), \cdots, \phi_M(\mathbf{x})^T
\end{bmatrix}
$$

模型可以简写为：

$$
y = \pmb \theta^T \phi(\mathbf{x}) + \epsilon
$$

当有 $ N $ 个样本时，可以用如下形式表达特征矩阵

$$
\Phi =
\begin{bmatrix}
\phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_M(\mathbf{x}_1) \\\\
\phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \cdots & \phi_M(\mathbf{x}_2) \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \cdots & \phi_M(\mathbf{x}_N)
\end{bmatrix}
\in \mathbb{R}^{N \times (M+1)}
$$

特征 $ x $ 可以被改造成任意的表达式，意味着 **不论我们如何变化特征的处理方法，线性回归的性质都不会受到影响**，我们可以借此使得线性回归模型变成任何我们想要的函数形式。以下是一些具体的例子：

**应用1：复杂的特征工程**
对于有周期性的数据，我们通常使用三角函数进行变换，以获取其时序特征。假如我们需要处理每日气温，我们可以定义**傅里叶基函数**：

$$
\begin{align}
& \phi_0(x)=1 \ \ \texttt{this is a dummy} & \\\\
& \phi_1(x) = \sin\left(\frac{2\pi x}{365}\right) \ \ \text{年周期正弦}& \\\\
& \phi_2(x) = \cos\left(\frac{2\pi x}{365}\right) \ \ \text{年周期余弦}& \\\\
& \phi_3(x) = \sin\left(\frac{4\pi x}{365}\right) \ \ \text{半年周期正弦}& \\\\
& \phi_4(x) = \cos\left(\frac{4\pi x}{365}\right) \ \ \text{半年周期余弦}& 
\end{align}
$$

模型形式变化为：

$$
y = \theta_0 + \theta_1 \sin\left(\frac{2\pi x}{365}\right) + \theta_2\cos\left(\frac{2\pi x}{365}\right) + \theta_3\sin\left(\frac{4\pi x}{365}\right) + \theta_4 \cos\left(\frac{4\pi x}{365}\right)
$$

**应用2：多项式回归**
要拟合一个曲线曲线，我们可以将基函数改变为幂函数，使得模型整体成为一个多项式

修改基函数如下：

$$
\begin{align}
& \phi_0(x)=1 \ \ \texttt{this is a dummy} & \\\\
& \phi_1(x)=x \ \ \text{一次项} & \\\\
& \phi_2(x)=x^2 \ \ \text{二次项} & \\\\
& \phi_3(x)=x^3 \ \ \text{三次项} & \\\\
\end{align}
$$

模型形式变化为：

$$
y = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3
$$

现在模型可以拟合一个三次函数曲线，而不仅仅是直线。我们可以增加更多的幂函数来强化模型的表达能力，但这种做法存在**方差-偏差均衡**和过**拟合等**问题，我们将在稍后讨论。

**应用3：SINDy[^2]**
SINDy希望通过给定的动力系统观测数据 $ \{(\mathbf{x}_i, \mathbf{\dot{x_i}})\} $ 来找到支配系统演化的方程 $ \mathbf{\dot{x}} = f(x) $ 。SINDy同时假设 $ f(\mathbf{x}) $ 可以表达为候选函数的稀疏线性组合。

当SINDy使用线性回归作为**求解器**时（具体为LASSO），我们可以将制定候选函数视作改变线性回归的基函数：

我们可以定义如下的候选函数集：

$$
\Theta(\mathbf{x}) =
\left[
1,\;
x_1,\;
x_2,\;
x_1^2,\;
x_1 x_2,\;
x_2^2,\;
\sin(x_1),\;
e^{x_2},
\;\ldots
\right]
$$

此时模型变化为：

$$
\dot{x}_k =
\xi_{k,0} \cdot 1
+
\xi_{k,1} \cdot x_1
+
\xi_{k,2} \cdot x_2
+
\xi_{k,3} \cdot x_1^2
+
\cdots
$$

它也拥有线性回归模型的性质

> 注：理论上来讲，任何稀疏回归算法都可以被SINDy使用，LASSO是最常用的一个。LASSO是对损失应用了L1惩罚项的线性回归，其在零点不可微分的特性使得其能够彻底抹除低相关性特征的权重（具体原因参考次梯度定义），使得对应特征完全无法影响预测行为，达到筛选特征的效果。这也意味着SINDy损失的优化过程不同于一般的线性回归。

#### 2.3 模型的误差
我们认为模型的误差 $ \epsilon $ 来自于两个方面：
- **系统误差**：由模型本身导致的误差，该误差来自于参数估计（即参数和实际的潜在分布参数的差异）。这种误差是我们需要优化的部分
- **随机误差**：数据自带的噪声。这种误差是不可约的，其方差是我们能做到的最低的误差水平，该噪声相关的假设对我们后续的分析至关重要。

实操过程中，可以通过这两个角度来调整分析范围和特征工程。

以房屋售卖价格为例，假如我们分析了房屋面积 $ a $ ，单位面积物业费 $ p $ ，最近交通枢纽距离 $ d_1 $ ，并以这三个变量来预测房屋价格 $ y $ ，我们可能因为如下原因产生系统误差：
- 遗漏变量：任何可能影响房屋价格，但我们没有分析的变量，例如最近大型商场距离 $ d_2 $ ，租户-业主比 $ r $ ，等等
- 使用“错误“的函数形式：例如 $ y $ 有可能和 $ a $ 的关系呈现二次曲线，但建模时使用了一次函数，直线再怎么接近观测值，也无法拟合一个曲线。

我们可能因为如下原因产生随机误差：
- 微观随机性：真正不可控的随机因素。例如，也许业主心情好，看买家顺眼，就能把价格砍下来
- 数据来源：获得的观测值受到精度、人为错误等原因无法体现潜在分布。例如：房屋面积四舍五入，故意少算公摊面积
- 某些现象本身的随机性

我们将于 **心智模型** 一章从分布的角度来具体分析误差。
## 3 心智模型
先前的章节中，我们通过直观建模的层面 - 或者说采样/数据的层面 - 介绍了线性回归，但某些更深层的分析，如置信区间的获取，难以通过这种层面的内容来解释。故而该章节将从**概率分布**的层面来重新介绍**一元线性回归**，介绍数据是如何”被生成“的，我们后续的参数估计也需要基于这一步展开。

#### 3.1 从观测值到随机变量
我们将模型改写成随机变量形式：

$$
Y = \theta_0 + \theta_1 X + \varepsilon
$$

此时 $ Y $ 是一个随机变量。现在，我们回归线性回归最关键的假设 - 误差呈正态分布，且独立同分布：

$$
\varepsilon_i \stackrel{iid}{\sim} N(0,\sigma^2), \quad i=1,\dots,N
$$

选择正态分布大致有如下两个原因
> 部分推导和公式超出该笔记范围，此处略过。但不论如何，以下三个原因的影响都非常大，不会因为看起来简单而被削弱。

- 中心极限定理：大量随机独立变量的和趋向于正态分布
- 最大熵原理：在满足已知约束的所有分布中，正态分布的熵最大，意味着它”最不需要额外假设“
- 正态分布的便捷性：两个正态分布的条件概率分布也是正态分布（读者可以尝试使用正态分布pdf、贝叶斯公式和联合正态分布定义完成推导）

如果我们给定随机变量 $ X $ （例如用X预测Y），此时条件概率分布 $ Y|X $ 的随机性将完全来自于误差 $ \varepsilon $ ，它的分布是：

$$
Y|X \sim \mathcal{N}(\theta_0+\theta_1X, \sigma^2)
$$

相当于对 $ \varepsilon $ 的分布作了偏移。该分布会在后续的偏差分析中起到重要作用。

#### 3.2 条件概率分布的几何理解
将分布 $ Y|X $ 可视化后，我们可以得到如下图案。由于 $ Y $ 和 $ X $ 之间的关系是**线性**的，故而联合分布的概率密度3D图呈现出一个**沿着回归线移动的钟形图案**，而非一个圆形或者椭圆形的曲面。
![](/images/lnr3.png)
当 $ Y $ 与 $ X $ 的线性关系变弱时（ $ Var(\frac{X}{\theta})\leq \sigma^2 $ ），想象回归线附近会出现很多散点，每一个散点都会对应一条正态分布的钟形曲线，进而钟形图案会形成堆叠。即使图像仍然沿回归线移动，但其在沿着回归线垂直的方向开始变“肥”，移动方向开始倾向于 $ y=\mu $ 。

当 $ Y $ 与 $ X $ 独立时， $ Y|X = Y $ ，曲面退化为 $ Y $ 的概率分布，这也是 $ Y $ 的**边缘概率分布**
#### 3.3 似然函数
##### 3.3.1 独立同分布假设
***3.1***中，我们已经假设误差独立同分布，根据独立性我们可以得到：

$$
P(\varepsilon_1, \varepsilon_1, \cdots, \varepsilon_N) = \prod \limits_{i=0}^N P(\varepsilon_i)
$$

又由于 $ Y_i = \theta_0 + \theta_1X_i + \varepsilon_i $ ，且 $ Y_i $ 的随机性完全来自于 $ \epsilon_i $ ，因此，在给定 $ X $ 时， $ Y $ 之间也相互独立

$$
P(Y_1, \cdots, Y_N|X_1, \cdots, X_N;\theta) = \prod \limits_{i=0}^N P(Y_i|X_i;\pmb\theta)
$$

##### 3.3.2 似然函数定义
有了独立性假设，我们可以进一步获取**似然函数**的定义。我们已知条件概率分布 $ Y_i|X_i \sim \mathcal{N}(\theta_0 + \theta_1 X_i, \sigma^2) $ ，它的概率密度函数为：

$$
p(y_i|x_i;\pmb\theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left\{          -\frac{(y_i-\theta_0-\theta_1 x_i)^2}{2\sigma^2}    \right\}
$$

结合独立性假设，我们可以得到整个数据集的联合概率密度：

$$
\begin{align}
p(y_1, \cdots, y_N|x_1, \cdots, x_N;\pmb\theta) &= \prod \limits_{i=0}^Np(y_i|x_i;\theta) \\\\
&=\prod \limits_{i=0}^N \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left\{          -\frac{(y_i-\theta_0-\theta_1 x_i)^2}{2\sigma^2}    \right\} \\\\
&=\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^N \cdot \exp \left\{
-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-\theta_0-\theta_1x_i)^2
\right\}
\end{align}
$$

**似然函数是切换了视角后的联合概率密度**，即：

$$
\mathcal{L}(\pmb\theta, y) = \prod \limits_{i=0}^N \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left\{          -\frac{(y_i-\theta_0-\theta_1 x_i)^2}{2\sigma^2}    \right\}
$$

> 没错，这个就是似然函数的定义了，啥都没有，就一个公式

似然函数的意义是：在给定观测数据 $ y $ 的情况下，模型参数 $ \pmb\theta $ 出现的概率。

| 对象  | 联合概率密度                         | 似然函数                  |
| --- | ------------------------------ | --------------------- |
| 已知  | 模型参数                           | 观测值                   |
| 未知  | 观测值                            | 模型参数                  |
| 含义  | 在确定模型参数和与目标有关系的观测值时，目标观测值出现的概率 | 在已有目标观测值时，某个模型参数出现的概率 |

让我们再次回到***3.2***中的钟形3D图中：
![](/images/lnr4.png)
我们可以发现每一个观测点 $ (x_i, y_i) $ 在曲面上都可以对应一个高度，这个高度既是**已知观测点时回归线参数对应的似然**，也是**已知回归线参数和 $ x $ 观测值时，对应 $ y $ 观测值出现的概率**。

#### 心智模型总结
通过先前所有内容，我们建立了对线性回归的两个理解角度：

**第一层：线性的判别式模型**
- 核心认知：定义模型**对参数是线性的**，误差独立同分布，可以通过修改基函数来得到非线性表达能力
- 关键公式： $ y = \pmb\theta^T\mathbf{x} + \epsilon $ 或者 $ y = \pmb \theta^T \phi(\mathbf{x}) + \epsilon $

**第二层：条件概率分布**
- 核心认知：给定 $ X $ 时， $ Y $ 是一个随机性完全取决于误差的、服从正态分布的随机变量， $ X $ 和 $ Y $ 分别都**独立同分布**
- 关键公式： $ Y|X \sim \mathcal{N}(\theta_0+\theta_1X, \sigma^2) $

除此之外，我们可以通过联合概率密度和似然函数，来实现**参数**和**观测值**之间的**推断**
## 4 参数估计
#### 4.1 最小二乘估计
##### 4.1.1 一元线性回归
二乘，即残差的平方。在一元线性回归中它可以被如下定义

$$
e_i^2 = (y_i-\hat{y}_i)^2 = (y_i-\theta_0-\theta_1 x_i)^2
$$

最小二乘法的目标是缩小总体的残差平方，其目标可以如此表示：

$$
\min_{\theta_0, \theta_1}\text{RSS} = \min_{\theta_0, \theta_1}\sum_{i=1}^N(y_i-\theta_0 - \theta_1 x_i)^2
$$

对于一元线性回归，我们通常直接求闭式解，即尝试寻找驻点。

$$
\begin{align}
&\frac{\partial \text{RSS}}{\partial \theta_0} = -2\sum_{i=1}^{N}(y_i - \theta_0 - \theta_1 x_i)=0& \\\\
&\frac{\partial \text{RSS}}{\partial \theta_1} = -2\sum_{i=1}^{N}(y_i - \theta_0 - \theta_1 x_i)x_i=0&
\end{align}
$$

该方程组的求解结果为：

$$
\begin{align}
&\hat{\theta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}& \\\\
&\hat{\theta}_0 = \bar{y} - \hat{\theta}_1\bar{x}&
\end{align}
$$

该求解结果也表明，回归线一定会经过点 $ (\bar{x}, \bar{y}) $

##### 4.1.2 多元线性回归
我们用矩阵形式来表达多元线性回归的估计过程。

定义矩阵 $ \mathbf{X} \in \mathbb{R}^{N\times (M+1)} $ ，观测值 $ \mathbf{y} \in \mathbb{R}^N $
目标函数改写为：

$$
\min_{\boldsymbol{\theta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 = \min_{\boldsymbol{\theta}} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
$$

将其展开可以得到如下形式：

$$
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 = \mathbf{y}^T\mathbf{y} - 2 \pmb\theta^T\mathbf{X}^T\mathbf{y}+\pmb\theta^T\mathbf{X}^T\mathbf{X}\pmb\theta
$$

对 $ \pmb \theta $ 求导并令其等于0得到：

$$
\begin{align}
\frac{\partial \text{RSS}}{\partial \boldsymbol{\theta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} &= 0 \\\\
   2\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} &= 2\mathbf{X}^T\mathbf{y}\\\\
\end{align}
$$

乘以逆矩阵后可以得到 $ \pmb\theta $ 的估计：

$$
\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

其中 $ \mathbf{X}^T\mathbf{X} $ 就是黑森矩阵。由于目标函数是凸的，我们可以确定这个矩阵是半正定的

需要注意的是，当矩阵的维度过高时，求闭式解会有极大的运算负担：
- 计算 $ \mathbf{X}^T \mathbf{X} $ : $ O(NM^2) $
- 计算 $ \mathbf{X}^T \mathbf{y} $ ： $ O(NM) $
- 计算 $ (\mathbf{X}^T \mathbf{X})^{-1} $ ： $ O(M^3) $
- 矩阵乘法： $ O(M^2) $
总体时间复杂度： $ O(NM^2+M^3) $
此时推荐使用矩阵分解或者改用启发式算法。

#### 4.2 极大似然估计
***3.3.2***中我们给出的似然函数的定义：

$$
\mathcal{L}(\pmb\theta, y) = \prod \limits_{i=0}^N \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left\{          -\frac{(y_i-\theta_0-\theta_1 x_i)^2}{2\sigma^2}    \right\}
$$

现在我们的目标是最大化似然函数，即找到根据已有观测点，最可能出现的参数值。在可视化中，具体呈现为：使回归线上的点，对应的高度最高：

$$
\max_{\pmb\theta}\mathcal{L}(\pmb\theta, y)
$$

通常我们可以取负对数来使得这个优化问题更简单。对数的底通常是自然常数

$$
\begin{align}
-\ln\mathcal{L}(\pmb\theta, y) &= + (\frac{N}{2}\ln(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-\theta_0-\theta_1x_i)^2)
\\\\
&=\frac{N}{2}\ln(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-\theta_0-\theta_1x_i)^2
\end{align}
$$

问题转换为

$$
\min_{\pmb\theta} -\ln\mathcal{L}(\pmb\theta, y) = \min_{\pmb\theta} \left(\frac{N}{2}\ln(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-\theta_0-\theta_1x_i)^2\right)
$$

此时唯一可以调整的量为 $ (y_i-\theta_0-\theta_1x_i)^2 $ ，容易发现：线性回归中，极大似然估计等效于最小二乘估计，求闭式解的方式与***4.1***完全相同

#### 4.3 启发式方法：梯度下降算法
> 直线搜索梯度下降法详见[梯度下降及其收敛性](/posts/梯度下降及其收敛性/)，此处仅作简单介绍

如果将闭式解理解成“一步优化”，梯度下降法则是“多步优化”。梯度下降法分为两种：
- **带直线搜索的梯度下降法**：每一次更新都是一次**优化问题**，通过某种**直线搜索**（更严谨来说，射线搜索） 方法来获取满足停止准则的步长，然后让函数沿梯度根据步长下降，直线搜索可以是回溯直线搜索，也可以是精确直线搜索
- **固定步长梯度下降**：**每一次更新的工作量仅为计算梯度**，直接按照固定步长朝着每一步的梯度方向进行下降

梯度下降的一般形式为：

$$
\pmb\theta^{(k+1)} \gets \pmb\theta^{(k)} - \alpha^{(k)}\nabla\text{RSS}(\pmb\theta^{(k)})
$$

- $ \nabla\text{RSS}(\pmb\theta^{(k)}) $ - 残差平方和的梯度
- $ \alpha^{(k)} $ - 第 $ k $ 步的步长，根据具体的实现方法更改

## 5 估计的性质与推断
该章节将介绍线性回归中，通过最小二乘估计或者极大似然估计得到的参数的特征，和一些结果检验技巧。
#### 5.1 无偏性
如果一个估计量 $ \boldsymbol{\hat{\theta}} $ 的期望等于真实参数 $ \boldsymbol{\theta} $ ，则可以称该估计量是无偏的：

$$
E[\boldsymbol{\hat{\theta}}] = \boldsymbol{\theta}
$$

我们以OLS的估计为例： $ \boldsymbol{\hat{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y} $
代入数据生成过程：

$$
\begin{align}
&\mathbf{y}= \mathbf{X}\boldsymbol{\theta}+\varepsilon \\\\
& \boldsymbol{\hat{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T ((\mathbf{X}\boldsymbol{\theta}+\varepsilon))
\end{align}
$$

展开并化简：

$$
\begin{align}
\boldsymbol{\hat{\theta}} &= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T 
\mathbf{X} \boldsymbol{\theta} + (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \varepsilon \\\\
&=\boxed{\boldsymbol{\theta} + (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \varepsilon}
\end{align}
$$

估计参数是原始参数的线性组合。由于参数本身是基于条件概率定义的（给定 $ X $ ），此处我们取以 $ X $ 为条件的条件期望：

$$
E[\boldsymbol{\hat{\theta}}|\mathbf{X}] = E[\boldsymbol{\theta}|\mathbf{X}] + 
E[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \varepsilon|\mathbf{X}]
$$

对于第一项，由于 $ \boldsymbol{\theta} $ 是固定值（实际参数值），故而有：

$$
E[\boldsymbol{\theta}|\mathbf{X}] = \boldsymbol{\theta}
$$

对于第二项：
首先，根据 ***3.1***中对分布的描述： $ Y|X \sim \mathcal{N}(\theta_0+\theta_1X, \sigma^2) $ ，结合残差的定义： $ \varepsilon = Y-(\theta_0+\theta_1 X) $ ，我们可以得到：

$$
\begin{align}
E[\varepsilon|X] &= E[Y-(\theta_0+\theta_1 X)|X] \\\\
&=E[Y|X] - (\theta_0 + \theta_1 X) \\\\
&= (\theta_0 + \theta_1 X) - (\theta_0 + \theta_1 X) \\\\
& =0
\end{align}
$$

由于 $ \mathbf{X} $ 是条件的一部分，我们可以把所有相关项视为常数，并将其移动到期望外：

$$
\begin{align}
E[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \varepsilon|\mathbf{X}] &=
(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^TE[\varepsilon|\mathbf{X}]\\\\
 &= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \cdot 0 \\\\
 &= 0
\end{align}
$$

由此，我们可以得到：

$$
\boxed{\boldsymbol{\hat{\theta}}  = \boldsymbol{\theta}+0 = \boldsymbol{\theta}}
$$

证毕，最小二乘法对参数的估计是无偏的。

%% #### 5.2 方差与Gauss-Markov定理

#### 5.3 偏差-方差均衡问题

#### 5.4 假设检验与置信区间 %%

## 参考文献

[^1]: Bishop, Christopher M., and Nasser M. Nasrabadi. _Pattern recognition and machine learning_. Vol. 4. No. 4. New York: springer, 2006.

[^2]: Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." _Proceedings of the national academy of sciences_ 113.15 (2016): 3932-3937.
