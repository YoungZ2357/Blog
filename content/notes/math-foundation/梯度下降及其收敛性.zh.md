---
title: "梯度下降及其收敛性"
date: 2025-12-11
draft: false
math: true
tags: ["凸优化", "下降方法", "优化理论"]
categories: ["学习笔记"]
---

## 1. 通用下降方法

### 1.1 下降方法基本形式

> 注：下降方法不会要求凸性，但拥有凸性会让求解得到显著保障。对凸性的定义见[[凸集与凸函数深入#2.1 凸函数定义|凸函数定义]]

我们先来回顾梯度下降的原型 - **下降方法**

下降算法会产生优化点列 $x^{(k)}, k=1, \cdots$，其中

$$
x^{(k+1)} = x^{(k)} + t^{(k)}\Delta x^{(k)}
$$

且 $t^{(k)} > 0$（除非 $x^{(k)}$ 已经最优了）。该概念参考自[[Convex Optimization.pdf|Convex Optimization]][^1]

此处 $\Delta x^{(k)}$ 是一个向量，称为**搜索方向**，$t^{(k)}\Delta x^{(k)}$ 称为**步径**（实践中常为和参数同尺寸的多维数组）；$k$ 代表迭代次数；标量 $t^{(k)}$ 是更新步长。

### 1.2 下降条件

对于**所有下降方法**，只要 $x^{(k)}$ 不是最优点，则都有：

$$
f(x^{(k+1)}) < f(x^{(k)})
$$

> 注意：下降算法默认为最小化目标函数，一切最初形式为"最大化目标函数"的问题(例如TD误差)均可被描述为如此形式

以下是下降方法的下降条件：

**凸函数**情况下，若选择 $d=y-x$ 方向更新，现考虑凸函数的[[凸集与凸函数深入#2.2.1 一阶条件|一阶条件]][^2] $f(y) \geq f(x)+\nabla f(x)^T(y-x)$，要使函数下降，即 $f(y) < f(x)$，**则必须有 $\nabla f(x)^{T}(y-x) < 0$。**

此处，$\nabla f(x)^{T}(y-x)$ 即为**方向导数**


| 特征   | 搜索方向            | 步径                          | 方向导数                         |
| ---- | --------------- | --------------------------- | ---------------------------- |
| 数学定义 | `$d^{(k)}$`     | `$\alpha^{(k)} d^{(k)}$`    | `$\nabla f(x)^{T}(y-x)$`     |
| 几何含义 | 移动的方向           | 实际的位移向量                     | 函数沿方向的变化率                    |

### 1.3 通用下降法算法框架 

通用下降方法通常有如下的步骤：

给定初始点 $x \in \mathbf{dom}\ f$ 

重复进行：
1. 确认下降方向 $\Delta x$
2. 直线搜索以**寻找步长**，选择步长 $t > 0$
3. 更新点 $x \leftarrow x+\alpha\Delta x$

直到 **满足停止准则**

停止准则见[[梯度下降及其收敛性#4.3 常用的停止准则|常用停止准则]]

## 2 直线搜索

> 直线搜索的使用，意味着通用下降法不是一个**固定步长方法**。且严格意义讲，该方法应当成为**射线搜索**，搜索域为 $t \in [0, +\infty)$

通用下降法中通常有以下两种直线搜索方法

### 2.1 精确直线搜索

精确直线搜索要求每次都采用有最大下降的步长，其沿着射线 $\{x+\alpha\Delta x \mid t\in\mathbf{R}_{+}\}$ 更新：

$$
\alpha = \arg\min_{s\geq 0} f(x+s\Delta x)
$$

需要注意，该方法会将每一步都视为一维优化问题，也就是每一次更新都需要调用一次优化方法（如黄金比例切割法），进而导致该方法在实际操作中运算量过大，故而此处不详细讨论

### 2.2 回溯直线搜索 

> 原本是该有图的，但我没完全学懂

回溯直线搜索是一种**非精确**方法。和精确直线搜索类似，它也是在射线 $\{x+\alpha\Delta x \mid t\in\mathbf{R}_{+}\}$ 进行更新，但只要求函数值**有足够的减少**即可。该方法主要有一个收缩参数 $\rho$，它用于控制每次下降的程度。其他参数则由具体的下降条件决定

1. 给定下降方向 $\Delta x$，参数 $\rho \in (0, 1)$
2. $\alpha\leftarrow 1$
3. 若 **满足下降条件**
   则 $\alpha\leftarrow \rho \alpha$

其中，下降条件可以是如下的任意一种：

**(1)Armijo搜索条件(充分下降条件，Sufficient Descent Condition)**

$$
f(x^k + \alpha_k d^k) \leq f(x^k) + \sigma_1 \alpha_k \nabla f(x^k)^T d^k
$$

其中：
- $x^k$ 为第 $k$ 次迭代当前点
- $d^k$ 为第 $k$ 次迭代搜索方向
- $\alpha^k$ 为第 $k$ 次迭代的步长
- $\sigma_1$ 为Armijo参数，控制充分下降程度

左侧为实际能够到达的函数值，右侧为原函数值和变化值的 $\sigma_1$ 倍

即，实际能到达位置应当小于等于原本函数值产生 $\sigma_1$ 倍变化后的值

```python
from autograd import grad
def line_search_armijo(f, x, delta_x, rho=.5, c=1e-4, max_iter=100) -> float:
	"""使用Armijo条件的回溯直线搜索
	
	:param f: 目标函数
	:param x: 当前点
	:param delta_x: 下降方向
	:param rho: 收缩因子
	:param c: Armijo常数，默认为1e-4, 取值范围通常为[1e-6, 1e-4]
	:param max_iter: 最大迭代次数
	:return alpha: 合适的步长
	"""
	alpha = 1.
	grad_f = grad(f)  # 对函数自动求导
	f_x = f(x)
	grad_x = grad_f(x)

	dir_deriv = np.dot(grad_x, delta_x)
	if dir_deriv >= 0:
		print("Directional vector is not a descending direction vector")
		return 0

	for i in range(max_iter):
		x_new = x + alpha * delta_x
		f_new = f(x_new)

		if f_new <= f_x + c * alpha * dir_deriv:
			return alpha
		alpha = rho * alpha
	print(f"Reached maximum iter number: {max_iter}")
	return alpha
```

函数应当如下定义。
以 $f(x) = (1-x_1)^2 + 100 \times (x_2 - x_1^2)^2$ 为例

```python
def rosenbrock(x: list):
	return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

![Armijo示意图](/images/math/cvx/descent1.png)

**(2)曲率条件**

$$
\nabla f(x^k + \alpha_k d^k)^T d^k \geq \sigma_2 \nabla f(x^k)^T d^k
$$

即，到达位置的梯度应当大于原位置的 $\sigma_2$ 倍

**(3)Wolfe条件(弱Wolfe条件)**

同时满足**Armijo条件**和**曲率条件**。如果使用该条件，算法将一共需要3个参数

**(4)强Wolfe条件**

同时满足**Armijo条件**和**强曲率条件**

其中强曲率条件：

$$
|\nabla f(x^k + \alpha_k d^k)^T d^k| \leq |\sigma_2 \nabla f(x^k)^T d^k|
$$

## 3 梯度下降法

梯度下降法通常有如下形式：

给定初始点 $x \in \mathbf{dom}\ f$ 

重复进行：
1. $\Delta x \leftarrow -\nabla f(x)$
2. 直线搜索以**寻找步长**，选择步长 $t > 0$
3. 更新点 $x \leftarrow x+\alpha\Delta x$

直到 **满足停止准则**

## 4 收敛性分析

### 4.1 基本假设

要分析收敛性，我们通常需要以下假设

#### 假设1：Lipschitz光滑梯度条件（光滑性）

> 详细介绍见[[强凸性与光滑性#2.1 梯度Lipschitz条件定义（标准定义/一阶条件）|光滑性的Lipschitz条件定义]]

这个条件通常是必须的。存在常数 $L > 0$，使得对所有 $x, y\in \mathbb{R}^n$ 有：

$$
\|\nabla f(x) - \nabla f(y)\| < L \|x-y\|
$$

#### 假设2 强凸性一阶条件定义

> 详细介绍见[[强凸性与光滑性#1.2 基于一阶条件的定义|强凸性一阶条件定义]]

这个条件不是必须的，仅用于收敛速度分析。存在常数 $\mu > 0$，使得对所有 $x, y \in \mathbb{R}^n$ 有：

$$
f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2
$$

### 4.2 收敛性定理

> PS：我打算直接用了，懒得推理了

#### 4.2.1 使用固定步长的收敛性

在**假设1**下，若使用固定步长 $\alpha \leq \frac{1}{L}$，则有：

$$
f(x_k) - f^* \leq \frac{\|x_0 - x^*\|^2}{2\alpha k}
$$

例：假设初始点距离最优点10个单位，即 $\|x_0 - x^*\| = 10$，步长固定为 $\alpha = 1e-2$

则误差上界为：

$$
f(x_k) - f^* \leq \frac{100}{2 \times 0.01 \times k} = \frac{5000}{k}
$$

要让误差 $\leq 0.1$，则需要 $k \geq 50000$ 步

要让误差 $\leq 0.01$，则需要 $k\geq 500000$ 步

这种下降方法具有 $O(\frac{1}{k})$ 的次线性收敛速度，收敛速度会逐渐减慢

#### 4.2.2 使用回溯直线搜索的收敛性

在**假设1**下，使用回溯直线搜索的梯度下降满足：

$$
f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|^2}{2k} = \frac{\|x_0 - x^*\|^2}{2(\frac{1}{L})k}
$$

该下降方法和固定步长方法同样具有 $O(\frac{1}{k})$ 的次线性收敛速度，但是容易受到 $L$ 值的线性影响

#### 4.2.3 强凸函数的收敛性

在**假设1**和**假设2**下，梯度下降具有**线性收敛**速度：

$$
f(x_k) - f^* \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f^*)
$$

其中收敛率 $\rho = 1 - \frac{\mu}{L} = 1 - \frac{1}{\kappa}$，$\kappa$ 是条件数

可见，强凸目标函数会受到病态问题的影响。条件数和病态相关的知识见[[强凸性与光滑性#3.2 条件数|条件数]]

具体的：

要使得误差降低为原本的 $\epsilon$ 倍，则需要的迭代次数为：$k = \frac{\log(\epsilon)}{\log(\rho)} = \frac{\log(\epsilon)}{\log(1/\kappa)}$

可以有如下的表格：

| 条件数 `$\kappa$` | 收敛率 `$\rho$` | 每步误差减少比例 `$\epsilon$` | 精度提高10倍所需迭代数 `$k$` |
| ------------- | ----------- | -------------------- | ----------------- |
| 2             | 0.5         | 50%                  | ~3                |
| 10            | 0.9         | 10%                  | ~22               |
| 100           | 0.99        | 1%                   | ~230              |
| 1000          | 0.999       | 0.1%                 | ~2300             |

### 4.3 收敛速度分类

收敛速度通常由以下两种定义方法：
- 函数值误差：$f(x_k) - f^*$，即目标函数值与最优函数值的差异，这也是最优化的**主要目标**
- 点距离误差：$\|x_k-x^*\|$，即当前点到最优点的距离

该部分符号含义：
- $f^*$：函数最优值
- $K$：某个有限迭代次数的阈值，从此点开始收敛性质成立
- $k$：真实迭代次数
- $p$：收敛速度指数
- $\rho$：收敛率，越小收敛越快
- $\epsilon$：精度参数，表示容许误差大小

所有收敛速度的对比图如下：

![收敛速度对比](/images/math/cvx/descent8.png)

#### 4.3.1 对数收敛(Logarithmic Convergence)

存在常数 $C > 0, K\in\mathbb{N}$，使得对所有的 $k\geq K$ 有：

$$
f(x_k) - f^* \leq \frac{C}{\log k}
$$

该收敛速度达到精度的迭代数为 $O(e^{1/\epsilon})$

例：假设 $C=1$，则有：

| 迭代次数       | 误差                              |
| ---------- | ------------------------------- |
| `$k=10$`   | `$< 1/\log(10) \approx 0.434$`  |
| `$k=100$`  | `$< 1/\log(100) \approx 0.217$` |
| `$k=1000$` | `$< 1/\log(1000) \approx 0.145$` |

![对数收敛](/images/math/cvx/descent5.png)

#### 4.3.2 次线性收敛(Sublinear Convergence)

存在常数 $C > 0, K\in \mathbb{N}$，使得对于所有的 $k\geq K$ 有：

$$
f(x_k) - f^* \leq \frac{C}{k^p}
$$

| 函数值收敛                            | 渐进行为 | 达到 `$\epsilon$` 迭代数     |
| -------------------------------- | ---- | ----------------------- |
| `$f(x_k) - f^* \leq C/\sqrt{k}$` | 慢    | `$O(1/\epsilon^{2})$`   |
| `$f(x_k) - f^* \leq C/k$`        | 中等   | `$O(1/\epsilon)$`       |
| `$f(x_k) - f^* \leq C/k^2$`      | 快    | `$O(1/\sqrt{\epsilon})$` |

![次线性收敛](/images/math/cvx/descent3.png)

#### 4.3.3 线性收敛(Linear Convergence)

存在常数 $C > 0, \rho \in(0, 1)$，使得：

$$
\|x_k - x^*\| < C\rho^k
$$

将其进行对数变换有：

$$
\log\|x_k - x^*\| \leq \log C + k\log \rho
$$

![线性收敛](/images/math/cvx/descent2.png)

#### 4.3.4 超线性收敛(Super Convergence)

![超线性收敛](/images/math/cvx/descent6.png)

#### 4.3.5 二次收敛(Quadratic Convergence)

![二次收敛](/images/math/cvx/descent7.png)

### 4.4 常用的停止准则

如下的准则会因为目标函数的非凸性和问题本身性质而不满足(或使用)先前假设/定理，但实际工程中仍然奏效，且推荐多方法堆叠使用。

#### 4.4.1 基于梯度的准则

> 当处在最优点 $x^*$ 时，有 $\nabla f(x^*)=0$

##### (1)绝对梯度准则

$$
|\nabla f(x_k)| < \epsilon_{abs}
$$

其中：
- $\epsilon_{abs}$ 是判断收敛的**绝对**阈值，是一个正小数
- $|\ |$ 是绝对值符号，而非范数符号，**此处不以矩阵表示，下同**

当梯度绝对值小于该 $\epsilon_{abs}$ 时，认为算法收敛

注意：
- 需要根据问题调整，无法自动适应不同问题

##### (2)相对梯度准则

$$
|\nabla f(x_k)| < \epsilon_{rel} \cdot |\nabla f(x_0)| 
$$

其中：
- $\epsilon_{rel}$ 是判断收敛的**相对**阈值，是一个正小数
- $x_0$ 是算法的起始点

当梯度下降到初始点梯度的 $\epsilon_{rel}$ 倍时，认为算法收敛。

> 例如：如果设置 $\epsilon_{rel}=0.01$，则在 $|\nabla f(x_k)|$ 是 $|\nabla f(x_0)|$ 的1%时才会判定算法收敛

注意：
- 可以适应不同问题
- 初始点敏感，$|\nabla f(x_0)|$ 过小或将导致过早停止

以下是针对初始点做出的一种改良：

$$
|\nabla f(x_k)| < \epsilon_{rel} \cdot \max(1, |\nabla f(x_0)|)
$$

#### 4.4.2 基于函数值的准则

##### (1)绝对函数值变化准则

$$
|f(x_k) - f(x_{k-1})| < \epsilon_f
$$

其中：
- $\epsilon_f$ 表示对函数值绝对变化的阈值

当函数值变化小于 $\epsilon_f$，判定算法收敛

注意：
- 无需梯度信息，更简单快速
- 平坦区域易过早停止

##### (2)相对函数值变化准则

$$
\frac{|f(x_k) - f(x_{k-1})|}{|f(x_{k-1})| + \epsilon_{mach}} < \epsilon_{f,rel}
$$

其中：
- $\epsilon_{mach}$ 是正小数，用于防止除零，通常为机器的精度
- $\epsilon_{f, rel}$ 是对函数值相对变化的阈值

当函数变化值与上一步函数值的比值小于阈值，判定算法收敛

#### 4.4.3 基于参数的准则

> 注意：此处的符号 $\theta$ 是指**模型的可训练参数**

##### (1)绝对参数变化

$$
|\theta_k - \theta_{k-1}| < \epsilon_{\theta}
$$

##### (2)相对参数变化

$$
\frac{|\theta_k - \theta_{k-1}|}{|\theta_{k-1}| + \epsilon_{mach}} < \epsilon_{x,rel}
$$

##### (3)参数差异的计算方法

通常情况下，参数 $\theta$ 会以**矩阵**的形式呈现，而 $\epsilon$ 却是一个**标量**，在工程中通常使用**展平+计算范数**的方法来将参数变为标量并计算变化。

以下示例代码使用了：
- 绝对参数变化
- 展平+二范数

```python
def param_method_abs(params_k: dict, params_k_1: dict, epsilon=1e-6) -> bool:  
    """
    :param params_k: 第k次迭代参数字典，如{'W1': matrix, 'b1': vector}  
    :param params_k_1: 第k+1次迭代参数字典，如{'W1': matrix, 'b1': vector}  
    :param epsilon: 收敛的绝对阈值  
    :return abs_change < epsilon: 是否满足绝对阈值收敛条件  
    """    
    flat_k = np.concatenate([p.flatten() for p in params_k.values()])  
    flat_k_1 = np.concatenate([p.flatten() for p in params_k_1.values()])  
    abs_change = np.linalg.norm(flat_k - flat_k_1)  
    return abs_change < epsilon
```

#### 4.4.4 基于资源限制的准则

##### (1)最大迭代次数

用于防止无限运行

$$
k > k_{max}
$$

##### (2)最大运行时间

同上，用于防止无限运行和便于服务器资源调度(如集群定时任务限制)

$$
t_{elapsed} < t_{max}
$$

## 引用

[^1]: BOYD S P, VANDENBERGHE L. Convex optimization[M]. Cambridge: Cambridge University Press, 2004.

[^2]: 笔记：数学基础/优化理论/凸分析/凸集与凸函数深入#一阶条件 2025.07.02更新