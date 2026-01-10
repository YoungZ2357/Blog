---
title: "Logistic Regression"
date: 2025-12-09
draft: false
tags: ["Supervised Learning", "Classification"]
categories: ["Study Notes"]
---
> IMPORTANT: This note is translated by LLM and fixed manually. See Chinese version for a more accurate note

## 1. Algorithm Overview


Logistic regression predicts the probability of an event by mapping the **output of linear regression** to the **probability space**:
$$
P(y=1|x;\theta)=h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
This formula represents: given model parameters $\theta$ and input feature values $x$, the probability that the classification label $y=1$ is $h_{\theta}(x)$.
The pseudocode for a single **inference** of this model is approximately as follows:

**Algorithm: LogisticForward**
Input: Feature values $x$
Parameters: Vector $\theta$
Output: Probability value that the data is positive class $P(y=1|x;\theta)$

1. Calculate linear combination $z = \theta^T x$
2. Apply Sigmoid function: $p=\sigma(z)=1/(1+e^{-z})$
3. Classification decision: if $p \geq 0.5$, then $\hat{y}=1$, otherwise $\hat{y}=0$
4. Return $\hat{y}$

The pseudocode for the **training process** of this model using **gradient descent** is approximately as follows:

**Algorithm: LogisticTrainWithGD**
Input: Training set $(X, y)$, where $X\in \mathbb{R}^{m\times n}$
Hyperparameters: Learning rate $\alpha$, number of iterations $T$
Output: Parameter vector $\theta$

1. Initialize parameters $\theta$  # Initialization can be all zeros, or normal distribution random initialization, etc.
2. For $t=1$ to $T$:
		1. Initialize gradient $g\leftarrow 0$
		2. For $i=1$ to $m$:
			1. $p_i \leftarrow$ **LogisticForward($x_i$, $\theta$)**
			2. $g \leftarrow g+(p_i-y_i)\cdot x_i$   # Calculate new gradient
		3. Update parameters: $\theta \leftarrow \theta - \frac{\alpha}{m}\cdot g$  # Gradient descent update, **see Objective Function section for details**
3. Return $\theta$

To use other update methods, simply replace the two steps of calculating gradient and updating parameters with the corresponding formulas.

> In practice, this usually manifests as directly modifying values at corresponding physical addresses, rather than returning values
## 2. Objective Function
### 2.1 Objective Function Formula
Given training data $(x_i, y_i)$, the objective function is:
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y_ilog(h_{\theta}(x_i))+(1-y_i)log(1-h_{\theta}(x_i))]
$$
where $h_{\theta}$ is the Sigmoid function:
$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
The Sigmoid function can be used to convert output values to binary classification probability values. For multi-class problems, the Softmax function is used.
![Function of Sigmoid and Softmax](imgs/lgstr.png)


This objective function is also called **logistic loss (statistical perspective)**. In binary classification, it is equivalent to **cross-entropy loss (information theory perspective)**, and can be extended to multi-class problems through Softmax. The fitting goal is to **minimize the objective function**.


Properties of the objective function are as follows:

| Property Name    | Property            | Comment                               |
| ------- | ------------- | -------------------------------- |
| **Convexity**  | Convex, not strongly convex         | Unique solution is theoretically guaranteed. Strong convexity can be ensured by adding a quadratic regularization term to accelerate convergence |
| **Smoothness** | Satisfies Lipschitz condition | Infinitely differentiable, allowing theoretical analysis of convergence speed |
### 2.2 Gradient Descent Parameter Update
> Gradient descent is a first-order optimization method. For its principle, see [[Gradient Descent and its Convergence|First-order Methods - Gradient Descent]]


The **gradient** of the objective function is as follows, calculation process omitted:
$$
\nabla J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)
$$
In general gradient descent, parameters $\theta$ are updated as follows:
$$
\theta \leftarrow \theta - \alpha \nabla J(\theta)
$$
where $\alpha$ is the learning rate, i.e., the update step size. The larger $\alpha$ is, the faster the update, but also more likely to miss the optimal point. $\leftarrow$ is the assignment operator, equivalent to the equals sign in Python.

> Note that many update methods implicitly minimize the objective function. In actual code, the loss value does not participate in calculation, but serves as an evaluation metric.
### 2.3 Quasi-Newton Method Parameter Update
> Quasi-Newton method is an optimization method between gradient descent (first-order) and Newton's method (second-order). For its principle, see [[Newton's Method|Second-order Methods - Newton's Method]]

The parameter update method for quasi-Newton method is as follows:
$$
\theta_{k+1} \leftarrow \theta_{k} - \alpha_kB_k^{-1}\nabla J(\theta_k)
$$
where $B_k^{-1}$ is the Hessian matrix approximation of $\theta_k$. There are many methods for calculating matrix approximation. Here we introduce the BFGS quasi-Newton method.

$$
H_{k+1} \leftarrow H_k + \frac{ss^T}{s^Ty} - \frac{H_k yy^T H_k}{y^T H_k y}
$$
where $s=\theta_{k+1} - \theta_k$. Detailed derivation process omitted (because I'm too lazy to learn it, it's very troublesome)
This method avoids calculating second-order derivatives, which would further increase computational complexity.

The Hessian matrix approximation can be initialized in the following ways:
- Identity matrix initialization
- Identity matrix initialization scaled by $\beta$
- Custom initialization based on specific tasks

### 2.4 Logistic Regression Function with Regularization Term
> This example is introduced in [[Strong Convexity and Smoothness#1.4.2 Logistic Regression + L2 Regularization|Strong Convexity Notes]]. For the specific strong convexity proof, please refer to the corresponding section.

According to the definition of strong convexity, adding a quadratic function to the objective function and transforming it can make the objective function strongly convex. The transformed form is as follows:
$$
J(w) = \frac{1}{n}\sum_{i=1}^{n}log(1+e^{(-y_iw^Tx_i)}) + \frac{\lambda}{2}\|w^2\|
$$
At this point, the objective function is $\lambda$-strongly convex.


## 3. Algorithm Implementation
We use the following method to obtain classification data for testing:
```python

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# Generate classification data
X, y = make_classification(n_samples=114514, n_features=5, n_class=2, n_informative=2, random_state=1919810, n_clusters_per_class=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=.2, random_state=1919810
)

```

### 3.1 Simple Implementation
> In practical applications, implement it this way, **don't reinvent the wheel**


**sklearn calling `LogisticRegression` to implement the algorithm**
> sklearn is a classic row-based operation library, suitable for research, implementing ideas, and other small-scale data analysis

```python
from sklearn.linear_model import LogisticRegression  # Directly import logistic regression method
import numpy as np

model = LogisticRegression(
	penalty='None',  # No regularization, consistent with the formula above. Set to L2 to make the objective strongly convex
	# C=1.0,   # Only effective when regularization is enabled. The smaller the number, the stronger the regularization intensity
	max_iter=100,  # Maximum number of iterations
	solver='saga',  # Gradient descent solver. The model default is actually L-BFGS, a higher-performance BFGS method
	random_state=1919810  # Random seed
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Predict, output predicted label value array
y_prob_pred = model.predict_proba(X_test)  # Output predicted label probability array
```

Note: Parameter $C$ is actually the reciprocal of $\lambda$ in the strong convexity definition, i.e.:
$$
C = \frac{1}{\lambda}
$$
The smaller $C$ is, the greater the strong convexity of the objective function. The $\frac{1}{2}$ in strong convexity does not participate in the reciprocal operation. This will be demonstrated in [[Logistic Regression#3.2 Step-by-step Implementation (Local Python, No Advanced API)|Step-by-step Implementation]]



**PySpark calling `LogisticRegression`**
> PySpark is the Python interface of Spark. It operates on columns as units, suitable for massive data processing, not suitable for building more sophisticated and complex classifiers with innovation. When writing PySpark code, focus on batch operations on data (entire DataFrame or columns)

> PySpark 3.0 or above is recommended


Assume we have DataFrame variables `traindata` and `testdata` in the following format:

| x_vec: Vector   | y_vec: Vector |
| --------------- | ------------- |
| [1, 1, 1, 1, 1] | 1             |
| [0, 1, 0, 1, 1] | 2             |
| ...             | ...           |

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder \  # Create Spark session to interact with cluster resources/local resources
		.appName("LRDemo") \
		.getOrGreate()

# This is an Estimator subclass that can define model parameters and generate a Transformer subclass with trained parameters through .fit
lr = LogisticRegression(
	featuresCol="x_vec",  # Select feature column
	labelCol="y_vec",  # Select label column
	maxIter=100,  # Maximum number of iterations
	family="binomial",  # Problem is binary classification
	regParam=1  # Regularization intensity, equivalent to sklearn's C

)

# This is a Transformer subclass that can only perform inference based on training parameters given by the Estimator subclass
model = lr.fit(train_data)  # PySpark generates new variables, unlike sklearn which modifies existing variables

predictions = model.transform(test_data)  # Model inference
# columns = [x_vec, y_vec, prediction]

```
### 3.2 Step-by-step Implementation (Local Python, No Advanced API)

**Objective Function (Loss Function) Calculation**
We only need to implement the calculation method for a single sample and then take the mean. The formula is:
$$
y_ilog(h_{\theta}(x_i)))+(1-y_i)log(1-h_{\theta}(x_i))
$$

```python

import numpy as np

# Sigmoid
def sigmoid(z):
	"""
	This function implements h(x), i.e., the Sigmoid function

	:param z: Sigmoid function input value, can be a scalar or vector
	:return : Sigmoid function output value
	"""
	# np.where has three parameters, [judgment condition/boolean value, value when condition is true, value when condition is false]
	return np.where(z >= 0,
					1 / (1+np.exp(-z)),  # Use standard form when greater than 0
					np.exp(z)/(1+np.exp(z)))  # Use equivalent transformation when less than 0 to avoid overflow

def logistic_loss(theta, X, y):
	"""
	Calculate standard log loss
	:param theta: Algorithm parameters, should be an array, size is (number of features, ), which is also what we want to update
	:param X: Feature values of data, should be an array, size is (number of samples, number of features)
	:param y: Label values of data, should be an array, size is (number of samples), values must be 0 or 1
	:return loss: Logistic (binary cross-entropy/log) loss value
	"""


	z = X @ theta  # Calculate linear combination
	h = sigmoid(z)
	epsilon=1e-15  # Very small number to avoid log(0)
	h = np.clip(h, epsilon, 1-epsilon)  # Clip operation results
	loss = -np.mean(
		y * np.log(h) + (1 - y) *np.log(1 - h)
	)
	return loss

```

Objective function with regularization term

```python
def logistic_loss_reg(theta, X, y, lambda_reg):
	"""
	Calculate loss with regularization term
	:param theta: Algorithm parameters
	:param X: Data feature values
	:param y: Data label values
	:param lambda_reg: Regularization intensity, directly determines the strong convexity of the objective function. Function becomes lambda_reg-strongly convex function. Set to 0 for no regularization
	@return : Log loss with regularization term

	"""
	base_loss = logistic_loss(theta, X, y)
	reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)  # Regularization needs to remove intercept term, same below
	return base_loss + reg_term


```

Calculate gradient
```python
def get_grad(theta, X, y):

	n_samples = len(y)
	z = X @ theta
	h = sigmoid(z)
	gradient = (1/n_samples) * X.T @ (h-y)  # Directly apply formula
	return gradient

def get_grad_reg(theta, X, y, lambda_reg):

	n_samples = len(y)
	gradient = get_grad(theta, X, y)
	reg_gradient[1:] = (lambda_reg / m) * theta[1:]
	return gradient + reg_gradient


```

Gradient descent update
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

Class implementation of all code
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
