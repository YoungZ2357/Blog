---
title: "Gradient Descent and Convergence Analysis"
date: 2025-12-11
draft: false
math: true
tags: ["convex optimization", "descent method", "optimization theory"]
categories: ["Learning Notes"]
---
> IMPORTANT: This note is translated by LLM and fixed manually. See Chinese version for a more accurate note

## 1. General Descent Methods

### 1.1 Basic Form of Descent Methods

> Note: Descent methods do not require convexity, but convexity provides significant guarantees for solving optimization problems.

The prototype of gradient descent is the **descent method**.

A descent algorithm generates a sequence of optimization points $x^{(k)}, k=1, \cdots$, where

$$ x^{(k+1)} = x^{(k)} + t^{(k)}\Delta x^{(k)} $$

and $t^{(k)} > 0$ (unless $x^{(k)}$ is already optimal).

Here, $\Delta x^{(k)}$ is a vector called the **search direction**, $t^{(k)}\Delta x^{(k)}$ is called the **step** (in practice, often a multi-dimensional array of the same size as the parameters); $k$ represents the iteration number; the scalar $t^{(k)}$ is the update step size.

### 1.2 Descent Condition

For **all descent methods**, as long as $x^{(k)}$ is not an optimal point, we have:

$$ f(x^{(k+1)}) < f(x^{(k)}) $$

> Note: Descent algorithms default to minimizing the objective function. All problems originally formulated as "maximizing an objective function" can be reformulated in this form.

The descent condition for descent methods:

For **convex functions**, if we choose the direction $d=y-x$ for updating, considering the first-order condition for convex functions: $f(y) \geq f(x)+\nabla f(x)^T(y-x)$, to make the function descend, i.e., $f(y) < f(x)$, **we must have $\nabla f(x)^{T}(y-x) < 0$.**

Here, $\nabla f(x)^{T}(y-x)$ is the **directional derivative**.

| Property                | Search Direction      | Step                       | Directional Derivative         |
| ----------------------- | --------------------- | -------------------------- | ------------------------------ |
| Mathematical Definition | $d^{(k)}$             | $\alpha^{(k)} d^{(k)}$     | $\nabla f(x)^{T}(y-x)$         |
| Geometric Meaning       | Direction of movement | Actual displacement vector | Rate of change along direction |

### 1.3 General Descent Algorithm Framework

General descent methods typically follow these steps:

Given starting point $x \in \mathbf{dom}\ f$

Repeat:

1. Determine descent direction $\Delta x$
2. Line search to **find step size**, choose step size $t > 0$
3. Update point $x \leftarrow x+\alpha\Delta x$

Until **stopping criterion is satisfied**

## 2. Line Search

> The use of line search means that general descent methods are not **fixed step size methods**. Strictly speaking, this method should be called **ray search**, as the search domain is $t \in [0, +\infty)$.

General descent methods typically use the following two line search methods:

### 2.1 Exact Line Search

Exact line search requires using the step size with maximum descent at each iteration, updating along the ray ${x+\alpha\Delta x \mid t\in\mathbf{R}_{+}}$:

$$ \alpha = \arg\min_{s\geq 0} f(x+s\Delta x) $$

Note that this method treats each step as a one-dimensional optimization problem, requiring an optimization method call (such as golden section search) for each update, leading to excessive computational cost in practice.

### 2.2 Backtracking Line Search

Backtracking line search is an **inexact** method. Similar to exact line search, it updates along the ray ${x+\alpha\Delta x \mid t\in\mathbf{R}_{+}}$, but only requires **sufficient decrease** in function value. This method mainly has a contraction parameter $\rho$ that controls the degree of descent at each iteration.

Algorithm:

1. Given descent direction $\Delta x$, parameter $\rho \in (0, 1)$
2. $\alpha\leftarrow 1$
3. While **descent condition not satisfied**: $\alpha\leftarrow \rho \alpha$

The descent condition can be any of the following:

**(1) Armijo Search Condition (Sufficient Descent Condition)**

$$ f(x^k + \alpha_k d^k) \leq f(x^k) + \sigma_1 \alpha_k \nabla f(x^k)^T d^k $$

Where:

- $x^k$ is the current point at iteration $k$
- $d^k$ is the search direction at iteration $k$
- $\alpha^k$ is the step size at iteration $k$
- $\sigma_1$ is the Armijo parameter, controlling sufficient descent

```python
from autograd import grad
import numpy as np

def line_search_armijo(f, x, delta_x, rho=.5, c=1e-4, max_iter=100) -> float:
    """Backtracking line search using Armijo condition
    
    :param f: objective function
    :param x: current point
    :param delta_x: descent direction
    :param rho: contraction factor
    :param c: Armijo constant, default 1e-4, typical range [1e-6, 1e-4]
    :param max_iter: maximum iterations
    :return alpha: appropriate step size
    """
    alpha = 1.
    grad_f = grad(f)
    f_x = f(x)
    grad_x = grad_f(x)

    dir_deriv = np.dot(grad_x, delta_x)
    if dir_deriv >= 0:
        print("Direction vector is not a descent direction")
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

Function should be defined as follows. Example with Rosenbrock function $f(x) = (1-x_1)^2 + 100 \times (x_2 - x_1^2)^2$:

```python
def rosenbrock(x: list):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

![Armijo Condition Illustration]( /images/math/cvx/descent1.png)

**(2) Curvature Condition**

$$ \nabla f(x^k + \alpha_k d^k)^T d^k \geq \sigma_2 \nabla f(x^k)^T d^k $$

**(3) Wolfe Condition (Weak Wolfe Condition)**

Satisfies both **Armijo condition** and **curvature condition**.

**(4) Strong Wolfe Condition**

Satisfies both **Armijo condition** and **strong curvature condition**:

$$ |\nabla f(x^k + \alpha_k d^k)^T d^k| \leq |\sigma_2 \nabla f(x^k)^T d^k| $$

## 3. Gradient Descent Method

Gradient descent typically has the following form:

Given starting point $x \in \mathbf{dom}\ f$

Repeat:

1. $\Delta x \leftarrow -\nabla f(x)$
2. Line search to **find step size**, choose step size $t > 0$
3. Update point $x \leftarrow x+\alpha\Delta x$

Until **stopping criterion is satisfied**

## 4. Convergence Analysis

### 4.1 Basic Assumptions

To analyze convergence, we typically need the following assumptions:

#### Assumption 1: Lipschitz Smooth Gradient Condition (Smoothness)

This condition is usually necessary. There exists constant $L > 0$ such that for all $x, y\in \mathbb{R}^n$:

$$ |\nabla f(x) - \nabla f(y)| < L |x-y| $$

#### Assumption 2: Strong Convexity First-Order Condition

This condition is not necessary, only used for convergence rate analysis. There exists constant $\mu > 0$ such that for all $x, y \in \mathbb{R}^n$:

$$ f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}|y-x|^2 $$

### 4.2 Convergence Theorems

#### 4.2.1 Convergence with Fixed Step Size

Under **Assumption 1**, with fixed step size $\alpha \leq \frac{1}{L}$:

$$ f(x_k) - f^* \leq \frac{|x_0 - x^*|^2}{2\alpha k} $$

Example: Suppose the initial point is 10 units from the optimal point, i.e., $|x_0 - x^*| = 10$, with fixed step size $\alpha = 1e-2$

The error upper bound is:

$$ f(x_k) - f^* \leq \frac{100}{2 \times 0.01 \times k} = \frac{5000}{k} $$

To achieve error $\leq 0.1$, we need $k \geq 50000$ steps To achieve error $\leq 0.01$, we need $k\geq 500000$ steps

This descent method has $O(\frac{1}{k})$ sublinear convergence rate, with convergence speed gradually slowing down.

#### 4.2.2 Convergence with Backtracking Line Search

Under **Assumption 1**, gradient descent with backtracking line search satisfies:

$$ f(x_k) - f^* \leq \frac{L|x_0 - x^*|^2}{2k} = \frac{|x_0 - x^*|^2}{2(\frac{1}{L})k} $$

This descent method also has $O(\frac{1}{k})$ sublinear convergence rate but is linearly affected by the value of $L$.

#### 4.2.3 Convergence for Strongly Convex Functions

Under **Assumptions 1 and 2**, gradient descent has **linear convergence** rate:

$$ f(x_k) - f^* \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f^*) $$

Where the convergence rate $\rho = 1 - \frac{\mu}{L} = 1 - \frac{1}{\kappa}$, and $\kappa$ is the condition number.

To reduce the error to $\epsilon$ times the original, the required number of iterations is: $k = \frac{\log(\epsilon)}{\log(\rho)} = \frac{\log(\epsilon)}{\log(1/\kappa)}$

| Condition Number $\kappa$ | Convergence Rate $\rho$ | Error Reduction per Step $\epsilon$ | Iterations for 10× Precision $k$ |
| ------------------------- | ----------------------- | ----------------------------------- | -------------------------------- |
| 2                         | 0.5                     | 50%                                 | ~3                               |
| 10                        | 0.9                     | 10%                                 | ~22                              |
| 100                       | 0.99                    | 1%                                  | ~230                             |
| 1000                      | 0.999                   | 0.1%                                | ~2300                            |

### 4.3 Classification of Convergence Rates

Convergence rates are typically defined by two methods:

- Function value error: $f(x_k) - f^*$, the difference between objective function value and optimal function value, the **primary goal** of optimization
- Point distance error: $|x_k-x^*|$, the distance from current point to optimal point

Notation:

- $f^*$: optimal function value
- $K$: a finite iteration threshold where convergence properties begin to hold
- $k$: actual iteration number
- $p$: convergence rate exponent
- $\rho$: convergence rate, smaller means faster convergence
- $\epsilon$: precision parameter, representing allowable error size

All convergence rates comparison:

![Convergence Rate Comparison]( /images/math/cvx/descent8.png)

#### 4.3.1 Logarithmic Convergence

There exist constants $C > 0, K\in\mathbb{N}$ such that for all $k\geq K$:

$$ f(x_k) - f^* \leq \frac{C}{\log k} $$

Iterations needed to reach precision: $O(e^{1/\epsilon})$

#### 4.3.2 Sublinear Convergence

There exist constants $C > 0, K\in \mathbb{N}$ such that for all $k\geq K$:

$$ f(x_k) - f^* \leq \frac{C}{k^p} $$

| Function Value Convergence     | Asymptotic Behavior | Iterations to Reach $\epsilon$ |
| ------------------------------ | ------------------- | ------------------------------ |
| $f(x_k) - f^* \leq C/\sqrt{k}$ | Slow                | $O(1/\epsilon^{2})$            |
| $f(x_k) - f^* \leq C/k$        | Medium              | $O(1/\epsilon)$                |
| $f(x_k) - f^* \leq C/k^2$      | Fast                | $O(1/\sqrt{\epsilon})$         |

#### 4.3.3 Linear Convergence

There exist constants $C > 0, \rho \in(0, 1)$ such that:

$$ |x_k - x^*| < C\rho^k $$

Taking logarithm:

$ \log|x_k - x^*| \leq \log C + k\log \rho $

![Linear Convergence]( /images/math/cvx/descent2.png)

#### 4.3.4 Superlinear Convergence

![Superlinear Convergence]( /images/math/cvx/descent6.png)

#### 4.3.5 Quadratic Convergence

![Quadratic Convergence]( /images/math/cvx/descent7.png)

### 4.4 Common Stopping Criteria

The following criteria may not satisfy (or use) previous assumptions/theorems due to non-convexity and problem characteristics, but still work in practice. Multiple criteria are recommended to be used together.

#### 4.4.1 Gradient-Based Criteria

> At optimal point $x^*$, we have $\nabla f(x^*)=0$

##### (1) Absolute Gradient Criterion

$$ |\nabla f(x_k)| < \epsilon_{abs} $$

Where $\epsilon_{abs}$ is the **absolute** threshold for convergence.

##### (2) Relative Gradient Criterion

$$ |\nabla f(x_k)| < \epsilon_{rel} \cdot |\nabla f(x_0)| $$

Where:

- $\epsilon_{rel}$ is the **relative** threshold for convergence
- $x_0$ is the starting point

Improved version to handle small initial gradients:

$$ |\nabla f(x_k)| < \epsilon_{rel} \cdot \max(1, |\nabla f(x_0)|) $$

#### 4.4.2 Function Value-Based Criteria

##### (1) Absolute Function Value Change Criterion

$$ |f(x_k) - f(x_{k-1})| < \epsilon_f $$

##### (2) Relative Function Value Change Criterion

$$ \frac{|f(x_k) - f(x_{k-1})|}{|f(x_{k-1})| + \epsilon_{mach}} < \epsilon_{f,rel} $$

Where $\epsilon_{mach}$ is machine precision to prevent division by zero.

#### 4.4.3 Parameter-Based Criteria

> Note: Here $\theta$ refers to **model trainable parameters**

##### (1) Absolute Parameter Change

$$ |\theta_k - \theta_{k-1}| < \epsilon_{\theta} $$

##### (2) Relative Parameter Change

$$ \frac{|\theta_k - \theta_{k-1}|}{|\theta_{k-1}| + \epsilon_{mach}} < \epsilon_{x,rel} $$

##### (3) Computing Parameter Differences

When parameters $\theta$ are in **matrix** form while $\epsilon$ is a **scalar**, we typically use **flatten + compute norm** to convert parameters to scalars:

```python
def param_method_abs(params_k: dict, params_k_1: dict, epsilon=1e-6) -> bool:  
    """
    :param params_k: parameter dict at iteration k, e.g. {'W1': matrix, 'b1': vector}  
    :param params_k_1: parameter dict at iteration k+1
    :param epsilon: absolute convergence threshold  
    :return: whether absolute threshold convergence condition is satisfied  
    """    
    flat_k = np.concatenate([p.flatten() for p in params_k.values()])  
    flat_k_1 = np.concatenate([p.flatten() for p in params_k_1.values()])  
    abs_change = np.linalg.norm(flat_k - flat_k_1)  
    return abs_change < epsilon
```

#### 4.4.4 Resource Limit-Based Criteria

##### (1) Maximum Iteration Count

Prevents infinite running:

$$ k > k_{max} $$

##### (2) Maximum Running Time

Prevents infinite running and facilitates server resource scheduling:

$$ t_{elapsed} < t_{max} $$

------

## References

Based on:

- BOYD S P, VANDENBERGHE L. Convex optimization. Cambridge University Press, 2004.