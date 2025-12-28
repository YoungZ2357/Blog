---
title: "PRCA论文阅读笔记"
date: 2025-12-26
draft: false
tags: ["RAG", "Generate-Read", "adapter", "PPO", "policy learning"]
categories: ["学习笔记"]
---


> **该模块可作为开发物件之一**
> 初步方案：符合语言的Seq2Seq类模型+PRCA调度逻辑微调

## 1 文献目标

问题
- LLM参数太多，调参成本有限
- 调整logits输出无法适用于通过API提供服务的黑盒模型

解决方案
- 创建即插即用PRCA 适配器
> PRCA模块和RECOMP信息压缩器处于同一生态位，都属于RAG适配器（Adapter）。具体实现上，可以只实现调度部分，
## 2 具体实现

### 2.1 训练潜在问题

PRCA模块本身也是一个生成式的模型，它与[[RECOMP(arXiv2310.04408)论文精读笔记#2.3 Abstractive Compressor| Abstractive Expressor]]相似，但是PRCA的训练信号来自于**奖励值**，而不是来自于LLM和自监督。

PRCA拦在LLM和文档之间的特点带来了如下问题：
1. 文档先经过PRCA吗，才给到LLM，使得检索质量、PRCA加工质量和生成器能力耦合在一起，形成黑箱，难以判定PRCA的实际效用
2. LLM本身也是个黑盒，无法从中获取梯度以传递给PRCA，标准的监督学习路径被截断
> 这也是RECOMP可能遇到的问题

PRCA通过划分信息提取阶段和奖励驱动阶段来解耦生成能力和内容调度能力的评估
### 2.2 信息提取阶段(Contextual Extraction Stage)

第一阶段遵从常规的**序列到序列模型**微调流程，其目标为：

$$
min \ L(\theta) = -\frac{1}{N}\sum_{N}^{i=1} C_{truth}^{(i)}\log(f_{PRCA}(S_{input}^{(i)}:\theta))
$$
符号：
- $C_{truth}^{(i)}$: 真实标签或者期望答案。论文原文没有给出具体获取途径，需要单独设计
- $f_{PRCA}$: PRCA神经网络，表达前向传播
- $S_{input}^{(i)}$: 查询

原论文中，$f_{PRCA}$使用了BART-Large，此处可以用任意**序列到序列模型**替代,或更改预训练数据集，以适配具体需求和语种

此处，$C$和$S$都是自然语言序列，具体获取方法为用序列到序列的tokenizer来分别处理，以得到可以被计算机处理的数组数据。

信息提取阶段是PRCA训练的**基础**，其作用为让序列到序列模型具备文本输出能力或适应领域知识。
### 2.3 奖励驱动阶段(Reward-Driven Stage)

#### 2.3.1 问题定义

第二阶段通过强化学习微调PRCA，使其生成的上下文更能帮助下游生成器产生正确答案。 具体而言，PRCA学习提取和组织信息的方式，以最大化生成器输出与真实答案的相似度。

需要注意的是，在PRCA的设计下：为保障模型的通用性，LLM被视作一个黑盒，只能**输出自然语言**，允许不输出如logprob，梯度等内部状态。其作用/目的如下：
1. **适配闭源模型** - 可使用任何提供文本API的LLM作为生成器(如GPT-4、Claude、Gemini等)，无需访问模型内部参数或梯度
2. **跨模型迁移能力** 
  - 训练阶段:可使用成本较低的模型(如DeepSeek)作为奖励信号来源 
  - 推理阶段:训练好的PRCA可无缝切换到其他生成器(如Claude、GPT-4) 
  - **降低训练成本(使用便宜的API训练)提升推理质量(使用更强的API推理)** 规避ToS限制(某些模型禁止用输出训练,但可用于推理)
> 请注意检查LLM提供商的使用协议，以保障训练行为合规！

该阶段将问题建模为马尔可夫决策过程(MDP): 

| 组件     | 定义                   |
| ------ | -------------------- |
| **状态** | 查询+预置内容+已生成token     |
| **动作** | 选择下一个token           |
| **奖励** | ROUGE-L(生成器答案, 真实答案) |
| **策略** | PRCA模型的参数θ           |
PRCA仍然是一个MDP而非POMDP，已生成token相当于Agent对环境做出的改变
#### 2.3.2 PPO Clip 的针对性转换
> PPO 是TRPO的工程化改进版本，继承了置信域方法"限制策略更新幅度"的核心思想，但用简单的Clip操作(PPO Clip)替代了TRPO中复杂的KL约束优化。详情以及PPO Penalty原理见<a href="https://arxiv.org/abs/1707.06347">PPO(arXiv)</a>


PRCA参考了**PPO Clip**的目标函数：
$$
max_{\theta} \quad J(\theta) = E_t[min(r_t(\theta)\cdot A^{GAE}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)) - \beta(V(s_t) - R_t)]
$$

其中$V$是Critic网络，价值网络需要如下的物件才能进行训练：
- 当前状态价值估计$V_t$
- 下一阶段状态价值估计$V_{t+1}$
- 即时奖励$r_t$

然而，该设计下的奖励值是在生成EOS token(结束生成)时得到的，其粒度大于模型所能做出的行为，故而即时奖励$r_t$是无法获取的，Critic网络无法训练。

原论文设计中利用了策略网络$\pi_{\theta}$来分配这个总奖励值，策略网络输出的是概率值：
$$
R_t = R_{EOS} \ * \ \frac{e^{\pi_{\theta}(a_t|s_t)}}{\sum_{t=1}^K e^{\pi_{\theta}(a_t|s_t)}} 
$$

其中，整个生成内容的奖励定义如下：
$$
R_{EOS} = \texttt{ROGUE-L}(O, O^*) - \beta \cdot D_{KL}(\pi_{\theta}||\pi_{\theta_{ori}}) 
$$
$$
\texttt{ROGUE-L}(X, Y) = \frac{\texttt{LCS}(X, Y)}{max(|X|, |Y|)}
$$

$K$是总token数

### 3 模型限制

1. 可插拔特性依赖于多个生成器的奖励信号
2. 强化学习方法收敛困难