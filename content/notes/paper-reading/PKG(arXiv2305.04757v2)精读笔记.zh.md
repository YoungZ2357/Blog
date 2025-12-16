---
title: "PKG论文精读笔记"
date: 2025-12-15
draft: false
tags: ["RAG", "LLM", "Generate-Read"]
categories: ["学习笔记"]
---

## 1 文献目标

问题：
- LLM缺乏领域针对性知识的获取，如相关知识及专有名词
- SOTA LLM通常是黑盒，缺乏透明性且成本高昂导致难以针对领域只是微调
- 能承担微调费用的用户有隐私暴露风险
解决方案：
- 通过instruction fine-tuning，在与训练阶段将指定领域的知识融入PKG模块
- 提问时，PKG先生成领域知识，领域知识传递给黑盒LLM辅助作答

## 2 具体实现

套用**Generate-Read** 的Modular RAG范式

### 2.1 内容生成
得到问题$Q$后，LLM通过最大化后验估计(MAP)来生成回应
$$
\hat{A} = argmax_{A} P(A|Q, M^{LLM})
$$
其中$M^{LLM}$是黑盒LLM的参数。该公式含义为：在给定查询$Q$和LLM参数$M^{LLM}$的情况下，总是返回“最可能”的答案。该公式是直接使用LLM时的答案返回机制。

### 2.2 知识对齐
此时向问题传入PKG模块，其参数为$M^{PKG}$。该模型同样通过最大后验估计来生成回答，此处仅处理背景知识：

$$
\hat{K} = argmax_KP(K|Q, M^{PKG})
$$
即，通过已有的PKG模块$M^{PKG}$，总是返回“最可能”的背景知识。

这一步的实现具体为使用 instructions, input, response三元组的形式来控制PKG模块的输出，大概模板具体如下：
```markdown
<元指令，告诉模型接下来的输入数据格式是什么，让模型清楚知道自己在干啥>
### Instruction
<用于PKG模块的查询>
% 例如: 
### Input
<具体输入的指令和数据>
例如： 
### Response:
<期望输出>
% 在此编写你希望模型回复的形式，例如具体的计算过程，要不要用某些指定框架实现，怎么呈现背景知识
```
以下是一个具体的例子
```markdown
The following content is an instruction of how to solve a background knowledge providing problem, which contains "instruction", "input" and "expected output". You need to follow such steps for further queries and return content similar to "expected output" descriped below.
### Instruction
Generate background knowledge that helps with data mining problems.
### Input
What is the average salary.
Table:
| name | salary |
| Alice | 70000 |
| Bob | 80000 |
### Output
To calculate average:
- Sum: 70000+80000=150000
- Count: 2 employees
- Average: 150000 / 2 - 75000
```

### 2.3 强化查询

将PKG模块融入到LLM中后，后验概率变更为：
$$
P(A|Q) = P(A|K, Q, M^{LLM}) P(K|Q, M^{PKG})
$$
即，在使用PKG模块的情况下，背景知识$K$由PKG完成建模，LLM再依赖于背景知识、问题和LLM参数做出回答。在已知查询$Q$产生回答$A$的概率被定义为 生成$K$ 的概率 乘以 基于$K$再生成回答的概率。

## 3 模型表现

黑盒模型：InstructGPT3.5
PKG模块载体：LLaMa-7B

Baseline:
- 直接生成答案
- Retrive-Read范式完成RAG，具体使用了BM25和DPR（REPLUG微调）
- 让InstructGPT自己指导自己，而非挂载PKG - 这将说明微调LLM的必要性，排除通过API反复调用黑盒模型来辅助自己的可能最优

实验数据集：FM2，NQ-Table，MedMC-QA

结果：
- 对于FM2和NQ-Table, Retrieval-Read和self-guiding都能涨点，但是两者效果平齐，PKG略胜于前两者
- 对于MedMC-QA，PKG有较显著优势