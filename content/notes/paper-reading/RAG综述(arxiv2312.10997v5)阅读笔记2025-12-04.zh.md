---
title: "RAG综述阅读笔记"
date: 2025-12-11
draft: false
tags: ["RAG", "LLM", "综述"]
categories: ["学习笔记"]
---



> 论文链接：https://arxiv.org/abs/2312.10997
> [[RAG survey.pdf|RAG综述]]
>
> 近期更新：2025.12.15，更新次数：1
## 1.RAG

### 1.1 什么是RAG
全称：**检索增强生成**（**Retrieval-Augmented Generation**）。
### 1.2 RAG作用阶段
RAG可作用于LLM生命周期的如下位置
- 预训练：需要自行训练
- 微调：需要本地部署
- 推理：这是RAG最早被应用的LLM生命阶段，也是我唯一能够执行操作的阶段

RAG技术早期注重于推理阶段应用，而后逐渐转向微调阶段和预训练阶段

### 1.3 RAG常见范式
#### **Naive RAG**
Naive RAG只有indexing, retrieval和generation三个步骤，是一个线性的过程。

1. indexing: 包含数据预处理（PDF, markdown, etc. $\rightarrow$ txt），分块，获取嵌入向量
2. retrieval: 用相同的嵌入模型嵌入用户查询，然后检索文件块
3. generation: 用文件块扩充查询，让LLM生成答复

> 特别注意：**文档嵌入是离线的**，用户输入嵌入是在线的

该方法弊端：
1. 检索部分在精确率和召回率存在不足
2. 模型或将遭遇幻觉，导致输出不存在内容
3. 强化存在障碍：输出不连贯、相似检索导致重复响应

#### **Advanced RAG**
该范式主要基于Naive RAG增加了预处理、后处理手段：
1. **预强化(pre-retrieval)**：用于强化indexing和用户查询的效用，这是可操作的部分之一。包含如下方法
   - Query Routing: 要求某个查询使用指定的检索范围
   - Query Rewriting: 重写查询
   - Query Expansion: 扩展查询以具体化检索目标，防止找不到先验知识

2. 后强化(post-retrieval)：对查询进行进一步处理。这一步**已经被主流RAG框架实现了**，无需重新搓轮子。主要包含如下步骤
   - Rerank: 重排序文件块
   - Summary: 压缩文件块内容
   - Fusion: 组合多查询（包括不同的查询策略）整合文件块

#### **Modular RAG**
这是一种更优秀，更复杂的RAG范式。论文介绍了如下模块（有一部分是对Advanced RAG中改良方法的封装）

| 模块名          | 作用                  |
| ------------ | ------------------- |
| Search       | 场景针对性、多数据源          |
| Fusion       | 多查询策略，可并行           |
| Memory       | 强化对话/文档上下文，无界记忆池    |
| Routing      | 多数据源，**更优查询路径**     |
| Predict      | 以LLM**直接预测**相关内容，降噪 |
| Task Adapter | 对下流任务定制retrieval方案  |

> 注：DeepSeek R1 使用的是全强化学习+GRPO，而非类似于Modular RAG + RL的过程

Modular RAG 允许RAG系统**跳出传统的线性执行顺序**，论文介绍了如下新实现方式



| 方案名                             | 类型      | 流程                                                                    | 特征         | 适用情景        |
| ------------------------------- | ------- | --------------------------------------------------------------------- | ---------- | ----------- |
| RRR(Rewrite-Retrieve-Read)      | Pattern | Query $\rightarrow$ Rewrite $\rightarrow$ Retrieve $\rightarrow$ Read | 改写查询       | 查询不清晰、用词不当  |
| Generate-Read                   | Pattern | Query $\rightarrow$ Generate $\rightarrow$ Read                       | 无检索        | 模型本身具有知识    |
| Recite-Read                     | Pattern | Query$\rightarrow$Recite$\rightarrow$Read                             | 带权检索       | 知识点多，需要侧重   |
| DSP(Demonstrate-Search-Predict) | Pattern | Demonstrate$\rightarrow$Search$\rightarrow$Predict                    | 模块间协作，相互强化 | 复杂推理，需要已有示例 |
| Hybrid Retrieval                | Pattern | Query$\rightarrow$多路检索$\rightarrow$Fusion$\rightarrow$Read            | 多策略融合      | 需要多角度信息源    |
> Read是指将内容读入LLM



## 2.检索技术

### 2.1 检索源

| 数据类型                 | 特点       | 主要挑战       | 代表方法                                | 检索粒度                             |
| -------------------- | -------- | ---------- | ----------------------------------- | -------------------------------- |
| **非结构化 (Text)**      | 最常用,来源广泛 | 语义理解,长文本处理 | Atlas, FLARE, RAVEN, 大部分RAG方法       | Token, Sentence, Chunk, Document |
| **半结构化 (PDF/Table)** | 包含文本+表格  | 分割、检索困难    | TableGPT, PKG                       | Chunk, Document                  |
| **结构化 (KG)**         | 精确       | 构建维护成本高    | KnowledGPT, G-Retriever, SURGE, RoG | Entity, Triplet, Sub-Graph       |
| **LLM生成**            | 无需外部数据   | 知识有限，幻觉风险  | GenRead, SKR, Selfmem               | Chunk, Document                  |
### 2.2 索引化优化
索引化是指将文档分块、嵌入并存入向量数据库的过程，其主要有如下的优化方案：

#### 分块策略
最简单常见的方法是**固定token大小分块**，大块含有更多信息但产生噪声，小块则反之。分块或将导致**句义截断**，需要应用滑窗方法。

代表方案：
- 等token大小分块: 块大小影响显著，语义完整 - 内容长度 难以平衡
- Small2Big: 小文档作为单元，组织后传入LLM
#### 元数据附加
可通过向文档块附加元数据来增强检索效果（需要[[有限资源NLP综述.pdf|Efficient NLP]]？），先过滤一遍知识块能提升检索精度，也可通过限制时间戳来使得RAG对时间敏感

代表方案：
- Reverse HyDE: 用LLM生成可根据文档回答的问题，基于此计算相似度而非文本本身

#### 结构化索引
对文档建立一个分层结构以加速检索和处理

代表方案
- 分层索引结构：对文档建立父子关系，并使得chunk（作为节点）也以此相连，形成树状结构。该方法可削弱LLM幻觉。
- 知识图谱索引：使用知识图谱结构组织chunk（作为节点）而非简单的树结构，从语义层级标记chunk（类似实体），能从因果层级辅助LLM进行推理。**KGP**是该思想的一种实现方式。

### 2.3 查询优化
用户的提问主要存在如下的问题：
- 问题本身不清晰。这一点可能是问题复杂但语言能力不足，亦或是描述本身不精确，LLM没法识别
- 语言特性导致文本难识别。如中文分词，英文缩写（尽管可能被广泛认可，LLM仍然可能乱识别）

代表方案
- 查询扩展(Query Extension)：将单个问询分割成多个查询。主要有：1) 直接分割成多个查询（Prompt Engineering实现）；2) 生成子查询（Prompt Engineering，least-to-most method）；3) 验证链(Chain-of-Verification)：让LLM来验证子问题是否合理可靠
- 查询转换(Query Transformation)：直接变化查询本身。主要有：1) 查询重写，可用如RRR，BEQUE等方法；2) 让LLM根据原本的查询生成新的查询，可用如HyDE(主要步骤为让LLM根据查询估计更好的查询，然后基于此检索)
- 查询路由(Query Routing)：根据查询本身，将其导向不同的Pipeline。主要有：1)元数据路由/过滤。同之前的元数据附加所属，基于元数据进行chunk筛选；2) 语义路由。提前获取chunk的语义，以此辅助检索


### 2.4 嵌入
嵌入可以通过稀疏编码器（如BM25，情感词典，概率模型）或者稠密检索器实现（如预训练语言模型）

代表方案：
- 混合检索：即组合使用稀疏编码器和稠密检索器。还可使用预训练语言模型获取术语权重，并以此强化稀疏编码器
- 微调嵌入模型：这一步可以使用任意微调方法（如LoRA）来具体实现。主要方案有：1) 根据领域知识微调；2) 令检索器对齐LLM（如LM-supervised Retriever）；3) LLM生成查询微调知识稀缺的模型（如PROMPTAGATOR）；4) LLM生成奖励信号，嵌入模型通过硬编码标签和奖励信号微调（如LLM-Embedder）；5) 检索器与LLM协同计算与文档的KL散度（REPLUG）；6) RLHF微调检索器




### 2.5 适配器
适配器是独立于LLM的一个部分，当使用API时，优化适配器是强化RAG系统性能的重要举措。以下是原问提及的经典方法

| 方法名                             | 流程/特征                                                    | 作用位置 |
| ---------------------------------- | ------------------------------------------------------------ | -------- |
| UP-RISE                            | 训练轻量化检索器，以通过预先设立的提示词池选取并给出输入提示 | 检索前   |
| AAR(Augmentation-Adapted Retriver) | 适用多任务。PS：我没理解它到底能干啥                         | 检索后   |
| PRCA                               | 即插即用的奖励驱动内容适配器。PS：这个也是                   | 检索后   |
| BGM                                | 训练序列到序列模型来连接检索器和LLM，且保持这两者不变        | 端到端   |
| PKG                                | 额外挂载一个开源LLM，在预训练时对其融入领域知识。在RAG流程中可以作为SOTA LLM的额外信息源，以保护隐私和强化领域知识。该方法对**表格数据**明显有效。见[查看RAG综述论文笔记]({{</* relref "PKG(arXiv2305.04757v2)精读笔记.zh" */>}}) | 端到端   |

## 3.查询生成技术
原论问从**调整检索结果**和**微调LLM**两个角度讲解了如何生成新查询

### 3.1 调整检索结果方法
1. 重排序(Reranking): 即重新排序文件块以使得更相关的内容优先排序。分为Rule-based和Model-based两类
2. 内容选择: 限制输入文档量，分为降低token量和降低文档块量。降低token：LLMLingua(使用轻量LLM移除无意义token)，PRCA(训练信息提取器)，RECOMP(对比学习创建information condenser)。降低块：Filter-Reranker范式(SLM-LLM协同，SLM作过滤器)，Chatlaw(LLM自行判别)
> SLM，即Small Language Model，指体积小的语言模型
### 3.2 微调LLM方法
针对指定场景微调LLM主要有如下优势：1. 领域强相关而非泛泛而谈；2.调整模型输入输出（如针对某些格式的解析、输出风格）

- SANTA框架：分三部分训练以概括语义和结构的细微差异
- RA-DIT：以KL散度为基础在检索器和生成器之前构建评分

## 4.强化过程
原论文介绍了三种强化过程，比你高说明单强化步骤能力不足，且对需要多步推理的问题存在显著不足。

### 4.1 迭代式强化(Iterative Retrieval)
该种过程特征为**尝试多次获取文档数据**。迭代式强化有利于获取更多额外参考信息，并辅助之后的回答生成，但同时也更容易受到无关内容累积和语义截断的影响
<center><img src="imgs/aug1.png"></center>
**经典方法：RETGEN**

### 4.2 递归式强化
该种过程旨在通过反馈回路来逐渐提升检索效果

<center><img src="imgs/aug2.png"></center>
**经典方法：IRCoT(使用chain-of-thought)，ToC(使用分类树处理查询)**
### 4.2 适应性强化
该种过程允许RAG框架灵活地适时选取适当的数据，该过程强调强化流程的**自动化**
<center><img src="imgs/aug3.png"></center>

**经典方法：Self-RAG, AutoGPT, Toolformer, Graph-ToolFormer, WebGPT(融入强化学习以强化GPT3)**

## 5.任务类型与评估

### 5.1 下流任务(Downstream Task)
主要有：
- 问题回答(QA)
- 信息提取(IE)
具体见论文原文 II表格

### 5.2 评估目的

RAG的评估目的主要有如下两个：
1. 检索质量：即检索结果的有效性。该项的指标可套用搜索引擎和推荐系统的评价指标
2. 生成结果质量：LLM是否正确地根据参考信息生成结果

### 5.3 评估方向

| 方向   | 指标名    | 描述                       |
| ---- | ------ | ------------------------ |
| 质量分数 | 内容相关度  | LLM是否参考并善用了检索结果          |
| 质量分数 | 答案忠实度  | LLM是否保持了检索结果真实而不乱生成信息    |
| 质量分数 | 回答相关度  | 答案是否与问题直接在**语义层级上相关**    |
| 必要能力 | 噪声鲁棒性  | RAG系统评估问题相关但缺乏实质信息文档的能力  |
| 必要能力 | 负面拒绝评估 | 检索到文档不含有对应知识时拒绝之的能力      |
| 必要能力 | 信息整合度  | RAG系统整合多来源信息并用于回答复杂问题的能力 |
| 必要能力 | 反事实稳健性 | RAG系统分析文档中不确定之处的能力       |



