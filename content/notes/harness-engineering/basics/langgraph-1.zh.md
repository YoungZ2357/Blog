---
title: "LangGraph设计初衷"
date: 2026-05-13
draft: false
math: true
mermaid: true
tags: ["harness engineering"]
categories: ["LLM"]
---
> 该目录笔记参考自LangChain Academy 课程[^1]

使用LLM时，可使用一个链条串联工作步骤。该结构特点为可靠（Anthropic将其定义为Workflow而非Agent[^2]）
```mermaid
flowchart LR
A(LLM)
B[步骤1]
C[步骤2]
D[...]
E[步骤$$n$$]

A --> B
B --> C
C --> D
D --> E


```

然而，有时可以让LLM拥有一定自主权，可以让LLM自行选择步骤，该操作能提升Agent的灵活性（亦称Router）
```mermaid
flowchart LR
A(LLM)
B[步骤1]
C[步骤2]
D[步骤3-1]
E[步骤3-2]

A --> B
B --> C
C --> D
C --> E
```

更极端一点，可以让LLM完全自主，直接和环境交互（Autonomous Agent亦称Fully Autonomous Agent，前者与Workflow的对比，强调Agent自主权）
```mermaid
flowchart TD

A(LLM)
B[Steps]

A --> A
A --> B
B --> A
```

随着LLM的自主性提升，Agent的可靠性会逐渐下降。
- LangGraph旨在提供一种于可靠性和灵活性之间trade-off方案。

可以基于该点，开发者可以先定义固定的工作流程，并有限地暴露部分自主权给Agent
```mermaid
flowchart LR
A(LLM)
B[步骤1]
C("LLM(自主化行为)")
D[步骤$$n$$]

A --> B
B --> C

C ==> B
C --> D
```


## 参考资料

[^1]: LangChain Academy. [GitHub](https://github.com/langchain-ai/langchain-academy)  [Course: Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph)

[^2]: [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
