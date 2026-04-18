---
title: "Vector DB Selection & Design" 
date: 2025-12-09 
draft: true
tags: ["RAG", "Vector DB"] 
categories: ["inspiration"]
---



## Why PostgreSQL

- Smooth learning curve: familiarity brings smoother development and use of PostgreSQL features
- Affordable delay & Source of Truth: I usually need more than a million chunks to have the demand for professional vector database. For personal use and deployment, PostgreSQL+ pgvector is affordable. PostgreSQL can be the source of truth while pgvector helps with approximate searching.
- Allows custom distance measuring(BM25, cosine similarity, TF-IDF and hybrids of all above)
- Suitable for dehydrate & hydrate design. Make the system focuses on retrieving based on plain text



## Problem Setup

Let a chunk in the database be $C$. Then $C$ may have the following part:

- Main text $t$, usually a description or definition. This part is crucial for searching
- Content important for answering but harmful for searching $c_{imp}$
- Noise in the chunk $c_n$

There is :

$$
C = (t, \{c_{imp}\}_{i=1}^{m}, \{c_n\}_{j=1}^{k})
$$


This project uses a dehydrate & hydrate functionality to focus on $t$ , use $c\_{imp}$ only when querying and remove $c_n$ .