## Context Pooling: Novel Query-specific Graph Pooling to Identify Logically Relevant Neighbors for Inductive Link Prediction in Knowledge Graphs

This is the repository for IJCAI anonymous submission **Context Pooling: Novel Query-specific Graph Pooling to Identify Logically Relevant Neighbors for Inductive Link Prediction in Knowledge Graphs**.

In this paper, we introduce a novel method, named Context Pooling, to enhance GNN-based models' efficacy for link predictions in KGs. To our best of knowledge, Context Pooling is the first methodology that applies graph pooling in KGs. 
Additionally, Context Pooling is first-of-its-kind to enable the generation of query-specific graphs for inductive settings, where testing entities are unseen during training.

![fig](https://github.com/IJCAI2024AnonymousSubmission/Context-Pooling/blob/master/fig.png)

## Requirements

- networkx==2.5
- numpy==1.21.5
- ray==2.6.3
- scipy==1.8.1
- torch==1.13.0+cu116
- torch_scatter==2.0.9

## Quick Start

This repository contains the implementation of `RED-GNN+DP`, which is our distinctive pooling architecture based on `RED-GNN`(https://github.com/LARS-research/RED-GNN).

For transductive and inductive link prediction, we've set the default parameters in `main.py` in the respective folders. Please train and test using:
```shell
./train.sh
```

If you want to add a new dataset and fine-tune parameters by yourself. Please use:
```shell
./tuning.sh
```

## Unapproximated version

We have also provided the code for unapproximated version of distinctive pooling, please switch to branch `unapproximated`.

## Citations

Currently not available.

## Q&A

For any questions, feel free to leave an issue.
Thank you very much for your attention and further contribution :)
