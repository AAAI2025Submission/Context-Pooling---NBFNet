## Context Pooling: Query-specific Graph Pooling for Generic Inductive Link Prediction in Knowledge Graphs



This is the repository for AAAI anonymous submission **Context Pooling: Query-specific Graph Pooling for Generic Inductive Link Prediction in Knowledge Graphs**.

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

This repository contains the implementation of `NBFNet+CP`, which is our Context Pooling architecture based on [`NBFNet`](https://github.com/KiddoZhu/NBFNet-PyG).

For the implementation of `RED-GNN+CP`, please refer it [here](https://github.com/AAAI2025Submission/Context-Pooling).

For transductive and inductive link prediction, we've set the default parameters in `main.py` in the respective folders. Please train and test using:
```shell
./train.sh
```

If you want to add a new dataset and fine-tune parameters by yourself. Please use:
```shell
./tuning.sh
```

For parameters in `NBFNet`, please refer to the `*.yaml` files in the `config\` folders.
For parameters in `Context Pooling`, please adjust them in `train.sh` and `tuning.sh`.


## Citations

Currently not available.

## Q&A

For any questions, feel free to leave an issue.
Thank you very much for your attention and further contribution :)
