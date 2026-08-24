# CoDiFormer: Multi-Stage Semantic Quantization and Adaptive Continuous-Discrete Fusion for Graph Node Classification

Official implementation of **CoDiFormer**, a graph neural network framework that integrates continuous node representations with discrete vector-quantized representations for robust node classification.

## Overview

Graph neural networks (GNNs) usually learn continuous node embeddings for downstream prediction tasks. However, continuous representations may struggle to capture discrete structural patterns and latent semantic clusters in complex graphs.

CoDiFormer introduces a continuous-to-discrete representation learning framework by combining:

- Continuous node representation learning
- Residual Vector Quantization (RVQ)
- Discrete node identity representation
- Knowledge distillation between continuous and discrete branches

The framework improves node classification performance by jointly exploiting continuous semantic information and discrete structural patterns.


## Requirements

The implementation is based on:

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4

Install dependencies:

```bash
pip install -r requirements.txt
