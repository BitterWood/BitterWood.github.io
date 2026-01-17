---
title: "Accurate Antibody-Antigen Interface Quality Assessment via Geometric Vector Perceptrons"
collection: Projects
type: "Machine Learning"
permalink: /Projects/antigen-antibody-gvp-ranking
date: 2024-05-20
excerpt: "Bridge the 'ranking gap' in AlphaFold 3 using a GVP-based geometric deep learning framework. Improved Top-1 success rate from 35% to 45%."
tags:
  - Geometric Deep Learning
  - Structural Bioinformatics
  - Pairwise Ranking
  - PyTorch
tech_stack:
  - name: Python
  - name: PyTorch Geometric (PyG)
  - name: GVP-GNN
  - name: AlphaFold3 Ecosystem
---

## 1. Project Overview

### 1.1 Problem Statement
State-of-the-art generators like AlphaFold 3 often sample near-native conformations but suffer from a **"ranking gap"**: internal confidence metrics (pLDDT) fail to prioritize the best-predicted structures from an ensemble. 

### 1.2 Proposed Solution
This project implements a **Geometric Vector Perceptron (GVP)** framework to discriminate high-quality binding interfaces. By capturing rotationally invariant scalar and equivariant vector features, the model effectively re-ranks decoys to identify biologically accurate structures.

## 2. Methodology and Architecture

### 2.1 Geometric Representation
We transform the 3D protein structure into a KNN-graph ($k=30$). Each residue is a node with:
*   **Scalar Features**: Dihedral angles ($\cos, \sin$), pLDDT, and interface masks.
*   **Vector Features**: Backbone orientations and side-chain vectors.

### 2.2 GVP-GNN Architecture
The model uses a Siamese architecture to learn from structural pairs. 

```python
# Core Model: GVP-based Quality Assessment
class ComplexQualityModel(nn.Module):
    def __init__(self, node_in_dim=(9, 3), node_h_dim=(100, 16), num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=0.1) for _ in range(num_layers)
        ])
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0)))

    def forward(self, batch):
        out = self.W_out(h_V)
        weighted_out = out * (batch.mask.view(-1, 1) * 10.0 + 1.0)
        graph_feats = scatter_mean(weighted_out, batch.batch, dim=0)
        return self.regression_head(graph_feats)
```

## 3. Performance and Evaluation

### 3.1 Training Convergence
The model was trained using a pairwise ranking objective ($s_1 - s_2$ optimized via BCE Loss).

![Learning Convergence](/images/Projects/antigen_antibody_training_curves.png)

### 3.2 Correlation with Ground Truth
We evaluated the model using Spearman’s rank correlation against objective DockQ scores to measure ranking consistency.

![Spearman Correlation](/images/Projects/antigen_antibody_correlation.png)

### 3.3 Improving Selection Success Rate
The GVP-discriminator bridges the performance gap in generative pipelines by selecting the highest-quality decoys.

![Success Rate Growth](/images/Projects/antigen_antibody_success_rate.png)

## 4. Technical Impact

### 4.1 Research Contribution
By accurately identifying "Acceptable" quality structures from stochastic outputs, this tool reduces the experimental costs of antibody discovery and therapeutic design.

### 4.2 Practical Application
This geometric deep learning framework ensures that downstream biological modeling is based on the most physically plausible protein-protein interfaces, improving the reliability of computational drug discovery.
