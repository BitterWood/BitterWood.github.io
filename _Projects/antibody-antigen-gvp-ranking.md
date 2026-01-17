---
title: "Accurate Antibody-Antigen Interface Quality Assessment via Geometric Vector Perceptrons"
collection: Projects
type: "Machine Learning / Structural Bioinformatics"
permalink: /Projects/antigen-antibody-gvp-ranking
date: 2026-01-16
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
State-of-the-art generators like AlphaFold 3 and Protenix often sample near-native conformations but suffer from a **"ranking gap"**: internal confidence metrics (such as pLDDT) frequently fail to prioritize the best-predicted structures from a stochastic ensemble. This discrepancy between generation and selection leads to suboptimal results in structural modeling workflows where high-quality decoys are sampled but not identified.

### 1.2 Proposed Solution
This project implements a **Geometric Vector Perceptron (GVP)** framework designed to discriminate high-quality binding interfaces. By capturing rotationally invariant scalar features and rotationally equivariant vector features, the model effectively re-ranks decoys to identify biologically accurate structures. The solution employs a Siamese architecture trained on a curated dataset of 2,530 complexes to bridge the gap between structural sampling and selection accuracy.

## 2. Methodology and Architecture

### 2.1 Geometric Representation
We transform the 3D protein structure into a KNN-graph ($k=30$). Data preprocessing includes a symmetry correction protocol where all antigen-antibody pairs are canonicalized based on alphabetical chain ordering (e.g., "AB" format) to ensure a unique representation. Each residue is a node with:
*   **Scalar Features ($s \in \mathbb{R}^9$)**: Dihedral angles ($\cos, \sin$), pLDDT, and interface masks.
*   **Vector Features ($v \in \mathbb{R}^{3 \times 3}$)**: Backbone orientations and local geometric vectors.
*   **Edge Features**: Scalar RBF embeddings ($s \in \mathbb{R}^{33}$) and displacement vector features ($v \in \mathbb{R}^{1 \times 3}$).

The dataset was partitioned using a 4-to-1 ratio at the **Complex ID level** (9,566 training, 2,302 validation files) to prevent information leakage and ensure generalization to unseen assemblies.

### 2.2 GVP-GNN Architecture
The model uses a Siamese architecture with three GVP calculation layers. A specialized **Interface-Weighted Global Pooling** strategy is implemented to focus the model on the interaction boundary by assigning 10-fold higher weights to interface residues.

```python
# Core Model: GVP-based Quality Assessment
class ComplexQualityModel(nn.Module):
    def __init__(self, node_in_dim=(9, 3), node_h_dim=(100, 16), num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=0.1) 
            for _ in range(num_layers)
        ])
        # Output GVP for scalar projection
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), GVP(node_h_dim, (1, 0)))

    def forward(self, batch):
        # ... GVP Message Passing ...
        out = self.W_out(h_V)
        # Interface-Weighted Global Pooling
        weighted_out = out * (batch.interface_mask.view(-1, 1) * 10.0 + 1.0)
        graph_feats = scatter_mean(weighted_out, batch.batch, dim=0)
        return self.regression_head(graph_feats)
```

## 3. Performance and Evaluation

### 3.1 Training Convergence
The model was optimized using `BCEWithLogitsLoss` against ground-truth labels derived from DockQ scores. Trained on an NVIDIA RTX 4090D for 100 epochs, the validation pairwise ranking accuracy reached a stable plateau of **64.0%**, demonstrating reliable generalization without overfitting.

![Loss and accuracy curves of training and validation sets](/images/Projects/antigen_antibody_training_curves.png)

### 3.2 Correlation with Ground Truth
We evaluated the model using Spearman’s rank correlation against objective DockQ scores. The model achieved a **median correlation of 0.405** and a **mean correlation of 0.340** per interface, confirming a strong monotonic relationship between the predicted quality scores and physical structural accuracy.

![Distribution of spearman correlation coefficients between predicted and true DockQ values](/images/Projects/antigen_antibody_correlation.png)

### 3.3 Improving Selection Success Rate
The GVP-discriminator bridges the performance gap in generative pipelines. In an ensemble of $N=8$ seeds, the model improved the **Top-1 Success Rate (DockQ $\ge$ 0.23)** from a baseline of ~35.0% (internal rankings) to approximately **45.0%**. This effectively recovers nearly half of the potential performance gain bounded by the theoretical Oracle.

![Success rate of Top 1 structure for different strategies](/images/Projects/antigen_antibody_success_rate.png)

## 4. Technical Impact

### 4.1 Research Contribution
This work quantifies and addresses the critical "selection error" in stochastic folding engines. By successfully identifying "Acceptable" or "Medium" quality structures that are sampled but overlooked by internal metrics, the tool reduces the experimental validation costs inherent in antibody discovery.

### 4.2 Practical Application
By leverage equivariant vector features, the model captures the relative orientation of $V_H$/$V_L$ chains and the directionality of CDR loops. This geometric deep learning framework ensures that downstream biological modeling is based on the most physically plausible protein-protein interfaces, enhancing the reliability of computational drug discovery.
