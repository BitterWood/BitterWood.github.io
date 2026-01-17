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
share: false
mathjax: true
---

## 1. Project Overview

### 1.1 Problem Statement
While state-of-the-art folding engines such as AlphaFold 3 and Protenix have demonstrated the capability to sample near-native conformations of antibody-antigen complexes, they frequently suffer from a **"ranking gap"**. Internal confidence metrics (e.g., predicted LDDT) often fail to identify the best-predicted structures within a stochastic ensemble. This discrepancy between generation and selection accuracy limits the effectiveness of current pipelines in therapeutic design.

### 1.2 Proposed Solution
This study proposes a geometric deep learning framework utilizing **Geometric Vector Perceptrons (GVPs)** to bridge the selection gap by discriminating high-quality binding interfaces. The model integrates rotationally invariant scalar features with equivariant vector features to capture the subtle geometric and physicochemical signatures of interactions. Trained on a curated dataset of 2,530 complexes via a pairwise ranking objective, the solution effectively re-ranks decoys to identify near-native experimental conformations.

## 2. Methodology and Architecture

### 2.1 Geometric Representation
Protein structures are transformed into a graph representation where residues serve as nodes and edges connect the $$k$$-nearest neighbors ($$k=30$$). We implemented a **Symmetry Correction** protocol where all antibody-antigen pairs are canonicalized based on alphabetical chain ordering (e.g., consistently saved in "AB" format) to deduplicate mirror-image relationships. Each residue is characterized by:
*   **Scalar Features ($$s \in \mathbb{R}^9$$)**: Dihedral angles ($$\cos, \sin$$), per-residue confidence scores (pLDDT), and binary interface masks.
*   **Vector Features ($$v \in \mathbb{R}^{3 \times 3}$$)**: Capturing local backbone geometry and orientation.
*   **Edge Features**: Scalar RBF embeddings ($$s \in \mathbb{R}^{33}$$) and displacement vectors ($$v \in \mathbb{R}^{1 \times 3}$$).

The dataset, comprising 11,868 JSON files, was partitioned using a 4-to-1 ratio at the **Complex ID level** (9,566 training, 2,302 validation) to ensure the model generalizes to unseen assemblies without information leakage from identical complex variations.

### 2.2 GVP-GNN Architecture
The model employs a GVP-based message-passing mechanism followed by a specialized **Interface-Weighted Global Pooling** strategy, which assigns a 10-fold higher weight to interface residues to focus on the binding boundary.

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
        # Interface-Weighted Global Pooling (10-fold weight for interaction boundary)
        weighted_out = out * (batch.interface_mask.view(-1, 1) * 10.0 + 1.0)
        graph_feats = scatter_mean(weighted_out, batch.batch, dim=0)
        return self.regression_head(graph_feats)
```

## 3. Performance and Evaluation

### 3.1 Training Convergence
The model was optimized using `BCEWithLogitsLoss` within a Siamese architecture to distinguish relative structural quality. Training was conducted for 100 epochs on an NVIDIA RTX 4090D GPU using 26,328 training pairs. The validation pairwise ranking accuracy reached a stable plateau of **64.0%**, indicating robust convergence and successful learning of discriminative features.

![Loss and accuracy curves of training and validation sets](https://bitterwood.github.io/images/learning_curves.png)

### 3.2 Correlation with Ground Truth
Evaluation using Spearman’s rank correlation against ground-truth DockQ scores revealed a strong monotonic relationship. The model achieved a **median correlation of 0.405** and a **mean correlation of 0.340** per interface. Statistical analysis showed that the model successfully interprets the full spectrum of structural quality, from non-specific arrangements to high-quality native-like hits.

![Distribution of spearman correlation coefficients between predicted and true DockQ values](https://bitterwood.github.io/images/spearman_distribution.png)

### 3.3 Improving Selection Success Rate
In a re-ranking task of stochastic ensembles with $$N=8$$ seeds, the GVP-discriminator significantly improved the **Top-1 Success Rate ($$DockQ \ge 0.23$$)**. While the baseline generator ranking provided a success rate of ~35.0%, our model improved this to approximately **45.0%**, effectively recovering nearly half of the potential performance gain bounded by the theoretical Oracle.

![Success rate of Top 1 structure for different strategies](https://bitterwood.github.io/images/success_rate_by_seed.png)

## 4. Technical Impact

### 4.1 Research Contribution
This work quantifies the "selection error" in modern folding engines. By successfully identifying "Acceptable" or "Medium" quality structures that are sampled but overlooked by internal scorers, the model mitigates a critical bottleneck in computational antibody design. The rigorous partitioning at the Complex ID level ensures that these performance gains are representative of true generalization to novel protein interfaces.

### 4.2 Practical Application
By explicitly processing rotationally equivariant vector features, the model captures essential physical characteristics such as the relative orientation of $$V_H/V_L$$ chains and the directionality of CDR loops. This provides a robust tool for identifying near-native antibody-antigen complexes in generative pipelines, reducing the downstream experimental validation costs in therapeutic discovery.
