---
title: "Accurate Antibody-Antigen Interface Quality Assessment via Geometric Vector Perceptrons"
collection: Projects
type: "Machine Learning"
permalink: /Projects/antibody-antigen-gvp-ranking.md
date: 2026-01-16
excerpt: "Bridge the 'ranking gap' in protein folding engines (e.g., AlphaFold 3) using a GVP-based geometric deep learning framework. Improved Top-1 success rate from 35% to 45%."
header:
  teaser: /projects/constructure.png
tags:
  - Geometric Deep Learning
  - Structural Bioinformatics
  - Pairwise Ranking
  - PyTorch
tech_stack:
  - name: Python
  - name: PyTorch Geometric (PyG)
  - name: GVP-GNN
  - name: AlphaFold3/Protenix
---

## Project Background
Current structural generators like AlphaFold 3 can sample near-native conformations but often struggle with the **"ranking gap"**—where internal confidence metrics (pLDDT/pAE) fail to prioritize the best structures. 

This project develops a specialized **Model Quality Assessment (MQA)** tool using **Geometric Vector Perceptrons (GVP)**. By processing 3D protein interfaces as graphs and learning from a pairwise ranking objective, the model identifies high-fidelity antibody-antigen complexes from stochastic ensembles.

## Key Technical Contributions
*   **Geometric Representation**: Represented residues as nodes in a KNN-graph ($k=30$) with rotationally invariant scalar features (dihedral angles, confidence) and equivariant vector features (backbone orientations).
*   **Weighted Global Pooling**: Implemented a spatial-aware pooling strategy that assigns **10x higher weight** to interface-defined residues, focusing the model on critical binding signatures.
*   **Pairwise Ranking Strategy**: Trained a Siamese architecture to optimize the relative ranking of structural decoys using `BCEWithLogitsLoss` on the score difference ($s_1 - s_2$).
*   **Data Strategy**: Curated a dataset of 2,530 post-cutoff complexes from SAbDab, ensuring zero data leakage by partitioning at the Complex ID level.

## Core Model Architecture (GVP-GNN)
The core architecture captures biological geometry by simultaneous message passing of scalar and vector signals.

```python
# Refined from models.py: GVP-based Quality Assessment Model
class ComplexQualityModel(nn.Module):
    def __init__(self, node_in_dim=(9, 3), node_h_dim=(100, 16), num_layers=3):
        super().__init__()
        # Initialize GVP Embeddings for nodes and edges
        self.W_v = nn.Sequential(LayerNorm(node_in_dim), GVP(node_in_dim, node_h_dim))
        self.W_e = nn.Sequential(LayerNorm(edge_in_dim), GVP(edge_in_dim, edge_h_dim))

        # Message Passing Layers
        self.layers = nn.ModuleList([
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=0.1) for _ in range(num_layers)
        ])

        # Interface-Weighted Readout
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), GVP(node_h_dim, (node_h_dim[0], 0)))
        self.regression_head = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(), nn.Linear(2*ns, 1)
        )

    def forward(self, batch):
        h_V = self.W_v((batch.node_s, batch.node_v))
        h_E = self.W_e((batch.edge_s, batch.edge_v))
        
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        # Apply 10x weight to interface residues for global pooling
        out = self.W_out(h_V)
        weighted_out = out * (batch.mask.view(-1, 1) * 10.0 + 1.0)
        graph_feats = scatter_mean(weighted_out, batch.batch, dim=0)
        return self.regression_head(graph_feats)
