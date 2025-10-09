# End-to-End Gene Regulatory Network Inference Pipeline using Optimized scGeneRAI

## Overview

Understanding the regulation of protamine genes is essential for insights into spermatogenesis and male fertility.  
This repository provides an **end-to-end pipeline** for **gene regulatory network (GRN)** inference built on an optimized version of the **scGeneRAI** method.  
The pipeline integrates single-cell data preprocessing, neural network–based inference of transcriptional regulation, and bootstrap-based stability assessment.  

A case study on **PRM1** and **PRM2** demonstrates how the method identifies putative transcriptional regulators, several of which are supported by published evidence.  
The approach produces interpretable, testable hypotheses for experimental validation and contributes a practical framework for GRN discovery in fertility research.

---

## Contents

| File / Folder | Description |
|----------------|-------------|
| `scGeneRAI.py` | Core implementation of the optimized scGeneRAI model with relevance propagation (LRP) for GRN inference. |
| `scanpy_pipeline.py` | Data preprocessing module using Scanpy, including quality control, normalization, HVG selection, PCA, UMAP, and clustering. |
| `fit_predict_edges.py` | High-level wrapper to train scGeneRAI, extract TF→target edges, and aggregate relevance scores. |
| `bootstrap_stability.py` | Statistical module for bootstrap-based stability analysis of inferred regulatory edges. |

---

## Pipeline Summary

1. **Preprocessing with Scanpy**
   - Performs filtering, normalization, log transformation, and highly variable gene selection.
   - Retains a specified gene panel of interest (e.g., PRM1, PRM2, GAPDH).
   - Computes PCA, nearest neighbors, UMAP embedding, and Leiden clusters.

2. **Model Training and Network Inference**
   - Uses an optimized scGeneRAI model to infer TF→target regulatory relationships.
   - Targets can be specified to reduce runtime and focus inference on specific genes.
   - Produces relevance-weighted edge lists based on Layer-wise Relevance Propagation (LRP).

3. **Bootstrap Stability Analysis**
   - Repeats training on resampled cell subsets.
   - Quantifies Top-K Jaccard similarity and edge frequency to assess reproducibility.
   - Identifies robust TF→target edges across resamples.

---

## Installation

Requirements:
- Python 3.9 or later  
- PyTorch ≥ 2.0  
- Scanpy ≥ 1.9  
- pandas, numpy, tqdm, scipy, matplotlib

Install with pip:

```bash
pip install torch scanpy pandas numpy tqdm scipy matplotlib
