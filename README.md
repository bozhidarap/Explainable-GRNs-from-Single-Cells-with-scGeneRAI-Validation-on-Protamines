# End-to-End Gene Regulatory Network Inference Pipeline using Optimized scGeneRAI

## Overview
Protamine 1 (**PRM1**) and protamine 2 (**PRM2**) are essential for packaging sperm DNA and maintaining genome stability during spermatogenesis. Alterations in their expression levels have been associated with male infertility, highlighting the importance of understanding the regulatory mechanisms controlling these genes.

This project investigates protamine regulation using an **end-to-end computational pipeline** that integrates transcriptomic preprocessing with **gene regulatory network (GRN) inference** based on an optimized implementation of **scGeneRAI**. The pipeline identifies candidate transcription factors that may regulate **PRM1** and **PRM2**, providing insights into the transcriptional control of spermatogenesis.

Among the predicted regulators, **HMGB4** and **TBPL1** have previously been linked to chromatin regulation and germ cell development, supporting the biological relevance of the inferred interactions. The analysis also identified **CREM**, a well-established regulator of spermatogenesis, indicating the pipeline’s ability to recover high-confidence regulatory candidates.

To further evaluate the robustness of the predictions, results were compared with those obtained using other GRN inference methods, including **GENIE3**, **GRNBoost2**, and **CLR**. Transcription factors consistently identified across methods represent promising candidates for future experimental validation and fertility-related studies.

Overall, the pipeline provides an efficient and interpretable framework for studying gene regulation in spermatogenesis. Its targeted design enables GRN inference with minimal computational resources and allows application to diverse **scRNA-seq datasets** across tissues and cell types.


---

## Contents

| File / Folder | Description |
|----------------|-------------|
| `scGeneRAI.py` | Core implementation of the optimized scGeneRAI model with relevance propagation (LRP) for GRN inference. |
| `data_preprocessing.py` | Data preprocessing with QC, normalization, and HVG selection (including PRM1/PRM2). |
| `train_and_predict.py` | scGeneRAI-based model fitting and LRP network extraction utilities. |
| `bootstrap_stability.py` | Statistical module for bootstrap-based stability analysis of inferred regulatory edges. |
| `existing_methods/` | Implementations of commonly used GRN inference methods for comparison. |
| `existing_methods/GRNboost2.ipynb` | Notebook implementing GRNBoost2 for transcription factor–target inference. |
| `existing_methods/GENIE3.ipynb` | Notebook implementing the GENIE3 random forest–based GRN inference method. |
| `existing_methods/CLR_GRN.ipynb` | Notebook implementing the Context Likelihood of Relatedness (CLR) method for GRN reconstruction. |
---

## Pipeline Summary

1. **Preprocessing with Scanpy**
   - Performs filtering, normalization, log transformation, and highly variable gene selection.
   - Retains a specified gene panel of interest (e.g., PRM1, PRM2).

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
