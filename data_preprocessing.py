import scanpy as sc
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple, Union
from urllib.request import urlretrieve


# ----------------------------
# 1) Loading (cells x genes)
# ----------------------------
def load_umi_table_cells_x_genes(
    path_or_df: Union[str, pd.DataFrame],
    sep: str = "\t",
    index_col: int = 0
) -> sc.AnnData:
    """
    Create AnnData from a UMI count matrix in cells x genes orientation.

    Args:
        path_or_df: File path to a delimited matrix (cells x genes) or a preloaded DataFrame.
        sep: Delimiter used in the text file when a path is provided.
        index_col: Index column for pandas when a path is provided.

    Returns:
        AnnData with cells as observations (rows) and genes as variables (columns).
    """
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, sep=sep, index_col=index_col)
    else:
        df = path_or_df

    adata = sc.AnnData(df)  # already cells x genes

    # Clean / unique gene names
    adata.var_names = adata.var_names.astype(str).str.strip()
    adata.var_names_make_unique()

    return adata


def add_raw_counts_layer(adata: sc.AnnData, layer_name: str = "counts") -> sc.AnnData:
    """Store the current X as raw counts layer."""
    adata.layers[layer_name] = adata.X.copy()
    return adata


# ----------------------------
# 2) QC + filtering
# ----------------------------
def compute_qc_and_filter(
    adata: sc.AnnData,
    mt_prefix: str = "MT-",
    min_counts: int = 1000,     # exclude cells with <1000 UMI
    min_genes: int = 200,       # exclude cells with <200 genes
    max_pct_mt: float = 15.0,   # exclude cells with >15% mitochondrial
    min_cells: int = 10         # exclude genes detected in <10 cells
) -> sc.AnnData:
    """
    QC metrics and filtering according to:
      - total_counts >= 1000
      - n_genes_by_counts >= 200
      - pct_counts_mt <= 15
      - genes expressed in >= 10 cells
    """
    adata.var["mt"] = adata.var_names.str.upper().str.startswith(mt_prefix.upper())
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    cell_mask = (
        (adata.obs["total_counts"] >= min_counts) &
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["pct_counts_mt"] <= max_pct_mt)
    )
    adata = adata[cell_mask, :].copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)

    return adata


# ----------------------------
# 3) Normalize + log
# ----------------------------
def normalize_and_log(
    adata: sc.AnnData,
    target_sum: float = 1e4
) -> sc.AnnData:
    """
    Normalize each cell to target_sum UMIs and log1p-transform into adata.X.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


# ----------------------------
# 4) HVG + keep PRM1/2
# ----------------------------
def select_hvgs_and_targets(
    adata: sc.AnnData,
    n_top_genes: int = 3000,
    flavor: str = "seurat_v3",
    hvg_layer: str = "counts",
    targets: Sequence[str] = ("PRM1", "PRM2"),
) -> sc.AnnData:
    """
    Select top HVGs (computed on raw counts layer by default) and ensure PRM1/PRM2 are kept if present.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        layer=hvg_layer,
        subset=False
    )

    hvg_set = set(adata.var_names[adata.var["highly_variable"]])
    keep = set(hvg_set)

    # Force-keep targets if present
    keep |= {g for g in targets if g in adata.var_names}

    keep = [g for g in keep if g in adata.var_names]
    adata = adata[:, keep].copy()
    return adata


# ----------------------------
# 5) TF list + final matrix (TFs + PRM1/2)
# ----------------------------
def download_human_tf_list(
    url: str = "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
    local_path: str = "hs_tfs_hg38.txt"
) -> Sequence[str]:
    """
    Download and load a human TF list (hg38) from Aerts lab resources.
    """
    urlretrieve(url, local_path)
    tf_list = (
        pd.read_csv(local_path, header=None)[0]
        .astype(str)
        .str.strip()
        .tolist()
    )
    return tf_list


def build_expr_matrix_for_model(
    adata: sc.AnnData,
    tf_list: Sequence[str],
    targets: Sequence[str] = ("PRM1", "PRM2"),
    zscore: bool = True
) -> pd.DataFrame:
    """
    Build final expression matrix: cells x (TFs + PRM1/PRM2).
    Uses adata.X (log-normalized).
    Optionally applies per-gene z-score.
    """
    X = pd.DataFrame(
        adata.X.A if hasattr(adata.X, "A") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    tfs_in_data = [t for t in tf_list if t in X.columns]
    keep_cols = sorted(set(tfs_in_data).union(targets))
    keep_cols = [c for c in keep_cols if c in X.columns]

    missing_targets = [g for g in targets if g not in keep_cols]
    if missing_targets:
        raise ValueError(f"Missing targets after filtering: {missing_targets}")

    expr_sub = X.loc[:, keep_cols].copy()

    if zscore:
        expr_sub = (expr_sub - expr_sub.mean(axis=0)) / expr_sub.std(axis=0).replace(0, np.nan)
        expr_sub = expr_sub.fillna(0.0)

    return expr_sub


# ----------------------------
# 6) End-to-end pipeline
# ----------------------------
def universal_scrna_pipeline_for_tf_plus_protamines(
    path_or_df: Union[str, pd.DataFrame],
    sep: str = "\t",
    index_col: int = 0,
    mt_prefix: str = "MT-",
    min_counts: int = 1000,
    min_genes: int = 200,
    max_pct_mt: float = 15.0,
    min_cells: int = 10,
    target_sum: float = 1e4,
    n_top_genes: int = 3000,
    hvg_flavor: str = "seurat_v3",
    hvg_layer: str = "counts",
    targets: Sequence[str] = ("PRM1", "PRM2"),
    tf_list_url: str = "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
    tf_list_local_path: str = "hs_tfs_hg38.txt",
    zscore: bool = True,
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Dataset-specific pipeline:

    - Input table is cells x genes
    - Remove cells with:
        * < 1000 UMI
        * < 200 genes
        * > 15% mitochondrial
    - Remove genes detected in < 10 cells
    - Normalize to 10,000 UMI per cell + log1p
    - Select top 3,000 HVGs
    - Finally keep ONLY: Transcription factors + PRM1/PRM2 (for the model)
    - Return:
        * processed AnnData (after HVG/targets subsetting)
        * final matrix expr_sub (cells x (TFs + PRM1/PRM2)), optionally z-scored
    """
    # 1) Load (cells x genes)
    adata = load_umi_table_cells_x_genes(path_or_df, sep=sep, index_col=index_col)

    # 2) Save raw counts for HVG
    add_raw_counts_layer(adata, layer_name=hvg_layer)

    # 3) QC + filtering (cells + genes)
    adata = compute_qc_and_filter(
        adata,
        mt_prefix=mt_prefix,
        min_counts=min_counts,
        min_genes=min_genes,
        max_pct_mt=max_pct_mt,
        min_cells=min_cells
    )

    # 4) Normalize to 10,000 UMI + log1p
    adata = normalize_and_log(adata, target_sum=target_sum)

    # 5) HVG (3000) + ensure PRM1/PRM2
    adata = select_hvgs_and_targets(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
        hvg_layer=hvg_layer,
        targets=targets
    )

    # 6) Download TF list and create final model matrix: TFs + PRM1/PRM2
    tf_list = download_human_tf_list(url=tf_list_url, local_path=tf_list_local_path)
    expr_sub = build_expr_matrix_for_model(adata, tf_list=tf_list, targets=targets, zscore=zscore)

    print(f"After HVG/targets: adata shape = {adata.shape} (cells x genes)")
    print(f"Final model matrix shape: {expr_sub.shape} (cells x (TFs + targets))")

    return adata, expr_sub


# ----------------------------
# Example usage:
# ----------------------------
# adata, expr_sub = universal_scrna_pipeline_for_tf_plus_protamines(
#     "/content/Combined_UMI_table.txt",
#     sep="\t",
#     index_col=0,
#     targets=("PRM1", "PRM2"),
#     zscore=True
# )
# print(expr_sub.columns[:10])

