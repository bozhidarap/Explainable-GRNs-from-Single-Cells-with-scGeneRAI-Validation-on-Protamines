import scanpy as sc
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple, Union


def load_umi_table(
    path_or_df: Union[str, pd.DataFrame],
    sep: str = "\t",
    index_col: int = 0
) -> sc.AnnData:
    """
    Create AnnData from a UMI count matrix in genes x cells orientation.

    Args:
        path_or_df: File path to a delimited matrix (genes x cells) or a preloaded DataFrame.
        sep: Delimiter used in the text file when a path is provided.
        index_col: Index column for pandas when a path is provided.

    Returns:
        AnnData with cells as observations (rows) and genes as variables (columns).
    """
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, sep=sep, index_col=index_col)
    else:
        df = path_or_df
    adata = sc.AnnData(df.T)
    adata.var_names = adata.var_names.astype(str).str.strip()
    adata.var_names_make_unique()
    return adata


def add_raw_counts_layer(adata: sc.AnnData, layer_name: str = "counts") -> sc.AnnData:
    """
    Store the current X matrix as a raw counts layer.

    Args:
        adata: Input AnnData.
        layer_name: Name of the layer to store counts.

    Returns:
        The same AnnData with counts copied to the specified layer.
    """
    adata.layers[layer_name] = adata.X.copy()
    return adata


def compute_qc_and_filter(
    adata: sc.AnnData,
    mt_prefix: str = "MT-",
    min_counts: int = 1000,
    max_pct_mt: float = 15.0,
    min_cells: int = 10
) -> sc.AnnData:
    """
    Compute QC metrics and filter low-quality cells/genes.

    Args:
        adata: Input AnnData.
        mt_prefix: Prefix used to flag mitochondrial genes (case-insensitive).
        min_counts: Minimum total UMI per cell to retain.
        max_pct_mt: Maximum percent mitochondrial UMIs per cell to retain.
        min_cells: Keep genes expressed in at least this many cells.

    Returns:
        Filtered AnnData.
    """
    adata.var["mt"] = adata.var_names.str.upper().str.startswith(mt_prefix.upper())
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata._inplace_subset_obs((adata.obs["total_counts"] >= min_counts) & (adata.obs["pct_counts_mt"] <= max_pct_mt))
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def normalize_and_log(
    adata: sc.AnnData,
    target_sum: float = 1e4,
    layer_for_log: Optional[str] = None
) -> sc.AnnData:
    """
    Library-size normalize and log1p-transform.

    Args:
        adata: Input AnnData.
        target_sum: Target library size per cell after normalization.
        layer_for_log: If provided, log-transform that layer into X; else log-transform X.

    Returns:
        Normalized and log-transformed AnnData.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata, layer=layer_for_log)
    if layer_for_log is not None:
        adata.X = adata.layers[layer_for_log].copy()
    return adata


def select_hvgs_with_panel(
    adata: sc.AnnData,
    n_top_genes: int = 3000,
    flavor: str = "seurat_v3",
    hvg_layer: str = "counts",
    panel_genes: Optional[Iterable[str]] = None
) -> sc.AnnData:
    """
    Mark HVGs on a specified layer and subset variables to HVGs ∪ panel_genes.

    Args:
        adata: Input AnnData.
        n_top_genes: Number of HVGs to select.
        flavor: HVG selection flavor.
        hvg_layer: Layer to compute HVGs on (typically raw counts).
        panel_genes: Additional genes to always include if present.

    Returns:
        AnnData subset to HVGs and panel genes.
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, layer=hvg_layer, subset=False)
    keep = set(adata.var_names[adata.var["highly_variable"]])
    if panel_genes is not None:
        keep |= {g for g in panel_genes if g in adata.var_names}
    keep = list(keep)
    adata = adata[:, keep].copy()
    return adata


def run_embedding_and_clustering(
    adata: sc.AnnData,
    scale_max: float = 10.0,
    n_neighbors: int = 15,
    n_pcs: int = 40,
    umap_min_dist: float = 0.5,
    leiden_resolution: float = 1.0,
    random_state: int = 0
) -> sc.AnnData:
    """
    Scale, compute PCA, neighborhood graph, UMAP, and Leiden clusters.

    Args:
        adata: Input AnnData.
        scale_max: Cap for scaled expression values.
        n_neighbors: Number of neighbors for graph construction.
        n_pcs: Number of principal components.
        umap_min_dist: UMAP min_dist parameter.
        leiden_resolution: Leiden resolution parameter.
        random_state: Random seed used by PCA/UMAP for reproducibility.

    Returns:
        AnnData with PCA/UMAP/Leiden results.
    """
    sc.pp.scale(adata, max_value=scale_max)
    sc.tl.pca(adata, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    sc.tl.umap(adata, min_dist=umap_min_dist, random_state=random_state)
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden")
    return adata


def quick_umap_plot(
    adata: sc.AnnData,
    color: Sequence[str],
    wspace: float = 0.4,
    show: bool = True,
    save: Optional[str] = None
) -> None:
    """
    Convenience wrapper for UMAP visualization.

    Args:
        adata: Input AnnData with UMAP computed.
        color: List of obs/var names to color by (e.g., ["leiden", "GAPDH"]).
        wspace: Horizontal spacing between panels.
        show: Whether to display the plot.
        save: If provided, a string filename (e.g., "umap.png") to save the figure.
    """
    sc.pl.umap(adata, color=list(color), wspace=wspace, show=show, save=save)


def universal_scrna_pipeline(
    path_or_df: Union[str, pd.DataFrame],
    sep: str = "\t",
    index_col: int = 0,
    mt_prefix: str = "MT-",
    min_counts: int = 1000,
    max_pct_mt: float = 15.0,
    min_cells: int = 10,
    target_sum: float = 1e4,
    n_top_genes: int = 3000,
    hvg_flavor: str = "seurat_v3",
    hvg_layer: str = "counts",
    panel_genes: Optional[Iterable[str]] = ("PRM1", "PRM2", "GAPDH"),
    scale_max: float = 10.0,
    n_neighbors: int = 15,
    n_pcs: int = 40,
    umap_min_dist: float = 0.5,
    leiden_resolution: float = 1.0,
    plot: bool = True,
    plot_genes: Optional[Sequence[str]] = None,
    random_state: int = 0
) -> sc.AnnData:
    """
    End-to-end Scanpy workflow: load, QC, normalize, HVG select (with panel), PCA/UMAP/Leiden, optional plotting.

    Args:
        path_or_df: File path to counts (genes x cells) or a DataFrame of same shape.
        sep: Delimiter when reading from file.
        index_col: Index column when reading from file.
        mt_prefix: Prefix identifying mitochondrial genes.
        min_counts: Minimum total counts per cell.
        max_pct_mt: Maximum mitochondrial percent per cell.
        min_cells: Minimum cells per gene.
        target_sum: Normalization target per cell.
        n_top_genes: Number of HVGs to select.
        hvg_flavor: HVG selection flavor.
        hvg_layer: Layer used to compute HVGs (should be raw counts).
        panel_genes: Extra genes to force-keep if present.
        scale_max: Clipping value in scaling.
        n_neighbors: Neighbors for graph construction.
        n_pcs: Number of PCs.
        umap_min_dist: UMAP min_dist.
        leiden_resolution: Leiden resolution.
        plot: If True, draws a UMAP colored by clusters and selected genes.
        plot_genes: Additional genes to plot alongside "leiden".
        random_state: Seed for reproducibility.

    Returns:
        Processed AnnData object ready for downstream analyses.
    """
    adata = load_umi_table(path_or_df, sep=sep, index_col=index_col)
    add_raw_counts_layer(adata, layer_name=hvg_layer)
    compute_qc_and_filter(adata, mt_prefix=mt_prefix, min_counts=min_counts, max_pct_mt=max_pct_mt, min_cells=min_cells)
    normalize_and_log(adata, target_sum=target_sum)
    adata = select_hvgs_with_panel(adata, n_top_genes=n_top_genes, flavor=hvg_flavor, hvg_layer=hvg_layer, panel_genes=panel_genes)
    run_embedding_and_clustering(adata, scale_max=scale_max, n_neighbors=n_neighbors, n_pcs=n_pcs, umap_min_dist=umap_min_dist, leiden_resolution=leiden_resolution, random_state=random_state)

    if plot:
        to_color = ["leiden"]
        if plot_genes is not None:
            to_color += [g for g in plot_genes if g in adata.var_names]
        elif panel_genes is not None:
            to_color += [g for g in panel_genes if g in adata.var_names]
        quick_umap_plot(adata, color=to_color, wspace=0.4, show=True, save=None)

    return adata
