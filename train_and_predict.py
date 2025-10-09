import os
import glob
import shutil
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def detect_device(prefer: Optional[str] = None) -> str:
    '''
    Choose a computation device string.

    Args:
        prefer: Optional explicit device ('cuda', 'cpu'). If None, uses CUDA if available.

    Returns:
        Device string compatible with scGeneRAI.fit/predict_networks.
    '''
    if prefer in ('cuda', 'cpu'):
        return prefer
    return 'cuda' if _HAS_CUDA else 'cpu'


def _read_lrp_csvs(outdir: str) -> pd.DataFrame:
    '''
    Load and concatenate LRP CSV files produced by scGeneRAI.predict_networks.

    Args:
        outdir: Directory passed to predict_networks (it may contain files directly
                or under a "results" subfolder).

    Returns:
        Concatenated DataFrame with at least ['source_gene','target_gene','LRP'] if present.
        Returns empty DataFrame if nothing was readable.
    '''
    patterns = [
        os.path.join(outdir, "LRP_*.csv"),
        os.path.join(outdir, "results", "LRP_*.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["source_gene", "target_gene", "LRP"])
        except Exception:
            continue
        missing_cols = {"source_gene", "target_gene", "LRP"} - set(d.columns)
        if missing_cols:
            continue
        dfs.append(d)
    if not dfs:
        return pd.DataFrame(columns=["source_gene", "target_gene", "LRP"])
    return pd.concat(dfs, ignore_index=True)


def fit_predict_edges(
    X_df: pd.DataFrame,
    tag: str,
    tfs_in_data: Iterable[str],
    targets: Iterable[str],
    epochs: int,
    depth: int,
    device: Optional[str] = None,
    early_stopping: bool = True,
    keep_abs: bool = True,
    remove_self_loops: bool = True,
    cleanup: bool = True,
    out_root: str = ".",
    lr: float = 2e-2,
    batch_size: int = 5,
    lr_decay: float = 0.995
) -> pd.DataFrame:
    '''
    Train scGeneRAI on a matrix and extract averaged TF→target edges via LRP.

    Args:
        X_df: Expression matrix (cells x genes) as a pandas DataFrame.
        tag: Tag for output directory naming.
        tfs_in_data: Iterable of TF names to keep as sources.
        targets: Iterable of target gene names to explain.
        epochs: Number of training epochs.
        depth: Hidden depth for scGeneRAI model.
        device: 'cuda' or 'cpu'. If None, auto-detects.
        early_stopping: Use early stopping to restore best snapshot.
        keep_abs: Take absolute value of LRP before averaging.
        remove_self_loops: Drop edges where source == target.
        cleanup: Remove the temporary output directory after reading CSVs.
        out_root: Parent directory for outputs.
        lr: Learning rate for training.
        batch_size: Mini-batch size.
        lr_decay: Exponential LR decay per epoch.

    Returns:
        DataFrame with columns ['source_gene','target_gene','LRP'] averaged across samples,
        sorted by descending LRP.
    '''
    device_name = detect_device(device)
    outdir = os.path.join(out_root, f"_out_{tag}")
    os.makedirs(outdir, exist_ok=True)

    m = scGeneRAI()
    m.fit(
        X_df,
        nepochs=epochs,
        model_depth=depth,
        lr=lr,
        batch_size=batch_size,
        lr_decay=lr_decay,
        early_stopping=early_stopping,
        device_name=device_name
    )
    m.predict_networks(
        X_df,
        PATH=outdir,
        device_name=device_name,
        targets=list(targets)
    )

    raw = _read_lrp_csvs(outdir)
    if cleanup:
        shutil.rmtree(outdir, ignore_errors=True)
    if raw.empty:
        return pd.DataFrame(columns=["source_gene", "target_gene", "LRP"])

    if remove_self_loops:
        raw = raw[raw["source_gene"] != raw["target_gene"]].copy()
    if keep_abs:
        raw["LRP"] = raw["LRP"].abs()

    tfs = set(tfs_in_data)
    tgts = set(targets)
    edges = raw[raw["source_gene"].isin(tfs) & raw["target_gene"].isin(tgts)].copy()
    if edges.empty:
        return pd.DataFrame(columns=["source_gene", "target_gene", "LRP"])

    edges = (
        edges.groupby(["source_gene", "target_gene"], as_index=False)["LRP"]
        .mean()
        .sort_values("LRP", ascending=False)
        .reset_index(drop=True)
    )
    return edges


def run_two_sets(
    X_tfonly: pd.DataFrame,
    X_ext: pd.DataFrame,
    tfs_in_data: Iterable[str],
    targets: Iterable[str],
    epochs: int,
    depth: int,
    device: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Convenience wrapper to train on two matrices (TF-only and extended) and return edges.

    Args:
        X_tfonly: DataFrame with TF-only columns.
        X_ext: DataFrame with an extended set of genes.
        tfs_in_data: TF list present in both matrices.
        targets: Target genes to explain.
        epochs: Training epochs for both runs.
        depth: Model depth for both runs.
        device: Optional explicit device.

    Returns:
        (edges_tfonly, edges_ext): Two DataFrames of averaged edges.
    '''
    edges_tfonly = fit_predict_edges(
        X_df=X_tfonly,
        tag="tfonly",
        tfs_in_data=tfs_in_data,
        targets=targets,
        epochs=epochs,
        depth=depth,
        device=device
    )
    edges_ext = fit_predict_edges(
        X_df=X_ext,
        tag="extended",
        tfs_in_data=tfs_in_data,
        targets=targets,
        epochs=epochs,
        depth=depth,
        device=device
    )
    return edges_tfonly, edges_ext


# Example usage:
# EPOCHS = 150
# DEPTH = 2
# edges_tfonly, edges_ext = run_two_sets(
#     X_tfonly=X_tfonly,
#     X_ext=X_ext,
#     tfs_in_data=tfs_in_data,
#     targets=["PRM1", "PRM2"],
#     epochs=EPOCHS,
#     depth=DEPTH,
# )
# print("[done] edges_tfonly:", edges_tfonly.shape, "| edges_ext:", edges_ext.shape)
# display(edges_tfonly.head(10))
