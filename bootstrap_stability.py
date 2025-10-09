import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EdgeDF = pd.DataFrame


def jaccard(a: set, b: set) -> float:
    '''
    Jaccard index between two sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        |a ∩ b| / |a ∪ b|, or 0.0 if both are empty.
    '''
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def topk_sets_by_target(
    edges: EdgeDF,
    targets: Sequence[str],
    k: int
) -> Dict[str, set]:
    '''
    Build Top-K edge sets per target as "TF->TARGET" strings.

    Args:
        edges: DataFrame with columns ['source_gene','target_gene','LRP'] sorted by descending LRP.
        targets: Target genes to extract.
        k: Number of top edges per target.

    Returns:
        Dict[target] -> set of "TF->target" strings for the top-K sources.
    '''
    out: Dict[str, set] = {}
    for t in targets:
        sel = edges[edges["target_gene"] == t]
        s = set((sel.head(k)["source_gene"] + "->" + t).tolist())
        out[t] = s
    return out


def bootstrap_stability(
    X_df: pd.DataFrame,
    targets: Sequence[str],
    fit_predict_fn: Callable[[pd.DataFrame, str], EdgeDF],
    tag: str,
    R: int,
    keep_frac: float,
    k: int,
    rng: Optional[np.random.Generator] = None,
    return_edge_freq: bool = True
) -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[str, pd.DataFrame]
]:
    '''
    Bootstrap Top-K stability for TF→target edges.

    Args:
        X_df: Expression matrix (cells x genes).
        targets: Target genes to evaluate.
        fit_predict_fn: Callable that trains on a provided matrix slice and returns
            an edge DataFrame with columns ['source_gene','target_gene','LRP'], sorted desc by LRP.
            Signature must be (X_subset: DataFrame, tag: str) -> DataFrame.
        tag: Label used in per-bootstrap tags forwarded to fit_predict_fn.
        R: Number of bootstrap resamples.
        keep_frac: Fraction of cells to keep per resample (without replacement).
        k: Top-K edges per target to compare.
        rng: Optional NumPy Generator; if None, uses default Generator.
        return_edge_freq: If True, also returns per-target edge frequency tables.

    Returns:
        stats: Dict[target] -> (mean_jaccard@K, std_jaccard@K) across bootstrap pairs.
        freq_tables: Dict[target] -> DataFrame with columns ['edge','count','freq'],
                     sorted by descending count. Empty dict if return_edge_freq=False.
    '''
    if rng is None:
        rng = np.random.default_rng()

    sets: Dict[str, List[set]] = {t: [] for t in targets}
    n_keep = max(1, int(keep_frac * len(X_df)))

    for r in range(R):
        keep_idx = rng.choice(X_df.index.to_numpy(), n_keep, replace=False)
        ed = fit_predict_fn(X_df.loc[keep_idx], f"{tag}_b{r}")
        tops = topk_sets_by_target(ed, targets=targets, k=k)
        for t in targets:
            sets[t].append(tops[t])

    stats: Dict[str, Tuple[float, float]] = {}
    freq_tables: Dict[str, pd.DataFrame] = {}

    for t in targets:
        lst = sets[t]
        pairs = list(combinations(range(len(lst)), 2))
        vals = [jaccard(lst[i], lst[j]) for i, j in pairs] if pairs else []
        mu = float(np.mean(vals)) if vals else np.nan
        sd = float(np.std(vals, ddof=0)) if vals else np.nan
        stats[t] = (mu, sd)

        if return_edge_freq:
            cnt = Counter([e for S in lst for e in S])
            if cnt:
                df = pd.DataFrame(
                    [{"edge": k_, "count": v_, "freq": v_ / len(lst)} for k_, v_ in cnt.items()]
                ).sort_values(["count", "edge"], ascending=[False, True]).reset_index(drop=True)
            else:
                df = pd.DataFrame(columns=["edge", "count", "freq"])
            freq_tables[t] = df

    return stats, freq_tables


def print_bootstrap_summary(
    stats: Mapping[str, Tuple[float, float]],
    k: int,
    R: int,
    keep_frac: float,
    header: Optional[str] = None
) -> None:
    '''
    Nicely format bootstrap stability summary.

    Args:
        stats: Dict[target] -> (mean_jaccard, std_jaccard).
        k: Top-K used.
        R: Number of resamples.
        keep_frac: Fraction of cells per resample.
        header: Optional header prefix to print.
    '''
    if header:
        print(header)
    print(f"[Bootstrap] Mean Jaccard@{k} (R={R}, keep={int(100*keep_frac)}% cells):")
    for t, (mu, sd) in stats.items():
        mu_s = "nan" if np.isnan(mu) else f"{mu:.3f}"
        sd_s = "nan" if np.isnan(sd) else f"{sd:.3f}"
        print(f"  {t}: {mu_s} ± {sd_s}")


# Example wiring with your existing fit_predict_edges:
# def fit_fn(sub_df: pd.DataFrame, tag: str) -> pd.DataFrame:
#     return fit_predict_edges(
#         X_df=sub_df,
#         tag=tag,
#         tfs_in_data=tfs_in_data,
#         targets=targets,
#         epochs=EPOCHS,
#         depth=DEPTH
#     )
#
# R_BOOT = 20
# KEEP_FRAC_BOOT = 0.7
# TOPK = 20
# rng = np.random.default_rng(42)
#
# boot_stats_tf, boot_freq_tf = bootstrap_stability(
#     X_df=X_tfonly,
#     targets=targets,
#     fit_predict_fn=fit_fn,
#     tag="tfonly",
#     R=R_BOOT,
#     keep_frac=KEEP_FRAC_BOOT,
#     k=TOPK,
#     rng=rng
# )
# print_bootstrap_summary(boot_stats_tf, k=TOPK, R=R_BOOT, keep_frac=KEEP_FRAC_BOOT, header=None)
# for t in targets:
#     print(f"\n== {t} ==")
#     display(boot_freq_tf[t].head(10))
