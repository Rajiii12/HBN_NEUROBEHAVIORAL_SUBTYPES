#!/usr/bin/env python3
"""
Tuned conventional-clustering baseline experiment.

Purpose
-------
Re-evaluate KMeans, Gaussian-mixture, hierarchical, and spectral clustering
using the exact saved subject-level inputs of the final EPG-SD pipeline.

This script DOES NOT rerun EEG preprocessing, CNN-AE training, tensor fusion,
subject aggregation, graph construction, VGAE training, UMAP, or the final
EPG-SD clustering.

Fairness safeguards
-------------------
1. Uses the saved standardized final node-feature matrix (430 x 196).
2. Uses its first 192 columns as the identically standardized EEG-only input.
3. Evaluates direct standardized inputs and PCA-reduced alternatives.
4. Tunes clustering hyperparameters using internal criteria only.
5. Clinical/held-out associations are computed only after model selection.
6. Rejects candidate configurations that repeatedly create clusters smaller
   than 5% of the cohort.
7. Saves the complete grid, selected configurations, labels, clinical results,
   environment, and LaTeX table rows.

Selection rule
--------------
A configuration is eligible when:
- at least 90% of attempted runs complete successfully; and
- at least 80% of successful runs have minimum cluster size >= ceil(0.05*N).

Eligible configurations are ranked by:
1. highest mean silhouette across repeated seeds;
2. lowest mean Davies-Bouldin index;
3. lowest mean balance ratio.

If no configuration for a method/input is eligible, the script retains the
least problematic configuration but clearly flags selection_rule='RELAXED'.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Limit BLAS/OpenMP oversubscription before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy import stats
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_CLUSTERS = 5
CLINICAL_VARIABLES = [
    "p_factor",
    "attention",
    "internalizing",
    "externalizing",
    "age",
    "ehq_total",
]
DISPLAY_VARIABLES = {
    "p_factor": "P-factor",
    "attention": "Attention",
    "internalizing": "Internalizing",
    "externalizing": "Externalizing",
    "age": "Age",
    "ehq_total": "EHQ total",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    input_name: str
    representation: str
    n_features: int
    method: str
    config_key: str
    params_json: str
    seed: int


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuned standardized conventional clustering baselines"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="TECHNICAL_FIX_SPECTRAL_UMAP project root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory; existing results are resumed safely",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default="full",
        help="full=complete documented grid; quick=small smoke test",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel worker processes",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Tasks processed before appending a checkpoint CSV",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Rerun tasks previously saved with status=error",
    )
    return parser.parse_args()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return out
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != K_CLUSTERS:
        raise ValueError(
            f"Expected {K_CLUSTERS} clusters, found {len(unique)}: "
            f"{dict(zip(unique.tolist(), counts.tolist()))}"
        )
    if np.any(counts <= 1):
        raise ValueError(f"Singleton/empty cluster detected: {counts.tolist()}")

    sil = float(silhouette_score(X, labels, metric="euclidean"))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    min_size = int(counts.min())
    max_size = int(counts.max())
    balance = float(max_size / min_size)

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "balance": balance,
        "cluster_sizes": json.dumps(sorted(counts.astype(int).tolist())),
    }


def anova_effects(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    y = np.asarray(values[mask], dtype=float)
    g = np.asarray(labels[mask], dtype=int)
    groups = [y[g == k] for k in sorted(np.unique(g))]
    groups = [x for x in groups if len(x) > 0]
    n = int(len(y))
    k = int(len(groups))

    if n <= k or k < 2:
        return {
            "n": n,
            "anova_F": np.nan,
            "anova_p": np.nan,
            "kruskal_H": np.nan,
            "kruskal_p": np.nan,
            "eta_squared": np.nan,
            "omega_squared": np.nan,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_stat, anova_p = stats.f_oneway(*groups)
        h_stat, kruskal_p = stats.kruskal(*groups)

    grand = float(np.mean(y))
    ss_between = float(sum(len(x) * (float(np.mean(x)) - grand) ** 2 for x in groups))
    ss_total = float(np.sum((y - grand) ** 2))
    ss_within = max(ss_total - ss_between, 0.0)
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    omega2 = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if ss_total > 0 and np.isfinite(ms_within)
        else np.nan
    )
    omega2 = max(float(omega2), 0.0) if np.isfinite(omega2) else np.nan

    return {
        "n": n,
        "anova_F": float(f_stat),
        "anova_p": float(anova_p),
        "kruskal_H": float(h_stat),
        "kruskal_p": float(kruskal_p),
        "eta_squared": float(eta2),
        "omega_squared": float(omega2),
    }


def clinical_associations(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in CLINICAL_VARIABLES:
        if variable not in metadata.columns:
            raise KeyError(f"Clinical metadata missing required column: {variable}")
        vals = pd.to_numeric(metadata[variable], errors="coerce").to_numpy(float)
        row = {"variable": variable, "display_variable": DISPLAY_VARIABLES[variable]}
        row.update(anova_effects(vals, labels))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["anova_p_fdr_bh"] = bh_fdr(df["anova_p"].to_numpy())
    df["kruskal_p_fdr_bh"] = bh_fdr(df["kruskal_p"].to_numpy())
    return df


def load_exact_inputs(root: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    graph_dir = (
        root
        / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
        / "graphK12_alpha0p6"
    )
    node_path = graph_dir / "node_features.npy"
    clinical_path = graph_dir / "UMAP30_K5" / "subject_subtypes.csv"
    eeg_raw_path = (
        root
        / "subject_embeddings_norm_fixed"
        / "subject_embedding_mean_std_median_norm_fixed.npy"
    )
    subject_meta_path = (
        root
        / "subject_embeddings_norm_fixed"
        / "subject_metadata_norm_fixed.csv"
    )

    for path in [node_path, clinical_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required saved final input not found: {path}")

    node = np.load(node_path).astype(np.float64)
    clinical = pd.read_csv(clinical_path)

    if node.ndim != 2 or node.shape[0] != len(clinical):
        raise ValueError(
            f"node_features shape {node.shape} does not align with clinical rows "
            f"{len(clinical)}"
        )
    if node.shape[1] != 196:
        raise ValueError(
            f"Expected exact final node feature dimension 196, found {node.shape[1]}"
        )
    if not np.isfinite(node).all():
        raise ValueError("Saved node_features.npy contains non-finite values")

    # The saved final node matrix is already standardized feature-wise.
    means = node.mean(axis=0)
    stds = node.std(axis=0, ddof=0)
    max_abs_mean = float(np.max(np.abs(means)))
    max_abs_std_error = float(np.max(np.abs(stds - 1.0)))
    if max_abs_mean > 1e-4 or max_abs_std_error > 1e-3:
        raise ValueError(
            "Saved node_features.npy is not standardized as expected: "
            f"max|mean|={max_abs_mean}, max|std-1|={max_abs_std_error}"
        )

    audit: dict[str, Any] = {
        "node_features_path": str(node_path.relative_to(root)),
        "clinical_file": str(clinical_path.relative_to(root)),
        "node_shape": list(node.shape),
        "max_abs_feature_mean": max_abs_mean,
        "max_abs_feature_std_error": max_abs_std_error,
        "used_saved_standardized_node_features": True,
    }

    # Strong audit: confirm first 192 columns equal StandardScaler(raw EEG).
    if eeg_raw_path.exists():
        from sklearn.preprocessing import StandardScaler

        eeg_raw = np.load(eeg_raw_path).astype(np.float64)
        if eeg_raw.shape != (node.shape[0], 192):
            raise ValueError(f"Unexpected raw EEG feature shape: {eeg_raw.shape}")
        eeg_scaled = StandardScaler().fit_transform(eeg_raw)
        max_err = float(np.max(np.abs(eeg_scaled - node[:, :192])))
        audit["raw_eeg_path"] = str(eeg_raw_path.relative_to(root))
        audit["first_192_vs_standardscaled_raw_max_abs_error"] = max_err
        if max_err > 1e-5:
            raise ValueError(
                "The first 192 node-feature columns do not match standardized "
                f"subject EEG features; max error={max_err}"
            )

    # Strong audit: confirm subject order when metadata is available.
    if subject_meta_path.exists():
        subject_meta = pd.read_csv(subject_meta_path)
        if len(subject_meta) != len(clinical):
            raise ValueError("Subject metadata and clinical table row counts differ")
        if "subject" in subject_meta.columns and "subject" in clinical.columns:
            left = subject_meta["subject"].astype(str).to_numpy()
            right = clinical["subject"].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError("Subject order mismatch between saved inputs")
            audit["subject_order_verified"] = True
            audit["subject_metadata_path"] = str(subject_meta_path.relative_to(root))

    inputs = {
        "EEG_PHENOTYPE": node,
        "EEG_ONLY": node[:, :192].copy(),
    }
    return inputs, clinical, audit


def build_representations(
    inputs: dict[str, np.ndarray], out: Path, profile: str
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    specs: list[tuple[str, Any]]
    if profile == "quick":
        specs = [("STD", None), ("PCA32", 32), ("PCA95", 0.95)]
    else:
        specs = [
            ("STD", None),
            ("PCA32", 32),
            ("PCA64", 64),
            ("PCA96", 96),
            ("PCA90", 0.90),
            ("PCA95", 0.95),
            ("PCA99", 0.99),
        ]

    representations: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    rep_dir = out / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)

    for input_name, X in inputs.items():
        for rep_name, n_components in specs:
            if n_components is None:
                Z = X.copy()
                explained = 1.0
                fitted_components = X.shape[1]
                pca = None
            else:
                pca = PCA(n_components=n_components, svd_solver="full")
                Z = pca.fit_transform(X)
                explained = float(pca.explained_variance_ratio_.sum())
                fitted_components = int(Z.shape[1])

            if not np.isfinite(Z).all():
                raise ValueError(f"Non-finite representation: {input_name}/{rep_name}")
            representations[(input_name, rep_name)] = Z.astype(np.float64)
            np.save(rep_dir / f"{input_name}__{rep_name}.npy", Z.astype(np.float32))
            if pca is not None:
                pd.DataFrame(
                    {
                        "component": np.arange(1, fitted_components + 1),
                        "explained_variance_ratio": pca.explained_variance_ratio_,
                        "cumulative_explained_variance": np.cumsum(
                            pca.explained_variance_ratio_
                        ),
                    }
                ).to_csv(
                    rep_dir / f"{input_name}__{rep_name}__pca_variance.csv",
                    index=False,
                )
            rows.append(
                {
                    "input_name": input_name,
                    "representation": rep_name,
                    "input_features": int(X.shape[1]),
                    "output_features": fitted_components,
                    "explained_variance": explained,
                    "n_subjects": int(X.shape[0]),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "representation_summary.csv", index=False)
    return representations, summary


def make_task(
    input_name: str,
    rep_name: str,
    n_features: int,
    method: str,
    params: dict[str, Any],
    seed: int,
) -> Task:
    params_json = json_dumps(params)
    config_key = f"{method}|{params_json}"
    task_id = f"{input_name}|{rep_name}|{config_key}|seed={seed}"
    return Task(
        task_id=task_id,
        input_name=input_name,
        representation=rep_name,
        n_features=n_features,
        method=method,
        config_key=config_key,
        params_json=params_json,
        seed=seed,
    )


def build_tasks(
    representations: dict[tuple[str, str], np.ndarray], profile: str
) -> list[Task]:
    if profile == "quick":
        km_seeds = [0, 1, 2]
        gmm_seeds = [0, 1]
        spectral_seeds = [0, 1]
        kmeans_n_init = [20]
        gmm_covariances = ["diag", "full"]
        gmm_regs = [1e-4]
        spectral_neighbors = [10, 20]
        spectral_gammas = ["scale", 0.1]
    else:
        km_seeds = list(range(20))
        gmm_seeds = list(range(10))
        spectral_seeds = list(range(10))
        kmeans_n_init = [20, 50]
        gmm_covariances = ["full", "tied", "diag", "spherical"]
        gmm_regs = [1e-6, 1e-5, 1e-4, 1e-3]
        spectral_neighbors = [5, 10, 12, 15, 20, 30]
        spectral_gammas = ["scale", 0.01, 0.1, 1.0]

    tasks: list[Task] = []
    for (input_name, rep_name), X in sorted(representations.items()):
        d = int(X.shape[1])

        for n_init in kmeans_n_init:
            for seed in km_seeds:
                tasks.append(
                    make_task(
                        input_name,
                        rep_name,
                        d,
                        "KMeans",
                        {"init": "k-means++", "n_init": n_init},
                        seed,
                    )
                )

        for covariance_type in gmm_covariances:
            for reg_covar in gmm_regs:
                for seed in gmm_seeds:
                    tasks.append(
                        make_task(
                            input_name,
                            rep_name,
                            d,
                            "GMM",
                            {
                                "covariance_type": covariance_type,
                                "reg_covar": reg_covar,
                                "n_init": 5,
                            },
                            seed,
                        )
                    )

        hierarchical_grid = [
            {"linkage": "ward", "metric": "euclidean"},
            {"linkage": "complete", "metric": "euclidean"},
            {"linkage": "complete", "metric": "cosine"},
            {"linkage": "average", "metric": "euclidean"},
            {"linkage": "average", "metric": "cosine"},
        ]
        for params in hierarchical_grid:
            tasks.append(
                make_task(
                    input_name,
                    rep_name,
                    d,
                    "Hierarchical",
                    params,
                    0,
                )
            )

        for n_neighbors in spectral_neighbors:
            for assign_labels in ["kmeans", "discretize"]:
                for seed in spectral_seeds:
                    tasks.append(
                        make_task(
                            input_name,
                            rep_name,
                            d,
                            "Spectral",
                            {
                                "affinity": "nearest_neighbors",
                                "n_neighbors": n_neighbors,
                                "assign_labels": assign_labels,
                                "n_init": 20,
                            },
                            seed,
                        )
                    )

        for gamma in spectral_gammas:
            for assign_labels in ["kmeans", "discretize"]:
                for seed in spectral_seeds:
                    tasks.append(
                        make_task(
                            input_name,
                            rep_name,
                            d,
                            "Spectral",
                            {
                                "affinity": "rbf",
                                "gamma": gamma,
                                "assign_labels": assign_labels,
                                "n_init": 20,
                            },
                            seed,
                        )
                    )

    return tasks


def fit_labels(task: Task, X: np.ndarray) -> np.ndarray:
    params = json.loads(task.params_json)
    method = task.method

    with threadpool_limits(limits=1):
        if method == "KMeans":
            model = KMeans(
                n_clusters=K_CLUSTERS,
                init=params["init"],
                n_init=int(params["n_init"]),
                random_state=task.seed,
                algorithm="lloyd",
            )
            return model.fit_predict(X)

        if method == "GMM":
            model = GaussianMixture(
                n_components=K_CLUSTERS,
                covariance_type=params["covariance_type"],
                reg_covar=float(params["reg_covar"]),
                n_init=int(params["n_init"]),
                init_params="kmeans",
                max_iter=500,
                random_state=task.seed,
            )
            return model.fit_predict(X)

        if method == "Hierarchical":
            model = AgglomerativeClustering(
                n_clusters=K_CLUSTERS,
                linkage=params["linkage"],
                metric=params["metric"],
            )
            return model.fit_predict(X)

        if method == "Spectral":
            gamma = params.get("gamma", 1.0)
            if gamma == "scale":
                gamma = 1.0 / max(X.shape[1], 1)
            kwargs: dict[str, Any] = {
                "n_clusters": K_CLUSTERS,
                "affinity": params["affinity"],
                "assign_labels": params["assign_labels"],
                "n_init": int(params["n_init"]),
                "random_state": task.seed,
                "eigen_solver": "arpack",
                "n_jobs": 1,
            }
            if params["affinity"] == "nearest_neighbors":
                kwargs["n_neighbors"] = int(params["n_neighbors"])
            else:
                kwargs["gamma"] = float(gamma)
            model = SpectralClustering(**kwargs)
            return model.fit_predict(X)

    raise ValueError(f"Unknown method: {method}")


def run_task(task: Task, X: np.ndarray, min_cluster_threshold: int) -> dict[str, Any]:
    start = time.time()
    base = asdict(task)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = fit_labels(task, X)
            metrics = cluster_metrics(X, labels)
        base.update(metrics)
        base["valid_min_cluster"] = bool(
            metrics["min_cluster_size"] >= min_cluster_threshold
        )
        base["status"] = "ok"
        base["error"] = ""
        base["warning_count"] = len(caught)
        base["warnings"] = " | ".join(str(w.message) for w in caught[:3])
    except Exception as exc:
        base.update(
            {
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "min_cluster_size": np.nan,
                "max_cluster_size": np.nan,
                "balance": np.nan,
                "cluster_sizes": "",
                "valid_min_cluster": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "warning_count": 0,
                "warnings": "",
            }
        )
    base["elapsed_seconds"] = time.time() - start
    return base


def checkpoint_results(
    tasks: list[Task],
    representations: dict[tuple[str, str], np.ndarray],
    out_csv: Path,
    min_cluster_threshold: int,
    n_jobs: int,
    chunk_size: int,
    retry_errors: bool,
) -> pd.DataFrame:
    if out_csv.exists():
        previous = pd.read_csv(out_csv)
    else:
        previous = pd.DataFrame()

    completed: set[str] = set()
    if not previous.empty and "task_id" in previous.columns:
        if retry_errors and "status" in previous.columns:
            completed = set(previous.loc[previous["status"] == "ok", "task_id"].astype(str))
        else:
            completed = set(previous["task_id"].astype(str))

    pending = [task for task in tasks if task.task_id not in completed]
    print(f"Total tasks: {len(tasks):,}")
    print(f"Already checkpointed: {len(completed):,}")
    print(f"Pending tasks: {len(pending):,}")

    frames = [previous] if not previous.empty else []
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        print(
            f"\nRunning tasks {start + 1:,}--{start + len(chunk):,} "
            f"of {len(pending):,} pending..."
        )
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(run_task)(
                task,
                representations[(task.input_name, task.representation)],
                min_cluster_threshold,
            )
            for task in chunk
        )
        frames.append(pd.DataFrame(results))
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["task_id"], keep="last")
        merged.to_csv(out_csv, index=False)
        frames = [merged]
        ok = int((merged["status"] == "ok").sum())
        err = int((merged["status"] == "error").sum())
        print(f"Checkpoint saved: {out_csv} | ok={ok:,}, error={err:,}")

    if out_csv.exists():
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def summarize_configs(runs: pd.DataFrame, min_cluster_threshold: int) -> pd.DataFrame:
    group_cols = [
        "input_name",
        "representation",
        "n_features",
        "method",
        "config_key",
        "params_json",
    ]
    rows: list[dict[str, Any]] = []
    for key, df in runs.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        attempted = len(df)
        success = df[df["status"] == "ok"].copy()
        n_success = len(success)
        valid = success[success["valid_min_cluster"].astype(bool)]
        n_valid = len(valid)
        row.update(
            {
                "n_attempted": attempted,
                "n_success": n_success,
                "n_valid": n_valid,
                "success_fraction": n_success / attempted if attempted else 0.0,
                "valid_fraction": n_valid / n_success if n_success else 0.0,
            }
        )
        for metric in [
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "min_cluster_size",
            "max_cluster_size",
            "balance",
            "elapsed_seconds",
        ]:
            vals = pd.to_numeric(success[metric], errors="coerce") if n_success else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        row["eligible"] = bool(
            row["success_fraction"] >= 0.90 and row["valid_fraction"] >= 0.80
        )
        row["min_cluster_threshold"] = min_cluster_threshold
        rows.append(row)
    return pd.DataFrame(rows)


def select_configs(summary: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for (input_name, method), group in summary.groupby(["input_name", "method"]):
        eligible = group[group["eligible"]].copy()
        if not eligible.empty:
            pool = eligible
            rule = "STRICT"
        else:
            pool = group.copy()
            rule = "RELAXED"

        pool = pool.sort_values(
            by=[
                "eligible",
                "valid_fraction",
                "success_fraction",
                "silhouette_mean",
                "davies_bouldin_mean",
                "balance_mean",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        )
        chosen = pool.iloc[0].copy()
        chosen["selection_rule"] = rule
        chosen["selection_rank"] = 1
        selected_rows.append(chosen)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def choose_representative_run(
    runs: pd.DataFrame, selected_config: pd.Series
) -> pd.Series:
    subset = runs[
        (runs["input_name"] == selected_config["input_name"])
        & (runs["method"] == selected_config["method"])
        & (runs["representation"] == selected_config["representation"])
        & (runs["config_key"] == selected_config["config_key"])
        & (runs["status"] == "ok")
    ].copy()
    valid = subset[subset["valid_min_cluster"].astype(bool)].copy()
    if not valid.empty:
        subset = valid
    target_sil = safe_float(selected_config["silhouette_mean"])
    target_db = safe_float(selected_config["davies_bouldin_mean"])
    subset["sil_distance"] = (subset["silhouette"] - target_sil).abs()
    subset["db_distance"] = (subset["davies_bouldin"] - target_db).abs()
    subset = subset.sort_values(
        ["sil_distance", "db_distance", "seed"], ascending=[True, True, True]
    )
    if subset.empty:
        raise RuntimeError(
            f"No successful representative run for {selected_config['input_name']} / "
            f"{selected_config['method']}"
        )
    return subset.iloc[0]


def load_epgsd_reference(root: Path) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    final_dir = (
        root
        / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
        / "graphK12_alpha0p6"
        / "UMAP30_K5"
    )
    quality_path = final_dir / "main_quality.csv"
    clinical_path = final_dir / "clinical_stats.csv"
    labels_path = final_dir / "subject_subtypes.csv"
    for p in [quality_path, clinical_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing archived final EPG-SD file: {p}")
    q = pd.read_csv(quality_path).iloc[0]
    clinical = pd.read_csv(clinical_path)
    labels_df = pd.read_csv(labels_path)
    labels = labels_df["subtype"].to_numpy(int)
    mean_eta2 = float(clinical["eta_squared"].mean())
    fdr_count = int((clinical["anova_p_fdr_bh"] < 0.05).sum())
    ref = {
        "input_name": "EPG-SD",
        "method": "EPG-SD proposed",
        "representation": "VGAE+UMAP+Spectral",
        "n_features": 2,
        "seed": np.nan,
        "silhouette": float(q["silhouette"]),
        "davies_bouldin": float(q["davies_bouldin"]),
        "calinski_harabasz": float(q["calinski_harabasz"]),
        "min_cluster_size": int(q["min_cluster_size"]),
        "max_cluster_size": int(q["max_cluster_size"]),
        "balance": float(q["cluster_size_ratio_max_min"]),
        "mean_eta_squared": mean_eta2,
        "fdr_count": f"{fdr_count}/6",
        "selection_rule": "ARCHIVED_FINAL",
        "params_json": "{}",
    }
    return ref, clinical, labels


def materialize_selected(
    root: Path,
    out: Path,
    representations: dict[tuple[str, str], np.ndarray],
    metadata: pd.DataFrame,
    runs: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_dir = out / "selected_labels"
    clinical_dir = out / "selected_clinical"
    labels_dir.mkdir(parents=True, exist_ok=True)
    clinical_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    clinical_rows: list[pd.DataFrame] = []

    for _, config in selected.iterrows():
        representative = choose_representative_run(runs, config)
        task = Task(
            task_id=str(representative["task_id"]),
            input_name=str(representative["input_name"]),
            representation=str(representative["representation"]),
            n_features=int(representative["n_features"]),
            method=str(representative["method"]),
            config_key=str(representative["config_key"]),
            params_json=str(representative["params_json"]),
            seed=int(representative["seed"]),
        )
        X = representations[(task.input_name, task.representation)]
        labels = fit_labels(task, X)
        metrics = cluster_metrics(X, labels)

        stem = f"{task.input_name}__{task.method}"
        pd.DataFrame(
            {
                "subject": metadata["subject"].astype(str)
                if "subject" in metadata.columns
                else np.arange(len(labels)).astype(str),
                "subtype": labels.astype(int),
            }
        ).to_csv(labels_dir / f"{stem}.csv", index=False)

        cdf = clinical_associations(metadata, labels)
        cdf.insert(0, "input_name", task.input_name)
        cdf.insert(1, "method", task.method)
        cdf.insert(2, "representation", task.representation)
        cdf.insert(3, "seed", task.seed)
        cdf.to_csv(clinical_dir / f"{stem}.csv", index=False)
        clinical_rows.append(cdf)

        mean_eta2 = float(cdf["eta_squared"].mean())
        fdr_n = int((cdf["anova_p_fdr_bh"] < 0.05).sum())
        metric_rows.append(
            {
                "input_name": task.input_name,
                "method": task.method,
                "representation": task.representation,
                "n_features": task.n_features,
                "seed": task.seed,
                "params_json": task.params_json,
                "selection_rule": config["selection_rule"],
                "config_silhouette_mean": config["silhouette_mean"],
                "config_silhouette_sd": config["silhouette_sd"],
                "config_db_mean": config["davies_bouldin_mean"],
                "config_valid_fraction": config["valid_fraction"],
                **metrics,
                "mean_eta_squared": mean_eta2,
                "fdr_count": f"{fdr_n}/6",
            }
        )

    epgsd_ref, epgsd_clinical, epgsd_labels = load_epgsd_reference(root)
    metric_rows.append(epgsd_ref)
    epgsd_clinical = epgsd_clinical.copy()
    epgsd_clinical.insert(0, "input_name", "EPG-SD")
    epgsd_clinical.insert(1, "method", "EPG-SD proposed")
    epgsd_clinical.insert(2, "representation", "VGAE+UMAP+Spectral")
    epgsd_clinical.insert(3, "seed", np.nan)
    clinical_rows.append(epgsd_clinical)
    pd.DataFrame(
        {
            "subject": metadata["subject"].astype(str)
            if "subject" in metadata.columns
            else np.arange(len(epgsd_labels)).astype(str),
            "subtype": epgsd_labels,
        }
    ).to_csv(labels_dir / "EPG-SD__archived_final.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    clinical_df = pd.concat(clinical_rows, ignore_index=True, sort=False)
    metrics_df.to_csv(out / "selected_representative_metrics.csv", index=False)
    clinical_df.to_csv(out / "selected_clinical_associations.csv", index=False)
    return metrics_df, clinical_df


def escape_latex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_latex_outputs(metrics: pd.DataFrame, out: Path) -> None:
    ordering = {"EEG_PHENOTYPE": 0, "EEG_ONLY": 1, "EPG-SD": 2}
    method_order = {"KMeans": 0, "GMM": 1, "Hierarchical": 2, "Spectral": 3, "EPG-SD proposed": 4}
    table = metrics.copy()
    table["_input_order"] = table["input_name"].map(ordering).fillna(9)
    table["_method_order"] = table["method"].map(method_order).fillna(9)
    table = table.sort_values(["_input_order", "_method_order"])

    rows: list[str] = []
    mean_sd_rows: list[str] = []
    for _, r in table.iterrows():
        if r["input_name"] == "EPG-SD":
            name = r"\textbf{EPG-SD proposed}"
            rep = "VGAE--UMAP--Spectral"
            row = (
                f"{name} & {rep} & "
                f"\\textbf{{{r['silhouette']:.4f}}} & "
                f"\\textbf{{{r['davies_bouldin']:.4f}}} & "
                f"\\textbf{{{r['calinski_harabasz']:.2f}}} & "
                f"\\textbf{{{int(r['min_cluster_size'])}}} & "
                f"\\textbf{{{r['balance']:.2f}}} & "
                f"\\textbf{{{r['mean_eta_squared']:.4f}}} & "
                f"\\textbf{{{r['fdr_count']}}} \\\\"
            )
            mean_sd = row
        else:
            input_label = (
                "EEG--phenotype" if r["input_name"] == "EEG_PHENOTYPE" else "EEG-only"
            )
            name = f"{escape_latex(str(r['method']))} ({input_label})"
            rep = escape_latex(str(r["representation"]))
            row = (
                f"{name} & {rep} & "
                f"{r['silhouette']:.4f} & {r['davies_bouldin']:.4f} & "
                f"{r['calinski_harabasz']:.2f} & {int(r['min_cluster_size'])} & "
                f"{r['balance']:.2f} & {r['mean_eta_squared']:.4f} & "
                f"{r['fdr_count']} \\\\"
            )
            mean_sd = (
                f"{name} & {rep} & "
                f"{r['config_silhouette_mean']:.4f}$\\pm${r['config_silhouette_sd']:.4f} & "
                f"{r['config_db_mean']:.4f} & {r['calinski_harabasz']:.2f} & "
                f"{int(r['min_cluster_size'])} & {r['balance']:.2f} & "
                f"{r['mean_eta_squared']:.4f} & {r['fdr_count']} \\\\"
            )
        rows.append(row)
        mean_sd_rows.append(mean_sd)

    (out / "tuned_table30_rows_representative.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (out / "tuned_table30_rows_mean_sd.tex").write_text(
        "\n".join(mean_sd_rows) + "\n", encoding="utf-8"
    )


def write_selected_hyperparameters(selected: pd.DataFrame, out: Path) -> None:
    lines = []
    for _, row in selected.sort_values(["input_name", "method"]).iterrows():
        lines.append(
            f"[{row['input_name']} | {row['method']}]\n"
            f"representation = {row['representation']} ({int(row['n_features'])} features)\n"
            f"parameters     = {row['params_json']}\n"
            f"selection_rule = {row['selection_rule']}\n"
            f"mean silhouette = {row['silhouette_mean']:.6f} +/- {row['silhouette_sd']:.6f}\n"
            f"mean DB         = {row['davies_bouldin_mean']:.6f}\n"
            f"valid fraction  = {row['valid_fraction']:.3f}\n"
        )
    (out / "selected_hyperparameters.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("TUNED STANDARDIZED CONVENTIONAL CLUSTERING BASELINES")
    print("=" * 88)
    print("Project root :", root)
    print("Output       :", out)
    print("Profile      :", args.profile)
    print("Parallel jobs:", args.n_jobs)
    print("No upstream EPG-SD stage will be rerun.\n")

    inputs, metadata, input_audit = load_exact_inputs(root)
    n_subjects = len(metadata)
    min_cluster_threshold = int(math.ceil(0.05 * n_subjects))
    print("Exact saved input audit passed:")
    print(json.dumps(input_audit, indent=2))
    print(f"Minimum cluster threshold: {min_cluster_threshold} subjects (5% of N)")

    representations, rep_summary = build_representations(inputs, out, args.profile)
    tasks = build_tasks(representations, args.profile)

    grid_manifest = pd.DataFrame([asdict(t) for t in tasks])
    grid_manifest.to_csv(out / "planned_grid_tasks.csv", index=False)

    runs = checkpoint_results(
        tasks=tasks,
        representations=representations,
        out_csv=out / "full_grid_per_run.csv",
        min_cluster_threshold=min_cluster_threshold,
        n_jobs=max(1, args.n_jobs),
        chunk_size=max(1, args.chunk_size),
        retry_errors=args.retry_errors,
    )

    if runs.empty:
        raise RuntimeError("No baseline runs were produced")

    summary = summarize_configs(runs, min_cluster_threshold)
    summary.to_csv(out / "full_grid_config_summary.csv", index=False)
    selected = select_configs(summary)
    selected.to_csv(out / "selected_configurations.csv", index=False)
    write_selected_hyperparameters(selected, out)

    selected_metrics, selected_clinical = materialize_selected(
        root=root,
        out=out,
        representations=representations,
        metadata=metadata,
        runs=runs,
        selected=selected,
    )
    write_latex_outputs(selected_metrics, out)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "profile": args.profile,
        "n_jobs": args.n_jobs,
        "n_subjects": n_subjects,
        "min_cluster_threshold": min_cluster_threshold,
        "selection_rule": {
            "success_fraction_min": 0.90,
            "valid_fraction_min": 0.80,
            "primary": "highest mean silhouette across repeated seeds",
            "tie_breaker_1": "lowest mean Davies-Bouldin",
            "tie_breaker_2": "lowest mean balance ratio",
            "clinical_outcomes_used_for_selection": False,
        },
        "input_audit": input_audit,
        "statement": (
            "All conventional baselines were fitted to saved standardized "
            "subject-level inputs. No EEG preprocessing, CNN-AE, PCA tensor "
            "fusion, subject aggregation, graph construction, VGAE, UMAP, or "
            "final EPG-SD clustering stage was rerun."
        ),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("SELECTED CONFIGURATIONS")
    print("=" * 88)
    cols = [
        "input_name",
        "method",
        "representation",
        "params_json",
        "selection_rule",
        "silhouette_mean",
        "silhouette_sd",
        "davies_bouldin_mean",
        "valid_fraction",
    ]
    print(selected[cols].to_string(index=False))

    print("\n" + "=" * 88)
    print("REPRESENTATIVE TABLE ROWS")
    print("=" * 88)
    print((out / "tuned_table30_rows_representative.tex").read_text())

    print("Saved complete results in:")
    print(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        traceback.print_exc()
        raise
