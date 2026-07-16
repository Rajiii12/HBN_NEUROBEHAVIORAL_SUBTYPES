#!/usr/bin/env python3
"""
Native-variable-K and matched-K DBSCAN sensitivity analysis.

This script reproduces the audited Du-style pipeline and reports three
separate selections from the same complete parameter grid:

1. original_k_guided
   The archived K-guided objective used in the earlier benchmark:
       silhouette
       - 0.01 * abs(K - 5)
       - 0.002 * noise_percent
       - 0.005 * log1p(balance_ratio)

2. native_variable_k
   A true variable-K DBSCAN selection with no K=5 term:
       silhouette
       - 0.002 * noise_percent
       - 0.005 * log1p(balance_ratio)

3. matched_k5_strict
   A sensitivity analysis restricted to candidates that naturally produce:
       exactly 5 non-noise clusters
       coverage >= 80%
       minimum cluster size >= 22
   No merging and no reassignment of DBSCAN noise are performed.

Internal metrics and clinical statistics are calculated on non-noise subjects.
The full 40 x 18 = 720 candidate grid is saved.

The script uses the exact saved 430 x 196 EEG-phenotype node-feature matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import __version__ as sklearn_version
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
K_REFERENCE = 5
CLINICAL_COLUMNS = [
    "age",
    "p_factor",
    "attention",
    "internalizing",
    "externalizing",
    "ehq_total",
]

EXPECTED_OLD = {
    "eps": 0.5282282338521644,
    "min_samples": 3,
    "clusters": 2,
    "noise_count": 140,
    "noise_percent": 32.55813953488372,
    "minimum_cluster_size": 3,
    "maximum_cluster_size": 287,
    "balance_ratio": 95.66666666666667,
    "silhouette": 0.19950162935333005,
    "davies_bouldin": 0.794497296191352,
    "calinski_harabasz": 6.825808335017752,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help="Defaults to the exact saved 430 x 196 final node_features.npy.",
    )
    p.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Defaults to subject_embeddings_norm_fixed/subject_metadata_norm_fixed.csv.",
    )
    p.add_argument("--native-noise-cap", type=float, default=35.0)
    p.add_argument("--k5-min-coverage", type=float, default=80.0)
    p.add_argument("--k5-min-cluster-size", type=int, default=22)
    return p.parse_args()


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted[order[i]] = min(running, 1.0)
    return adjusted


def dunn_index(x: np.ndarray, labels: np.ndarray) -> float:
    x = StandardScaler().fit_transform(x)
    labels = np.asarray(labels)
    valid = labels != -1
    x = x[valid]
    labels = labels[valid]
    unique = np.unique(labels)

    if len(unique) < 2:
        return float("nan")

    distances = pairwise_distances(x)
    max_intra = 0.0
    for cluster in unique:
        idx = np.where(labels == cluster)[0]
        if len(idx) > 1:
            max_intra = max(
                max_intra,
                float(distances[np.ix_(idx, idx)].max()),
            )

    min_inter = float("inf")
    for i, c1 in enumerate(unique):
        idx1 = np.where(labels == c1)[0]
        for c2 in unique[i + 1:]:
            idx2 = np.where(labels == c2)[0]
            if len(idx1) and len(idx2):
                min_inter = min(
                    min_inter,
                    float(distances[np.ix_(idx1, idx2)].min()),
                )

    if max_intra <= 0 or not np.isfinite(min_inter):
        return float("nan")
    return float(min_inter / max_intra)


def cluster_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    x_scaled = StandardScaler().fit_transform(x)
    labels = np.asarray(labels, dtype=int)

    valid = labels != -1
    x_valid = x_scaled[valid]
    labels_valid = labels[valid]
    unique, counts = np.unique(labels_valid, return_counts=True)

    noise_count = int(np.sum(labels == -1))
    assigned_n = int(np.sum(valid))
    n_total = int(len(labels))
    noise_percent = float(100.0 * noise_count / n_total)
    coverage_percent = float(100.0 * assigned_n / n_total)

    result: dict[str, Any] = {
        "clusters": int(len(unique)),
        "assigned_n": assigned_n,
        "noise_count": noise_count,
        "noise_percent": noise_percent,
        "coverage_percent": coverage_percent,
        "minimum_cluster_size": int(counts.min()) if len(counts) else np.nan,
        "maximum_cluster_size": int(counts.max()) if len(counts) else np.nan,
        "balance_ratio": (
            float(counts.max() / counts.min())
            if len(counts) and counts.min() > 0
            else np.nan
        ),
        "cluster_sizes": json.dumps(
            {int(k): int(v) for k, v in zip(unique, counts)}
        ),
    }

    if len(unique) > 1 and len(unique) < len(labels_valid):
        result["silhouette"] = float(
            silhouette_score(x_valid, labels_valid)
        )
        result["davies_bouldin"] = float(
            davies_bouldin_score(x_valid, labels_valid)
        )
        result["calinski_harabasz"] = float(
            calinski_harabasz_score(x_valid, labels_valid)
        )
        result["dunn"] = float(dunn_index(x_valid, labels_valid))
    else:
        result.update({
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "dunn": np.nan,
        })

    return result


def one_way_effect(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(values) & (labels != -1)
    values = values[valid]
    labels = labels[valid]

    groups = [
        values[labels == k]
        for k in np.unique(labels)
        if np.sum(labels == k) >= 2
    ]
    if len(groups) < 2:
        return {
            "anova_F": np.nan,
            "anova_p": 1.0,
            "kruskal_H": np.nan,
            "kruskal_p": 1.0,
            "eta_squared": 0.0,
            "omega_squared": 0.0,
            "n_used": int(len(values)),
        }

    anova_f, anova_p = stats.f_oneway(*groups)
    kruskal_h, kruskal_p = stats.kruskal(*groups)

    grand = float(np.mean(values))
    ss_between = sum(
        len(g) * (float(np.mean(g)) - grand) ** 2
        for g in groups
    )
    ss_within = sum(
        float(np.sum((g - float(np.mean(g))) ** 2))
        for g in groups
    )
    ss_total = ss_between + ss_within
    df_between = len(groups) - 1
    df_within = len(values) - len(groups)
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    omega_den = ss_total + ms_within
    omega_num = ss_between - df_between * ms_within
    omega_squared = (
        max(0.0, omega_num / omega_den) if omega_den > 0 else 0.0
    )

    return {
        "anova_F": float(anova_f),
        "anova_p": float(anova_p),
        "kruskal_H": float(kruskal_h),
        "kruskal_p": float(kruskal_p),
        "eta_squared": float(eta_squared),
        "omega_squared": float(omega_squared),
        "n_used": int(len(values)),
    }


def clinical_validation(
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for variable in CLINICAL_COLUMNS:
        if variable not in metadata.columns:
            continue
        values = pd.to_numeric(
            metadata[variable],
            errors="coerce",
        ).to_numpy()
        rows.append({
            "variable": variable,
            **one_way_effect(values, labels),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["anova_p_fdr_bh"] = bh_fdr(
        table["anova_p"].to_numpy()
    )
    table["kruskal_p_fdr_bh"] = bh_fdr(
        table["kruskal_p"].to_numpy()
    )
    table["anova_significant_fdr"] = (
        table["anova_p_fdr_bh"] < 0.05
    )
    return table


def select_best(
    grid: pd.DataFrame,
    eligibility_column: str,
    score_column: str,
) -> pd.Series | None:
    eligible = grid[grid[eligibility_column]].copy()
    eligible = eligible[np.isfinite(eligible[score_column])]
    if eligible.empty:
        return None

    eligible = eligible.sort_values(
        [
            score_column,
            "silhouette",
            "coverage_percent",
            "minimum_cluster_size",
            "balance_ratio",
            "eps",
            "min_samples",
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    return eligible.iloc[0]


def selected_row(
    mode: str,
    row: pd.Series | None,
) -> dict[str, Any]:
    if row is None:
        return {
            "mode": mode,
            "status": "NO_ELIGIBLE_SOLUTION",
        }

    fields = [
        "eps",
        "min_samples",
        "clusters",
        "assigned_n",
        "noise_count",
        "noise_percent",
        "coverage_percent",
        "minimum_cluster_size",
        "maximum_cluster_size",
        "balance_ratio",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "dunn",
        "cluster_sizes",
        "original_k_guided_score",
        "native_score",
    ]
    result = {
        "mode": mode,
        "status": "SELECTED",
    }
    for field in fields:
        result[field] = row[field]
    return result


def save_selection_outputs(
    mode: str,
    selected: pd.Series | None,
    x_tsne_scaled: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    if selected is None:
        (mode_dir / "NO_ELIGIBLE_SOLUTION.txt").write_text(
            "No candidate satisfied the predefined criteria.\n",
            encoding="utf-8",
        )
        return

    eps = float(selected["eps"])
    min_samples = int(selected["min_samples"])
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(x_tsne_scaled)

    pd.DataFrame({
        "subject": metadata["subject"].astype(str)
        if "subject" in metadata.columns
        else np.arange(len(labels)),
        "dbscan_label": labels,
        "is_noise": labels == -1,
    }).to_csv(mode_dir / "labels.csv", index=False)

    clinical = clinical_validation(metadata, labels)
    clinical.to_csv(mode_dir / "clinical_stats.csv", index=False)

    if not clinical.empty:
        clinical_summary = {
            "mean_eta_squared": float(
                clinical["eta_squared"].mean()
            ),
            "max_eta_squared": float(
                clinical["eta_squared"].max()
            ),
            "mean_omega_squared": float(
                clinical["omega_squared"].mean()
            ),
            "fdr_significant_count": int(
                clinical["anova_significant_fdr"].sum()
            ),
            "clinical_variables_evaluated": int(len(clinical)),
        }
    else:
        clinical_summary = {
            "mean_eta_squared": np.nan,
            "max_eta_squared": np.nan,
            "mean_omega_squared": np.nan,
            "fdr_significant_count": np.nan,
            "clinical_variables_evaluated": 0,
        }

    metrics = cluster_metrics(x_tsne_scaled, labels)
    summary = {
        "mode": mode,
        "eps": eps,
        "min_samples": min_samples,
        **metrics,
        **clinical_summary,
    }
    pd.DataFrame([summary]).to_csv(
        mode_dir / "selected_summary.csv",
        index=False,
    )

    valid = labels != -1
    plt.figure(figsize=(7, 6))
    plt.scatter(
        x_tsne_scaled[~valid, 0],
        x_tsne_scaled[~valid, 1],
        marker="x",
        s=24,
        alpha=0.6,
        label="Noise",
    )
    scatter = plt.scatter(
        x_tsne_scaled[valid, 0],
        x_tsne_scaled[valid, 1],
        c=labels[valid],
        s=22,
        alpha=0.8,
    )
    handles, legend_labels = scatter.legend_elements()
    plt.legend(
        handles + ([plt.Line2D([], [], marker="x", linestyle="None")] if np.any(~valid) else []),
        legend_labels + (["Noise"] if np.any(~valid) else []),
        title="DBSCAN",
        loc="best",
    )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(
        f"{mode}: K={metrics['clusters']}, "
        f"coverage={metrics['coverage_percent']:.1f}%"
    )
    plt.tight_layout()
    plt.savefig(mode_dir / "tsne_dbscan_projection.png", dpi=300)
    plt.savefig(mode_dir / "tsne_dbscan_projection.pdf")
    plt.close()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_file = (
        args.feature_file.expanduser().resolve()
        if args.feature_file is not None
        else (
            root
            / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
            / "graphK12_alpha0p6"
            / "node_features.npy"
        )
    )
    metadata_file = (
        args.metadata_file.expanduser().resolve()
        if args.metadata_file is not None
        else (
            root
            / "subject_embeddings_norm_fixed"
            / "subject_metadata_norm_fixed.csv"
        )
    )

    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    x = np.load(feature_file)
    metadata = pd.read_csv(metadata_file)

    if x.shape != (430, 196):
        raise ValueError(
            f"Expected exact saved node-feature shape (430, 196), found {x.shape}"
        )
    if len(metadata) != 430:
        raise ValueError(
            f"Expected 430 metadata rows, found {len(metadata)}"
        )

    x_standardized = StandardScaler().fit_transform(x)
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_standardized)

    perplexity = min(30, max(5, (len(x_pca) - 1) // 3))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    x_tsne = tsne.fit_transform(
        StandardScaler().fit_transform(x_pca)
    )
    x_tsne_scaled = StandardScaler().fit_transform(x_tsne)

    np.save(out_dir / "pca95_features.npy", x_pca)
    np.save(out_dir / "tsne_3d.npy", x_tsne)
    np.save(out_dir / "tsne_3d_standardized.npy", x_tsne_scaled)

    neighbors = NearestNeighbors(n_neighbors=10).fit(x_tsne_scaled)
    distances, _ = neighbors.kneighbors(x_tsne_scaled)
    base = distances[:, -1]

    eps_low = max(float(np.percentile(base, 5)), 1e-4)
    eps_high = max(float(np.percentile(base, 95)), 1e-3)
    eps_candidates = np.linspace(eps_low, eps_high, 40)
    min_samples_candidates = list(range(3, 21))

    rows = []
    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            labels = DBSCAN(
                eps=float(eps),
                min_samples=int(min_samples),
            ).fit_predict(x_tsne_scaled)
            metrics = cluster_metrics(x_tsne_scaled, labels)

            c = int(metrics["clusters"])
            noise = float(metrics["noise_percent"])
            balance = float(metrics["balance_ratio"])
            silhouette = float(metrics["silhouette"])

            historical_candidate = bool(
                2 <= c <= 10
                and noise <= 35.0
                and np.isfinite(silhouette)
            )
            native_candidate = bool(
                2 <= c <= 10
                and noise <= args.native_noise_cap
                and np.isfinite(silhouette)
            )
            matched_k5_strict = bool(
                c == 5
                and metrics["coverage_percent"] >= args.k5_min_coverage
                and metrics["minimum_cluster_size"]
                    >= args.k5_min_cluster_size
                and np.isfinite(silhouette)
            )
            matched_k5_relaxed = bool(
                c == 5
                and noise <= 35.0
                and np.isfinite(silhouette)
            )

            original_score = (
                silhouette
                - 0.01 * abs(c - K_REFERENCE)
                - 0.002 * noise
                - 0.005 * math.log1p(balance)
                if historical_candidate
                else np.nan
            )
            native_score = (
                silhouette
                - 0.002 * noise
                - 0.005 * math.log1p(balance)
                if native_candidate or matched_k5_relaxed
                else np.nan
            )

            rows.append({
                "eps": float(eps),
                "min_samples": int(min_samples),
                **metrics,
                "historical_candidate": historical_candidate,
                "native_candidate": native_candidate,
                "matched_k5_strict": matched_k5_strict,
                "matched_k5_relaxed": matched_k5_relaxed,
                "original_k_guided_score": original_score,
                "native_score": native_score,
            })

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "complete_dbscan_grid_720.csv", index=False)

    old_selected = select_best(
        grid,
        "historical_candidate",
        "original_k_guided_score",
    )
    native_selected = select_best(
        grid,
        "native_candidate",
        "native_score",
    )
    k5_strict_selected = select_best(
        grid,
        "matched_k5_strict",
        "native_score",
    )
    k5_relaxed_selected = select_best(
        grid,
        "matched_k5_relaxed",
        "native_score",
    )

    selections = pd.DataFrame([
        selected_row("original_k_guided", old_selected),
        selected_row("native_variable_k", native_selected),
        selected_row("matched_k5_strict", k5_strict_selected),
        selected_row("matched_k5_relaxed_diagnostic", k5_relaxed_selected),
    ])
    selections.to_csv(out_dir / "selected_dbscan_solutions.csv", index=False)

    save_selection_outputs(
        "original_k_guided",
        old_selected,
        x_tsne_scaled,
        metadata,
        out_dir,
    )
    save_selection_outputs(
        "native_variable_k",
        native_selected,
        x_tsne_scaled,
        metadata,
        out_dir,
    )
    save_selection_outputs(
        "matched_k5_strict",
        k5_strict_selected,
        x_tsne_scaled,
        metadata,
        out_dir,
    )
    save_selection_outputs(
        "matched_k5_relaxed_diagnostic",
        k5_relaxed_selected,
        x_tsne_scaled,
        metadata,
        out_dir,
    )

    reproduction = {
        "status": "NOT_EVALUATED",
        "differences": {},
    }
    if old_selected is not None:
        comparisons = {
            "eps": (
                float(old_selected["eps"]),
                EXPECTED_OLD["eps"],
            ),
            "min_samples": (
                int(old_selected["min_samples"]),
                EXPECTED_OLD["min_samples"],
            ),
            "clusters": (
                int(old_selected["clusters"]),
                EXPECTED_OLD["clusters"],
            ),
            "noise_count": (
                int(old_selected["noise_count"]),
                EXPECTED_OLD["noise_count"],
            ),
            "silhouette": (
                float(old_selected["silhouette"]),
                EXPECTED_OLD["silhouette"],
            ),
            "balance_ratio": (
                float(old_selected["balance_ratio"]),
                EXPECTED_OLD["balance_ratio"],
            ),
        }
        reproduction["differences"] = {
            key: {
                "current": current,
                "expected_saved": expected,
                "absolute_difference": float(abs(current - expected)),
            }
            for key, (current, expected) in comparisons.items()
        }
        reproduction["status"] = (
            "EXACT_OR_NEAR_EXACT_REPRODUCTION"
            if (
                reproduction["differences"]["eps"]["absolute_difference"] < 1e-8
                and reproduction["differences"]["min_samples"]["absolute_difference"] == 0
                and reproduction["differences"]["clusters"]["absolute_difference"] == 0
                and reproduction["differences"]["noise_count"]["absolute_difference"] == 0
                and reproduction["differences"]["silhouette"]["absolute_difference"] < 1e-8
            )
            else "NOT_EXACT_REPRODUCTION"
        )

    (out_dir / "archived_result_reproduction_audit.json").write_text(
        json.dumps(reproduction, indent=2),
        encoding="utf-8",
    )

    latex_lines = []
    for mode in [
        "native_variable_k",
        "matched_k5_strict",
    ]:
        row = selections[selections["mode"] == mode].iloc[0]
        if row["status"] != "SELECTED":
            latex_lines.append(
                f"{mode.replace('_', ' ')} & "
                r"\multicolumn{10}{c}{No eligible solution} \\"
            )
            continue
        latex_lines.append(
            f"{mode.replace('_', ' ')} & "
            f"{float(row['eps']):.4f} & "
            f"{int(row['min_samples'])} & "
            f"{int(row['clusters'])} & "
            f"{int(row['assigned_n'])} & "
            f"{int(row['noise_count'])} "
            f"({float(row['noise_percent']):.1f}\\%) & "
            f"{float(row['coverage_percent']):.1f}\\% & "
            f"{float(row['silhouette']):.4f} & "
            f"{float(row['davies_bouldin']):.4f} & "
            f"{int(row['minimum_cluster_size'])} & "
            f"{float(row['balance_ratio']):.2f} \\\\"
        )

    (out_dir / "dbscan_diagnostic_latex_rows.tex").write_text(
        "\n".join(latex_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "feature_file": feature_file.name,
        "metadata_file": metadata_file.name,
        "input_shape": list(x.shape),
        "random_state": RANDOM_STATE,
        "pca_components_retained": int(x_pca.shape[1]),
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "tsne": {
            "n_components": 3,
            "perplexity": perplexity,
            "learning_rate": "auto",
            "init": "pca",
        },
        "grid": {
            "eps_low": eps_low,
            "eps_high": eps_high,
            "eps_count": 40,
            "min_samples": [3, 20],
            "total_candidates": int(len(grid)),
        },
        "native_rules": {
            "clusters": "2 to 10",
            "noise_percent_max": args.native_noise_cap,
            "selection_score": (
                "silhouette - 0.002*noise_percent "
                "- 0.005*log1p(balance_ratio)"
            ),
        },
        "matched_k5_rules": {
            "clusters": 5,
            "coverage_percent_min": args.k5_min_coverage,
            "minimum_cluster_size_min": args.k5_min_cluster_size,
            "no_merging": True,
            "no_noise_reassignment": True,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn_version,
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("DBSCAN: ARCHIVED K-GUIDED, NATIVE VARIABLE-K, AND MATCHED K=5")
    print("=" * 96)
    print(selections.to_string(index=False))
    print("\nARCHIVED-RESULT REPRODUCTION AUDIT")
    print(json.dumps(reproduction, indent=2))
    print("\nLATEX ROWS")
    print(
        (out_dir / "dbscan_diagnostic_latex_rows.tex")
        .read_text(encoding="utf-8")
    )
    print("OUTPUT DIRECTORY")
    print(out_dir)

    if k5_strict_selected is None:
        print(
            "\nMAIN CONCLUSION: No non-degenerate K=5 DBSCAN solution "
            "satisfied the predefined coverage and minimum-size criteria."
        )
    else:
        print(
            "\nMAIN CONCLUSION: A strict naturally occurring K=5 "
            "DBSCAN solution was identified."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
