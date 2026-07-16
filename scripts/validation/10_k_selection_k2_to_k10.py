#!/usr/bin/env python3
"""
Evaluate candidate subtype counts K=2..10 on a saved EPG-SD UMAP representation.

This script does not rerun preprocessing, CNN-AE training, PCA, graph
construction, VGAE training, or UMAP. It:

1. loads the saved two-dimensional UMAP coordinates;
2. standardizes the two UMAP axes, matching the archived K-selection run;
3. applies nearest-neighbor spectral clustering for K=2..10;
4. computes internal validity, cluster balance, eigengap, and 80% subsampling
   stability; and
5. writes aggregate metrics, candidate labels, and diagnostic figures.

The graph adjacency is required for the eigengap calculation. No fallback graph
is constructed, because that would represent a different analysis.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-csv", required=True, type=Path)
    parser.add_argument("--adjacency", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--stability-repeats", type=int, default=50)
    parser.add_argument("--subsample-frac", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--umap-columns",
        nargs=2,
        default=None,
        metavar=("UMAP1", "UMAP2"),
        help="Two coordinate columns. Defaults to the first two numeric columns.",
    )
    return parser.parse_args()


def run_spectral(
    x: np.ndarray,
    k: int,
    graph_k: int,
    seed: int,
) -> np.ndarray:
    n_neighbors = min(graph_k, len(x) - 1)
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=seed,
    )
    return model.fit_predict(x)


def stability_for_k(
    x: np.ndarray,
    k: int,
    base_labels: np.ndarray,
    graph_k: int,
    seed: int,
    repeats: int,
    subsample_frac: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + k)
    n_subjects = x.shape[0]
    sample_size = int(round(subsample_frac * n_subjects))

    ari_values: list[float] = []
    nmi_values: list[float] = []

    for repeat in range(repeats):
        indices = rng.choice(n_subjects, size=sample_size, replace=False)
        labels = run_spectral(
            x[indices],
            k=k,
            graph_k=graph_k,
            seed=seed + repeat,
        )
        ari_values.append(adjusted_rand_score(base_labels[indices], labels))
        nmi_values.append(
            normalized_mutual_info_score(base_labels[indices], labels)
        )

    return {
        "stability_ARI_mean": float(np.mean(ari_values)),
        "stability_ARI_std": float(np.std(ari_values)),
        "stability_NMI_mean": float(np.mean(nmi_values)),
        "stability_NMI_std": float(np.std(nmi_values)),
    }


def main() -> None:
    args = parse_args()

    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("Require 2 <= k_min <= k_max.")
    if not 0.0 < args.subsample_frac < 1.0:
        raise ValueError("subsample_frac must lie strictly between 0 and 1.")
    if args.stability_repeats < 1:
        raise ValueError("stability_repeats must be positive.")

    umap_csv = args.umap_csv.expanduser().resolve()
    adjacency_file = args.adjacency.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not umap_csv.is_file():
        raise FileNotFoundError(f"Missing UMAP CSV: {umap_csv}")
    if not adjacency_file.is_file():
        raise FileNotFoundError(f"Missing adjacency array: {adjacency_file}")

    umap_df = pd.read_csv(umap_csv)
    if args.umap_columns is None:
        numeric_columns = (
            umap_df.select_dtypes(include=[np.number]).columns.tolist()
        )
        if len(numeric_columns) < 2:
            raise ValueError("The UMAP CSV needs at least two numeric columns.")
        coordinate_columns = numeric_columns[:2]
    else:
        coordinate_columns = list(args.umap_columns)
        missing = [c for c in coordinate_columns if c not in umap_df.columns]
        if missing:
            raise ValueError(f"Missing requested UMAP columns: {missing}")

    x_raw = umap_df[coordinate_columns].to_numpy(dtype=float)
    if not np.isfinite(x_raw).all():
        raise ValueError("UMAP coordinates contain non-finite values.")
    x = StandardScaler().fit_transform(x_raw)

    adjacency = np.load(adjacency_file)
    if adjacency.shape != (len(x), len(x)):
        raise ValueError(
            f"Adjacency shape {adjacency.shape} does not match "
            f"{len(x)} UMAP rows."
        )

    adjacency = np.asarray(adjacency, dtype=float)
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    adjacency_binary = (adjacency > 0).astype(float)

    normalized_laplacian = laplacian(adjacency_binary, normed=True)
    if hasattr(normalized_laplacian, "toarray"):
        normalized_laplacian = normalized_laplacian.toarray()

    eigenvalues = np.sort(eigh(normalized_laplacian, eigvals_only=True))
    eigengaps = np.diff(eigenvalues)

    eigengap_rows = []
    for k in range(1, min(31, len(eigengaps) + 1)):
        eigengap_rows.append(
            {
                "k": k,
                "lambda_k": float(eigenvalues[k - 1]),
                "lambda_k_plus_1": float(eigenvalues[k]),
                "eigengap_lambda_kplus1_minus_lambda_k": float(
                    eigengaps[k - 1]
                ),
            }
        )
    eigengap_df = pd.DataFrame(eigengap_rows)
    eigengap_df.to_csv(out / "eigengap.csv", index=False)

    rows = []
    k_range = list(range(args.k_min, args.k_max + 1))

    for k in k_range:
        labels = run_spectral(
            x,
            k=k,
            graph_k=args.graph_k,
            seed=args.seed,
        )

        unique, counts = np.unique(labels, return_counts=True)
        minimum = int(counts.min())
        maximum = int(counts.max())
        balance = float(maximum / maximum if minimum == 0 else maximum / minimum)

        stability = stability_for_k(
            x,
            k=k,
            base_labels=labels,
            graph_k=args.graph_k,
            seed=args.seed,
            repeats=args.stability_repeats,
            subsample_frac=args.subsample_frac,
        )

        eigengap_row = eigengap_df.loc[eigengap_df["k"] == k]
        if eigengap_row.empty:
            raise ValueError(f"No eigengap was available for K={k}.")
        eigengap = float(
            eigengap_row[
                "eigengap_lambda_kplus1_minus_lambda_k"
            ].iloc[0]
        )

        row = {
            "K": k,
            "silhouette": float(silhouette_score(x, labels)),
            "davies_bouldin": float(davies_bouldin_score(x, labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(x, labels)
            ),
            "eigengap": eigengap,
            "n_clusters_found": int(len(unique)),
            "min_cluster_size": minimum,
            "max_cluster_size": maximum,
            "cluster_size_ratio_max_min": balance,
            **stability,
        }
        rows.append(row)

        pd.DataFrame(
            {
                "subject_index": np.arange(len(labels)),
                "subtype": labels,
            }
        ).to_csv(out / f"labels_K{k}.csv", index=False)

        print(
            f"K={k:2d} | "
            f"Sil={row['silhouette']:.4f} | "
            f"DB={row['davies_bouldin']:.4f} | "
            f"CH={row['calinski_harabasz']:.2f} | "
            f"Gap={row['eigengap']:.6f} | "
            f"ARI={row['stability_ARI_mean']:.4f} | "
            f"NMI={row['stability_NMI_mean']:.4f} | "
            f"Min={minimum} | Balance={balance:.2f}"
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "k_selection_metrics.csv", index=False)

    ranked = metrics.copy()
    ranked["rank_silhouette"] = ranked["silhouette"].rank(ascending=False)
    ranked["rank_DB"] = ranked["davies_bouldin"].rank(ascending=True)
    ranked["rank_CH"] = ranked["calinski_harabasz"].rank(ascending=False)
    ranked["rank_eigengap"] = ranked["eigengap"].rank(ascending=False)
    ranked["rank_ARI"] = ranked["stability_ARI_mean"].rank(ascending=False)
    ranked["rank_NMI"] = ranked["stability_NMI_mean"].rank(ascending=False)
    ranked["rank_balance"] = ranked[
        "cluster_size_ratio_max_min"
    ].rank(ascending=True)
    ranked["rank_min_cluster"] = ranked[
        "min_cluster_size"
    ].rank(ascending=False)

    rank_columns = [
        "rank_silhouette",
        "rank_DB",
        "rank_CH",
        "rank_eigengap",
        "rank_ARI",
        "rank_NMI",
        "rank_balance",
        "rank_min_cluster",
    ]
    ranked["mean_rank_score_lower_is_better"] = ranked[
        rank_columns
    ].mean(axis=1)
    ranked.sort_values(
        "mean_rank_score_lower_is_better"
    ).to_csv(out / "k_selection_rank_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(metrics["K"], metrics["silhouette"], marker="o")
    plt.axvline(5, linestyle="--", label="Reported K=5")
    plt.xlabel("Number of subtypes K")
    plt.ylabel("Silhouette score")
    plt.title("Candidate subtype-resolution comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "k_selection_silhouette_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(metrics["K"], metrics["stability_ARI_mean"], marker="o")
    plt.axvline(5, linestyle="--", label="Reported K=5")
    plt.xlabel("Number of subtypes K")
    plt.ylabel("80% subsampling ARI")
    plt.title("Candidate-resolution stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "k_selection_stability_curve.png", dpi=300)
    plt.close()

    subset = eigengap_df[eigengap_df["k"].between(
        args.k_min,
        args.k_max,
    )]
    plt.figure(figsize=(7, 5))
    plt.plot(
        subset["k"],
        subset["eigengap_lambda_kplus1_minus_lambda_k"],
        marker="o",
    )
    plt.axvline(5, linestyle="--", label="Reported K=5")
    plt.xlabel("K")
    plt.ylabel(r"Eigengap $\lambda_{K+1}-\lambda_K$")
    plt.title("Graph-Laplacian eigengap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "eigengap_curve.png", dpi=300)
    plt.close()

    print(f"Saved validation outputs to: {out}")


if __name__ == "__main__":
    main()
