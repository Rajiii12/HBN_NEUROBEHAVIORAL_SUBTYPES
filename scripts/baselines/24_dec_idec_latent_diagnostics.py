#!/usr/bin/env python3
"""
Saved-output sanity checks for DEC and IDEC.

This script DOES NOT retrain DEC/IDEC. It audits the exact saved final
latents and labels already used for the manuscript table.

It computes:
- separation in the original standardized 196-D input under DEC/IDEC labels
- separation in the saved final DEC/IDEC latent spaces
- latent dimensional activity and effective rank
- exact/rounded/near-duplicate diagnostics
- within-cluster spread and centroid separation
- cluster sizes and balance
- PCA and UMAP visualizations of input and final latent spaces

It cannot recover:
- pretrained AE latents
- DEC/IDEC soft-assignment confidence
- IDEC reconstruction loss
- multi-seed stability

Those require the original training script/checkpoints and are intentionally
not guessed here.
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
from scipy.spatial.distance import pdist, squareform
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("DEC", "IDEC")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--node-features", type=Path, required=True)
    p.add_argument("--deep-output-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_labels(path: Path, expected_n: int) -> tuple[np.ndarray, str]:
    df = pd.read_csv(path)
    if len(df) != expected_n:
        raise ValueError(
            f"{path}: expected {expected_n} rows, found {len(df)}"
        )

    preferred = [
        c for c in df.columns
        if any(k in c.lower() for k in ("label", "cluster", "subtype"))
    ]
    candidates = preferred + [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in preferred
    ]

    for col in candidates:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().all():
            vals = x.to_numpy()
            if np.allclose(vals, np.round(vals)):
                labels = np.round(vals).astype(int)
                if 2 <= np.unique(labels).size <= 20:
                    return labels, col

    raise ValueError(
        f"Could not identify a discrete label column in {path}. "
        f"Columns: {list(df.columns)}"
    )


def safe_cluster_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values found in feature matrix.")
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= len(labels):
        raise ValueError("Invalid number of clusters for internal metrics.")
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }


def effective_rank(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    eig = s**2
    total = eig.sum()
    if total <= 0:
        return 0.0, 0.0, eig
    p = eig / total
    p_nonzero = p[p > 0]
    entropy_rank = float(np.exp(-(p_nonzero * np.log(p_nonzero)).sum()))
    participation_ratio = float(1.0 / np.square(p_nonzero).sum())
    return entropy_rank, participation_ratio, eig


def duplicate_diagnostics(z: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    zs = StandardScaler().fit_transform(z)
    n = len(zs)
    pair_count = n * (n - 1) // 2
    d = pdist(zs, metric="euclidean")

    summary: dict[str, Any] = {
        "n_rows": n,
        "exact_unique_rows": int(np.unique(z, axis=0).shape[0]),
        "exact_duplicate_rows": int(n - np.unique(z, axis=0).shape[0]),
    }

    for decimals in (8, 6, 4):
        unique_count = int(np.unique(np.round(zs, decimals), axis=0).shape[0])
        summary[f"unique_rows_rounded_{decimals}dp"] = unique_count
        summary[f"duplicate_rows_rounded_{decimals}dp"] = n - unique_count

    rows = []
    for tol in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        count = int(np.sum(d <= tol))
        rows.append({
            "distance_tolerance": tol,
            "near_duplicate_pairs": count,
            "all_pairs": pair_count,
            "near_duplicate_pair_fraction": (
                float(count / pair_count) if pair_count else float("nan")
            ),
        })

    summary["minimum_nonzero_pair_distance_standardized"] = (
        float(np.min(d[d > 0])) if np.any(d > 0) else 0.0
    )
    summary["median_pair_distance_standardized"] = float(np.median(d))
    return summary, pd.DataFrame(rows)


def cluster_geometry(z: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    unique = np.unique(labels)
    centroids = []
    rows = []

    for k in unique:
        xk = z[labels == k]
        centroid = xk.mean(axis=0)
        centroids.append((k, centroid))
        dist_centroid = np.linalg.norm(xk - centroid, axis=1)
        pairwise = pdist(xk) if len(xk) > 1 else np.array([0.0])
        cov_trace = float(np.var(xk, axis=0, ddof=1).sum()) if len(xk) > 1 else 0.0
        rows.append({
            "cluster": int(k),
            "n": int(len(xk)),
            "mean_distance_to_centroid": float(dist_centroid.mean()),
            "median_distance_to_centroid": float(np.median(dist_centroid)),
            "max_distance_to_centroid": float(dist_centroid.max()),
            "mean_within_pair_distance": float(pairwise.mean()),
            "median_within_pair_distance": float(np.median(pairwise)),
            "total_within_variance_trace": cov_trace,
        })

    centroid_matrix = np.vstack([c for _, c in centroids])
    centroid_dist = squareform(pdist(centroid_matrix))
    for i, (k, _) in enumerate(centroids):
        others = np.delete(centroid_dist[i], i)
        rows[i]["nearest_centroid_distance"] = float(others.min())
        denom = rows[i]["mean_distance_to_centroid"]
        rows[i]["nearest_centroid_to_mean_within_ratio"] = (
            float(others.min() / denom) if denom > 0 else float("inf")
        )

    return pd.DataFrame(rows)


def make_projections(
    x_input: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    method: str,
    outdir: Path,
) -> dict[str, Any]:
    pca_input = PCA(n_components=2, random_state=42)
    pca_latent = PCA(n_components=2, random_state=42)
    input_2d = pca_input.fit_transform(x_input)
    latent_2d = pca_latent.fit_transform(z)

    umap_available = False
    input_umap = None
    latent_umap = None
    umap_version = None
    try:
        import umap
        umap_version = getattr(umap, "__version__", "unknown")
        input_umap = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        ).fit_transform(x_input)
        latent_umap = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        ).fit_transform(z)
        umap_available = True
    except Exception as exc:
        print(f"WARNING: UMAP panels skipped for {method}: {exc}")

    if umap_available:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        panels = [
            (axes[0, 0], input_2d, f"{method} labels on original input PCA"),
            (axes[0, 1], latent_2d, f"{method} final latent PCA"),
            (axes[1, 0], input_umap, f"{method} labels on original input UMAP"),
            (axes[1, 1], latent_umap, f"{method} final latent UMAP"),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        panels = [
            (axes[0], input_2d, f"{method} labels on original input PCA"),
            (axes[1], latent_2d, f"{method} final latent PCA"),
        ]

    for ax, coords, title in panels:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        handles, legend_labels = scatter.legend_elements()
        ax.legend(handles, legend_labels, title="Cluster", loc="best")

    fig.tight_layout()
    png = outdir / f"{method}_saved_latent_diagnostics.png"
    pdf = outdir / f"{method}_saved_latent_diagnostics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "input_pca_variance_2d": float(pca_input.explained_variance_ratio_.sum()),
        "latent_pca_variance_2d": float(pca_latent.explained_variance_ratio_.sum()),
        "umap_available": umap_available,
        "umap_version": umap_version,
        "figure_png": png.name,
        "figure_pdf": pdf.name,
    }


def main() -> int:
    args = parse_args()
    input_path = args.node_features.expanduser().resolve()
    deep_dir = args.deep_output_dir.expanduser().resolve()
    outdir = args.out.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)


    if not input_path.exists():
        raise FileNotFoundError(f"Missing exact saved input: {input_path}")

    x_input = np.load(input_path)
    if x_input.shape != (430, 196):
        raise ValueError(
            f"Expected saved input shape (430, 196), found {x_input.shape}"
        )
    x_input_std = StandardScaler().fit_transform(x_input)

    all_summary = []
    all_cluster = []
    all_near = []
    manifest: dict[str, Any] = {
        "deep_output_dir": deep_dir.name,
        "input_path": input_path.name,
        "input_shape": list(x_input.shape),
        "methods": {},
        "limitations": [
            "Saved outputs do not contain pretrained AE latents.",
            "Saved outputs do not contain DEC/IDEC soft assignments.",
            "Saved outputs do not contain IDEC reconstruction history.",
            "Saved outputs represent a single run and cannot establish multi-seed stability.",
        ],
    }

    for method in METHODS:
        latent_path = deep_dir / f"{method}_latent.npy"
        label_path = deep_dir / f"{method}_labels.csv"
        size_path = deep_dir / f"{method}_cluster_sizes.csv"

        for p in (latent_path, label_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing {method} file: {p}")

        z = np.load(latent_path)
        if z.shape[0] != 430:
            raise ValueError(f"{method}: expected 430 latent rows, found {z.shape}")
        labels, label_col = load_labels(label_path, expected_n=430)

        input_metrics = safe_cluster_metrics(x_input_std, labels)
        latent_metrics = safe_cluster_metrics(z, labels)

        var = np.var(z, axis=0, ddof=1)
        entropy_rank, participation_ratio, eig = effective_rank(z)
        dup_summary, near_df = duplicate_diagnostics(z)
        geom_df = cluster_geometry(z, labels)
        proj = make_projections(x_input_std, z, labels, method, outdir)

        counts = pd.Series(labels).value_counts().sort_index()
        min_size = int(counts.min())
        max_size = int(counts.max())
        balance = float(max_size / min_size)

        summary = {
            "method": method,
            "n_subjects": int(z.shape[0]),
            "latent_dim": int(z.shape[1]),
            "label_column": label_col,
            "n_clusters": int(np.unique(labels).size),
            "cluster_sizes": json.dumps({int(k): int(v) for k, v in counts.items()}),
            "minimum_cluster_size": min_size,
            "maximum_cluster_size": max_size,
            "balance_ratio": balance,
            "input_silhouette": input_metrics["silhouette"],
            "input_davies_bouldin": input_metrics["davies_bouldin"],
            "input_calinski_harabasz": input_metrics["calinski_harabasz"],
            "final_latent_silhouette": latent_metrics["silhouette"],
            "final_latent_davies_bouldin": latent_metrics["davies_bouldin"],
            "final_latent_calinski_harabasz": latent_metrics["calinski_harabasz"],
            "minimum_latent_variance": float(var.min()),
            "median_latent_variance": float(np.median(var)),
            "maximum_latent_variance": float(var.max()),
            "active_dims_var_gt_1e-12": int(np.sum(var > 1e-12)),
            "active_dims_var_gt_1e-10": int(np.sum(var > 1e-10)),
            "active_dims_var_gt_1e-8": int(np.sum(var > 1e-8)),
            "effective_rank_entropy": entropy_rank,
            "effective_rank_participation": participation_ratio,
            **dup_summary,
            **proj,
        }
        all_summary.append(summary)

        geom_df.insert(0, "method", method)
        all_cluster.append(geom_df)

        near_df.insert(0, "method", method)
        all_near.append(near_df)

        manifest["methods"][method] = {
            "latent_path": latent_path.name,
            "label_path": label_path.name,
            "cluster_size_path": size_path.name if size_path.exists() else None,
            "latent_shape": list(z.shape),
            "label_column": label_col,
            "cluster_sizes": {int(k): int(v) for k, v in counts.items()},
        }

    summary_df = pd.DataFrame(all_summary)
    cluster_df = pd.concat(all_cluster, ignore_index=True)
    near_df = pd.concat(all_near, ignore_index=True)

    summary_path = outdir / "dec_idec_saved_latent_summary.csv"
    cluster_path = outdir / "dec_idec_cluster_geometry.csv"
    near_path = outdir / "dec_idec_near_duplicate_pairs.csv"
    summary_df.to_csv(summary_path, index=False)
    cluster_df.to_csv(cluster_path, index=False)
    near_df.to_csv(near_path, index=False)

    latex_lines = []
    for _, r in summary_df.iterrows():
        latex_lines.append(
            f"{r['method']} & "
            f"{r['input_silhouette']:.4f} & "
            f"{r['final_latent_silhouette']:.4f} & "
            f"{int(r['active_dims_var_gt_1e-8'])}/{int(r['latent_dim'])} & "
            f"{int(r['duplicate_rows_rounded_6dp'])} & "
            f"{r['effective_rank_participation']:.2f} & "
            f"{int(r['minimum_cluster_size'])} & "
            f"{r['balance_ratio']:.2f} \\\\"
        )
    latex_path = outdir / "dec_idec_saved_single_run_rows.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    manifest.update({
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn_version,
        "outputs": {
            "summary": summary_path.name,
            "cluster_geometry": cluster_path.name,
            "near_duplicates": near_path.name,
            "latex_rows": latex_path.name,
        },
        "statement": (
            "This audit uses the exact saved final DEC/IDEC latents and labels. "
            "It does not retrain or alter any deep-clustering model."
        ),
    })
    manifest_path = outdir / "diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("DEC / IDEC SAVED-LATENT SANITY CHECK")
    print("=" * 92)
    display_cols = [
        "method",
        "input_silhouette",
        "final_latent_silhouette",
        "final_latent_davies_bouldin",
        "final_latent_calinski_harabasz",
        "active_dims_var_gt_1e-8",
        "latent_dim",
        "effective_rank_participation",
        "exact_duplicate_rows",
        "duplicate_rows_rounded_6dp",
        "minimum_cluster_size",
        "balance_ratio",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("\nSAVED-SINGLE-RUN TABLE ROWS")
    print(latex_path.read_text(encoding="utf-8"))
    print("OUTPUT DIRECTORY")
    print(outdir)
    print("\nIMPORTANT LIMITATION")
    print(
        "This phase cannot establish multi-seed stability, assignment confidence, "
        "pretrained-latent separation, or IDEC reconstruction preservation. "
        "Those require the original training script/checkpoints."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
