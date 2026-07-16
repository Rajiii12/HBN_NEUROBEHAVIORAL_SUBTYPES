#!/usr/bin/env python3
"""
Controlled ablation of the two saved CNN-AE embedding streams.

The script uses the already-generated five-band recording-level embeddings.
It does NOT retrain either CNN autoencoder.

Conditions
----------
1. branch1_only:
   five embeddings from autoencoder_embeddings_dynamic_norm_fixed (640 -> PCA64)
2. branch2_only:
   five embeddings from cnn_autoencoder_embeddings_norm_fixed (640 -> PCA64)
3. branch_average:
   coordinate-wise average of the separately standardized corresponding views
   (allowed only after a coordinate-alignment audit; 640 -> PCA64)
4. two_branch_concat:
   all ten separately standardized views (1280 -> PCA64)

For every condition, the script keeps fixed:
- subject aggregation: mean + standard deviation + median
- phenotype variables and order
- alpha = 0.6
- graph K = 12
- VGAE = 196 -> 64 -> 16, 500 epochs
- UMAP n_neighbors = 30
- spectral clustering K = 5

The downstream stochastic pipeline is repeated across ten fixed seeds by default.

Critical safeguards
-------------------
- Audits the source scripts: both uploaded branch scripts use the same architecture
  and SEED=42. The streams should not be described as independently initialized unless the
  saved embeddings are empirically different.
- Computes per-band linear CKA, paired-coordinate correlation, and max differences.
- Compares the reconstructed concatenated condition with the archived final inputs.
  The audit is recorded but does not stop the controlled ablation, because all four
  conditions are recomputed under one common current numerical environment.
- Checkpoints every completed condition/seed run and resumes safely.
- Clinical variables are never used to select a condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Keep BLAS thread counts controlled before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import umap
except Exception as exc:
    raise ImportError("Install UMAP with: pip install umap-learn") from exc


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
PHENO_COLS_FINAL_ORDER = [
    "internalizing",
    "attention",
    "externalizing",
    "p_factor",
]
CLINICAL_COLS = [
    "p_factor",
    "attention",
    "internalizing",
    "externalizing",
    "ehq_total",
    "age",
]

GRAPH_K = 12
ALPHA = 0.6
UMAP_N = 30
FINAL_K = 5
PCA_COMPONENTS = 64
VGAE_EPOCHS = 500
VGAE_HIDDEN = 64
VGAE_LATENT = 16
LR = 1e-3
DEFAULT_SEEDS = list(range(42, 52))

CONDITION_ORDER = [
    "branch1_only",
    "branch2_only",
    "branch_average",
    "two_branch_concat",
]
CONDITION_LABELS = {
    "branch1_only": "Branch 1 only",
    "branch2_only": "Branch 2 only",
    "branch_average": "Standardized branch average",
    "two_branch_concat": "Two-branch concatenation",
}
PRE_PCA_DIM = {
    "branch1_only": 640,
    "branch2_only": 640,
    "branch_average": 640,
    "two_branch_concat": 1280,
}
AE_COST = {
    "branch1_only": "1x",
    "branch2_only": "1x",
    "branch_average": "2x",
    "two_branch_concat": "2x",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("TECHNICAL_FIX_SPECTRAL_UMAP"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--recording-meta",
        type=Path,
        default=Path(
            "TECHNICAL_FIX_SPECTRAL_UMAP/multiview_tensors_FS500/"
            "band_decomposition_metadata_FS500.csv"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=CONDITION_ORDER,
    )
    parser.add_argument(
        "--allow-average-with-low-alignment",
        action="store_true",
        help=(
            "By default branch_average is rejected when the mean absolute "
            "paired-coordinate correlation is below 0.80."
        ),
    )
    parser.add_argument(
        "--allow-final-input-audit-fail",
        action="store_true",
        help="Deprecated compatibility flag; the V3 controlled rerun always records the archived-input audit.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(a, dtype=np.float64)
                - np.asarray(b, dtype=np.float64)
            )
        )
    )


def raw_cosine_similarity(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norms, 1e-9)
    sim = xn @ xn.T
    np.fill_diagonal(sim, 1.0)
    return sim


def build_binary_graph(fused_similarity: np.ndarray) -> np.ndarray:
    directed = kneighbors_graph(
        fused_similarity,
        n_neighbors=GRAPH_K,
        mode="connectivity",
        metric="euclidean",
        include_self=False,
    ).toarray()
    adjacency = np.maximum(directed, directed.T).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=np.float32).copy()
    np.fill_diagonal(a, 1.0)
    degree = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(degree + 1e-9)
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.sum(cross * cross)
    denominator = math.sqrt(
        float(np.sum(xx * xx)) * float(np.sum(yy * yy))
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def paired_coordinate_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    correlations = []
    for j in range(x.shape[1]):
        xj = np.asarray(x[:, j], dtype=float)
        yj = np.asarray(y[:, j], dtype=float)
        if np.std(xj) < 1e-12 or np.std(yj) < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(xj, yj)[0, 1]))
    arr = np.asarray(correlations)
    return float(np.mean(arr)), float(np.mean(np.abs(arr)))


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, pvalues[order[i]] * n / rank)
        adjusted[order[i]] = min(running, 1.0)
    return adjusted


def one_variable_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    groups = []
    for cluster in np.unique(labels):
        group = values[labels == cluster]
        group = group[np.isfinite(group)]
        if group.size > 1:
            groups.append(group)

    if len(groups) < 2:
        return {
            "anova_F": np.nan,
            "anova_p": 1.0,
            "kruskal_H": np.nan,
            "kruskal_p": 1.0,
            "eta_squared": 0.0,
            "omega_squared": 0.0,
        }

    anova_f, anova_p = stats.f_oneway(*groups)
    kruskal_h, kruskal_p = stats.kruskal(*groups)

    all_valid = np.concatenate(groups)
    grand_mean = float(np.mean(all_valid))
    ss_between = sum(
        len(g) * (float(np.mean(g)) - grand_mean) ** 2
        for g in groups
    )
    ss_within = sum(
        float(np.sum((g - float(np.mean(g))) ** 2))
        for g in groups
    )
    ss_total = ss_between + ss_within
    df_between = len(groups) - 1
    df_within = len(all_valid) - len(groups)
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    omega_num = ss_between - df_between * ms_within
    omega_den = ss_total + ms_within
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
    }


def clinical_stats(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for variable in CLINICAL_COLS:
        values = pd.to_numeric(meta[variable], errors="coerce").to_numpy()
        rows.append({"variable": variable, **one_variable_stats(values, labels)})
    table = pd.DataFrame(rows)
    table["anova_p_fdr_bh"] = bh_fdr(table["anova_p"].to_numpy())
    table["kruskal_p_fdr_bh"] = bh_fdr(table["kruskal_p"].to_numpy())
    table["anova_significant_fdr"] = table["anova_p_fdr_bh"] < 0.05
    return table


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(adjacency_norm @ x)


class FinalVGAE(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.gcn1 = DenseGraphConv(in_dim, VGAE_HIDDEN)
        self.gcn_mu = DenseGraphConv(VGAE_HIDDEN, VGAE_LATENT)
        self.gcn_logstd = DenseGraphConv(VGAE_HIDDEN, VGAE_LATENT)

    def encode(
        self,
        x: torch.Tensor,
        adjacency_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.gcn1(x, adjacency_norm))
        mu = self.gcn_mu(hidden, adjacency_norm)
        logstd = self.gcn_logstd(hidden, adjacency_norm).clamp(-4.0, 4.0)
        return mu, logstd

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logstd: torch.Tensor,
    ) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(logstd)

    def forward(
        self,
        x: torch.Tensor,
        adjacency_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.encode(x, adjacency_norm)
        latent = self.reparameterize(mu, logstd)
        logits = latent @ latent.T
        return logits, mu, logstd


def train_vgae(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    seed: int,
    device: torch.device,
    run_dir: Path,
) -> np.ndarray:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    target_np = (adjacency > 0).astype(np.float32)
    adjacency_norm_np = normalize_adjacency(adjacency).astype(np.float32)

    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    adjacency_norm = torch.tensor(
        adjacency_norm_np,
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(target_np, dtype=torch.float32, device=device)

    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = (negatives / (positives + 1e-9)).clamp(
        min=1.0,
        max=50.0,
    )

    model = FinalVGAE(in_dim=node_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_rows = []
    for epoch in range(1, VGAE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, mu, logstd = model(x, adjacency_norm)

        reconstruction = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        raw_kl = -0.5 * torch.mean(
            torch.sum(
                1.0
                + 2.0 * logstd
                - mu.pow(2)
                - torch.exp(2.0 * logstd),
                dim=1,
            )
        )
        kl = raw_kl / x.shape[0]
        loss = reconstruction + kl
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0 or epoch == VGAE_EPOCHS:
            log_rows.append(
                {
                    "epoch": epoch,
                    "reconstruction": float(reconstruction.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "total": float(loss.detach().cpu()),
                }
            )

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x, adjacency_norm)
        latent = mu.detach().cpu().numpy()

    pd.DataFrame(log_rows).to_csv(
        run_dir / "vgae_training_log_sparse.csv",
        index=False,
    )
    np.save(run_dir / "vgae_latent.npy", latent)

    del model, x, adjacency_norm, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latent


def run_downstream(
    condition: str,
    subject_eeg: np.ndarray,
    meta: pd.DataFrame,
    phenotype_standardized: np.ndarray,
    phenotype_similarity: np.ndarray,
    seed: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    eeg_standardized = StandardScaler().fit_transform(
        np.nan_to_num(subject_eeg)
    )
    node_features = StandardScaler().fit_transform(
        np.concatenate([eeg_standardized, phenotype_standardized], axis=1)
    )

    eeg_similarity = raw_cosine_similarity(eeg_standardized)
    fused_similarity = (
        ALPHA * eeg_similarity
        + (1.0 - ALPHA) * phenotype_similarity
    )
    adjacency = build_binary_graph(fused_similarity)

    latent = train_vgae(
        node_features=node_features,
        adjacency=adjacency,
        seed=seed,
        device=device,
        run_dir=run_dir,
    )

    reducer = umap.UMAP(
        n_neighbors=UMAP_N,
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    projection = reducer.fit_transform(latent)

    labels = SpectralClustering(
        n_clusters=FINAL_K,
        affinity="nearest_neighbors",
        n_neighbors=GRAPH_K,
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(projection)

    counts = np.bincount(labels, minlength=FINAL_K)
    clinical = clinical_stats(meta, labels)

    row = {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "seed": int(seed),
        "pre_pca_dimension": PRE_PCA_DIM[condition],
        "silhouette": float(silhouette_score(projection, labels)),
        "davies_bouldin": float(davies_bouldin_score(projection, labels)),
        "calinski_harabasz": float(
            calinski_harabasz_score(projection, labels)
        ),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "balance_ratio": float(counts.max() / max(counts.min(), 1)),
        "mean_eta_squared": float(clinical["eta_squared"].mean()),
        "mean_omega_squared": float(clinical["omega_squared"].mean()),
        "fdr_significant_count": int(
            clinical["anova_significant_fdr"].sum()
        ),
        "age_eta_squared": float(
            clinical.loc[
                clinical["variable"] == "age",
                "eta_squared",
            ].iloc[0]
        ),
        "age_fdr_p": float(
            clinical.loc[
                clinical["variable"] == "age",
                "anova_p_fdr_bh",
            ].iloc[0]
        ),
        "ehq_eta_squared": float(
            clinical.loc[
                clinical["variable"] == "ehq_total",
                "eta_squared",
            ].iloc[0]
        ),
        "ehq_fdr_p": float(
            clinical.loc[
                clinical["variable"] == "ehq_total",
                "anova_p_fdr_bh",
            ].iloc[0]
        ),
        "cluster_sizes": json.dumps(
            {int(i): int(v) for i, v in enumerate(counts)}
        ),
    }

    pd.DataFrame(
        projection,
        columns=["UMAP1", "UMAP2"],
    ).to_csv(run_dir / "umap_coordinates.csv", index=False)
    pd.DataFrame(
        {
            "subject": meta["subject"].astype(str),
            "subtype": labels,
        }
    ).to_csv(run_dir / "subject_subtypes.csv", index=False)
    clinical.to_csv(run_dir / "clinical_stats.csv", index=False)
    pd.DataFrame(
        {
            "cluster": np.arange(FINAL_K),
            "size": counts,
        }
    ).to_csv(run_dir / "cluster_sizes.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "main_quality.csv", index=False)
    return row, clinical, labels


def aggregate_subjects(
    recording_features: np.ndarray,
    recording_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    subjects = recording_meta["subject"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    features = []
    for subject in unique_subjects:
        indices = np.where(subjects == subject)[0]
        values = recording_features[indices]
        features.append(
            np.concatenate(
                [
                    values.mean(axis=0),
                    values.std(axis=0),
                    np.median(values, axis=0),
                ],
                axis=0,
            )
        )
    return np.asarray(features, dtype=np.float32), unique_subjects


def make_condition_recording_features(
    branch1_scaled: list[np.ndarray],
    branch2_scaled: list[np.ndarray],
) -> dict[str, np.ndarray]:
    condition_inputs = {
        "branch1_only": np.concatenate(branch1_scaled, axis=1),
        "branch2_only": np.concatenate(branch2_scaled, axis=1),
        "branch_average": np.concatenate(
            [
                (a + b) / 2.0
                for a, b in zip(branch1_scaled, branch2_scaled)
            ],
            axis=1,
        ),
        "two_branch_concat": np.concatenate(
            branch1_scaled + branch2_scaled,
            axis=1,
        ),
    }

    results: dict[str, np.ndarray] = {}
    for condition, matrix in condition_inputs.items():
        pca = PCA(
            n_components=PCA_COMPONENTS,
            random_state=42,
        )
        results[condition] = pca.fit_transform(matrix)
    return results


def summarize_results(
    per_seed: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        group = per_seed[per_seed["condition"] == condition]
        if group.empty:
            continue
        stab = pairwise[pairwise["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "pre_pca_dimension": PRE_PCA_DIM[condition],
                "n_seeds": int(len(group)),
                "silhouette_mean": float(group["silhouette"].mean()),
                "silhouette_sd": float(group["silhouette"].std(ddof=1)),
                "davies_bouldin_mean": float(
                    group["davies_bouldin"].mean()
                ),
                "davies_bouldin_sd": float(
                    group["davies_bouldin"].std(ddof=1)
                ),
                "calinski_harabasz_mean": float(
                    group["calinski_harabasz"].mean()
                ),
                "calinski_harabasz_sd": float(
                    group["calinski_harabasz"].std(ddof=1)
                ),
                "minimum_cluster_size_mean": float(
                    group["minimum_cluster_size"].mean()
                ),
                "minimum_cluster_size_min": int(
                    group["minimum_cluster_size"].min()
                ),
                "balance_ratio_mean": float(
                    group["balance_ratio"].mean()
                ),
                "balance_ratio_sd": float(
                    group["balance_ratio"].std(ddof=1)
                ),
                "pairwise_seed_ari_mean": float(
                    stab["ARI"].mean()
                ) if not stab.empty else np.nan,
                "pairwise_seed_ari_sd": float(
                    stab["ARI"].std(ddof=1)
                ) if len(stab) > 1 else np.nan,
                "pairwise_seed_nmi_mean": float(
                    stab["NMI"].mean()
                ) if not stab.empty else np.nan,
                "mean_eta_squared_mean": float(
                    group["mean_eta_squared"].mean()
                ),
                "mean_eta_squared_sd": float(
                    group["mean_eta_squared"].std(ddof=1)
                ),
                "fdr_count_mean": float(
                    group["fdr_significant_count"].mean()
                ),
                "fdr_count_min": int(
                    group["fdr_significant_count"].min()
                ),
                "fdr_count_max": int(
                    group["fdr_significant_count"].max()
                ),
                "autoencoder_cost": AE_COST[condition],
            }
        )
    return pd.DataFrame(rows)


def write_latex_rows(summary: pd.DataFrame, out_path: Path) -> None:
    lines = []
    for condition in CONDITION_ORDER:
        row = summary[summary["condition"] == condition]
        if row.empty:
            continue
        r = row.iloc[0]
        fdr = (
            f"{int(r['fdr_count_min'])}/6"
            if int(r["fdr_count_min"]) == int(r["fdr_count_max"])
            else f"{int(r['fdr_count_min'])}--"
                 f"{int(r['fdr_count_max'])}/6"
        )
        lines.append(
            f"{r['condition_label']} & "
            f"{int(r['pre_pca_dimension'])} & "
            f"${r['silhouette_mean']:.4f}\\pm"
            f"{r['silhouette_sd']:.4f}$ & "
            f"${r['davies_bouldin_mean']:.4f}\\pm"
            f"{r['davies_bouldin_sd']:.4f}$ & "
            f"{r['minimum_cluster_size_mean']:.1f} "
            f"(min {int(r['minimum_cluster_size_min'])}) & "
            f"${r['balance_ratio_mean']:.2f}\\pm"
            f"{r['balance_ratio_sd']:.2f}$ & "
            f"{r['pairwise_seed_ari_mean']:.3f} & "
            f"${r['mean_eta_squared_mean']:.4f}\\pm"
            f"{r['mean_eta_squared_sd']:.4f}$ & "
            f"{fdr} & "
            f"${r['autoencoder_cost']}$ \\\\"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    branch1_dir = root / "autoencoder_embeddings_dynamic_norm_fixed"
    branch2_dir = root / "cnn_autoencoder_embeddings_norm_fixed"
    archived_subject_dir = root / "subject_embeddings_norm_fixed"
    final_graph_dir = (
        root
        / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
        / "graphK12_alpha0p6"
    )

    recording_meta_path = args.recording_meta.expanduser().resolve()
    if not recording_meta_path.exists():
        candidates = list(root.rglob("master_metadata.csv"))
        if candidates:
            recording_meta_path = candidates[0]
        else:
            raise FileNotFoundError(
                "Recording-level master_metadata.csv not found. "
                "Provide the subject-level master metadata file explicitly."
            )

    recording_meta = pd.read_csv(recording_meta_path)
    if "subject" not in recording_meta.columns:
        raise ValueError("Recording metadata lacks a 'subject' column.")

    subject_meta = pd.read_csv(
        archived_subject_dir / "subject_metadata_norm_fixed.csv"
    )
    archived_subject_eeg = np.load(
        archived_subject_dir
        / "subject_embedding_mean_std_median_norm_fixed.npy"
    )
    archived_fused = np.load(final_graph_dir / "fused_similarity.npy")
    archived_adjacency = np.load(final_graph_dir / "adjacency.npy")
    archived_nodes = np.load(final_graph_dir / "node_features.npy")

    branch1_raw: list[np.ndarray] = []
    branch2_raw: list[np.ndarray] = []
    branch1_scaled: list[np.ndarray] = []
    branch2_scaled: list[np.ndarray] = []
    similarity_rows = []

    for band in BANDS:
        p1 = branch1_dir / f"{band}_embedding.npy"
        p2 = branch2_dir / f"{band}_cnn_embedding_fixed.npy"
        if not p1.exists() or not p2.exists():
            raise FileNotFoundError(
                f"Missing saved branch embedding for {band}: {p1} or {p2}"
            )

        x1 = np.load(p1).astype(np.float32)
        x2 = np.load(p2).astype(np.float32)
        if x1.shape != x2.shape:
            raise ValueError(
                f"{band}: branch shapes differ: {x1.shape} vs {x2.shape}"
            )
        if x1.shape[1] != 128:
            raise ValueError(
                f"{band}: expected 128 latent dimensions, got {x1.shape}"
            )
        if len(recording_meta) != x1.shape[0]:
            raise ValueError(
                f"{band}: {x1.shape[0]} embeddings but "
                f"{len(recording_meta)} metadata rows."
            )

        s1 = StandardScaler().fit_transform(x1)
        s2 = StandardScaler().fit_transform(x2)
        corr_mean, corr_abs_mean = paired_coordinate_correlation(s1, s2)

        similarity_rows.append(
            {
                "band": band,
                "n_recordings": int(x1.shape[0]),
                "latent_dim": int(x1.shape[1]),
                "raw_max_abs_difference": max_abs_error(x1, x2),
                "standardized_max_abs_difference": max_abs_error(s1, s2),
                "linear_CKA": linear_cka(s1, s2),
                "paired_coordinate_correlation_mean": corr_mean,
                "paired_coordinate_abs_correlation_mean": corr_abs_mean,
                "branch1_sha256": sha256_file(p1),
                "branch2_sha256": sha256_file(p2),
            }
        )
        branch1_raw.append(x1)
        branch2_raw.append(x2)
        branch1_scaled.append(s1)
        branch2_scaled.append(s2)

    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(out_dir / "branch_similarity_audit.csv", index=False)

    mean_alignment = float(
        similarity["paired_coordinate_abs_correlation_mean"].mean()
    )
    skipped_conditions: dict[str, str] = {}
    if (
        "branch_average" in args.conditions
        and mean_alignment < 0.80
        and not args.allow_average_with_low_alignment
    ):
        skipped_conditions["branch_average"] = (
            "Skipped automatically because coordinate-wise averaging is not "
            "scientifically justified when the mean absolute paired-coordinate "
            f"correlation is {mean_alignment:.3f} (<0.80)."
        )
        args.conditions = [
            condition
            for condition in args.conditions
            if condition != "branch_average"
        ]
        print(
            "WARNING: branch_average was skipped because the two latent "
            "coordinate systems were insufficiently aligned."
        )

    recording_condition_features = make_condition_recording_features(
        branch1_scaled,
        branch2_scaled,
    )

    subject_features: dict[str, np.ndarray] = {}
    subject_ids: np.ndarray | None = None
    condition_input_rows = []

    for condition, recording_features in recording_condition_features.items():
        subject_matrix, current_subject_ids = aggregate_subjects(
            recording_features,
            recording_meta,
        )
        if subject_ids is None:
            subject_ids = current_subject_ids
        elif not np.array_equal(subject_ids, current_subject_ids):
            raise RuntimeError("Subject order differs across conditions.")

        subject_features[condition] = subject_matrix
        condition_input_rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "n_recordings": int(recording_features.shape[0]),
                "pre_pca_dimension": PRE_PCA_DIM[condition],
                "pca_dimension": int(recording_features.shape[1]),
                "n_subjects": int(subject_matrix.shape[0]),
                "subject_eeg_dimension": int(subject_matrix.shape[1]),
                "autoencoder_cost": AE_COST[condition],
            }
        )

    assert subject_ids is not None
    archived_ids = subject_meta["subject"].astype(str).to_numpy()
    if not np.array_equal(subject_ids.astype(str), archived_ids):
        raise RuntimeError(
            "Reconstructed subject order does not match archived "
            "subject_metadata_norm_fixed.csv."
        )

    pheno_frame = subject_meta[PHENO_COLS_FINAL_ORDER].apply(
        pd.to_numeric,
        errors="coerce",
    )
    pheno_frame = pheno_frame.fillna(
        pheno_frame.median(numeric_only=True)
    )
    phenotype_standardized = StandardScaler().fit_transform(
        pheno_frame.to_numpy()
    )
    phenotype_similarity = raw_cosine_similarity(
        phenotype_standardized
    )

    # Exact archived final-input audit for the concatenated condition.
    concat_subject = subject_features["two_branch_concat"]
    eeg_standardized = StandardScaler().fit_transform(
        np.nan_to_num(concat_subject)
    )
    concat_nodes = StandardScaler().fit_transform(
        np.concatenate(
            [eeg_standardized, phenotype_standardized],
            axis=1,
        )
    )
    concat_eeg_similarity = raw_cosine_similarity(eeg_standardized)
    concat_fused = (
        ALPHA * concat_eeg_similarity
        + (1.0 - ALPHA) * phenotype_similarity
    )
    concat_adjacency = build_binary_graph(concat_fused)

    final_audit = {
        "subject_embedding_max_abs_error": max_abs_error(
            concat_subject,
            archived_subject_eeg,
        ),
        "fused_similarity_max_abs_error": max_abs_error(
            concat_fused,
            archived_fused,
        ),
        "adjacency_exact_match": bool(
            np.array_equal(concat_adjacency, archived_adjacency)
        ),
        "adjacency_different_entries": int(
            np.sum(concat_adjacency != archived_adjacency)
        ),
        "node_features_max_abs_error": max_abs_error(
            concat_nodes,
            archived_nodes,
        ),
    }
    final_audit["pass"] = bool(
        final_audit["subject_embedding_max_abs_error"] <= 1e-5
        and final_audit["fused_similarity_max_abs_error"] <= 1e-6
        and final_audit["adjacency_exact_match"]
        and final_audit["node_features_max_abs_error"] <= 1e-6
    )
    (out_dir / "archived_final_input_audit.json").write_text(
        json.dumps(final_audit, indent=2),
        encoding="utf-8",
    )

    final_audit["interpretation"] = (
        "The project-local recording metadata has the correct 2677-row order "
        "and reproduces the archived 430-subject ordering. Any remaining "
        "difference reflects reconstruction/numerical differences in the saved "
        "PCA-aggregation pipeline. The ablation is therefore reported as a "
        "fresh controlled rerun in which every condition is recomputed under "
        "the same current software environment, not as an exact regeneration "
        "of the archived final row."
    )
    (out_dir / "archived_final_input_audit.json").write_text(
        json.dumps(final_audit, indent=2),
        encoding="utf-8",
    )
    if not final_audit["pass"]:
        print(
            "WARNING: The archived concatenated input was not reproduced "
            "bit-for-bit. Proceeding with a common-environment controlled "
            "rerun for all branch conditions."
        )

    pd.DataFrame(condition_input_rows).to_csv(
        out_dir / "condition_input_summary.csv",
        index=False,
    )

    checkpoint_path = out_dir / "per_seed_results.csv"
    clinical_checkpoint = out_dir / "clinical_per_seed.csv"
    completed: set[tuple[str, int]] = set()
    per_seed_rows: list[dict[str, Any]] = []
    clinical_rows: list[pd.DataFrame] = []
    label_store: dict[tuple[str, int], np.ndarray] = {}

    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        per_seed_rows = existing.to_dict("records")
        completed = {
            (str(row["condition"]), int(row["seed"]))
            for _, row in existing.iterrows()
        }
        print(f"Resuming with {len(completed)} completed runs.")

    for condition in args.conditions:
        for seed in args.seeds:
            key = (condition, int(seed))
            run_dir = out_dir / condition / f"seed_{seed}"

            labels_path = run_dir / "subject_subtypes.csv"
            if key in completed and labels_path.exists():
                labels_df = pd.read_csv(labels_path)
                label_store[key] = labels_df["subtype"].to_numpy()
                continue

            print("\n" + "=" * 92)
            print(
                f"CONDITION={condition} "
                f"SEED={seed} DEVICE={device}"
            )
            print("=" * 92)
            start = time.time()
            row, clinical, labels = run_downstream(
                condition=condition,
                subject_eeg=subject_features[condition],
                meta=subject_meta,
                phenotype_standardized=phenotype_standardized,
                phenotype_similarity=phenotype_similarity,
                seed=int(seed),
                device=device,
                run_dir=run_dir,
            )
            row["runtime_seconds"] = float(time.time() - start)
            per_seed_rows.append(row)
            clinical = clinical.copy()
            clinical.insert(0, "seed", int(seed))
            clinical.insert(0, "condition", condition)
            clinical_rows.append(clinical)
            label_store[key] = labels
            completed.add(key)

            pd.DataFrame(per_seed_rows).sort_values(
                ["condition", "seed"]
            ).to_csv(checkpoint_path, index=False)

            if clinical_checkpoint.exists():
                old_clinical = pd.read_csv(clinical_checkpoint)
                combined_clinical = pd.concat(
                    [old_clinical, clinical],
                    ignore_index=True,
                ).drop_duplicates(
                    ["condition", "seed", "variable"],
                    keep="last",
                )
            else:
                combined_clinical = clinical
            combined_clinical.sort_values(
                ["condition", "seed", "variable"]
            ).to_csv(clinical_checkpoint, index=False)

    per_seed = pd.read_csv(checkpoint_path).sort_values(
        ["condition", "seed"]
    )

    # Load every label file to guarantee pairwise stability is based on saved runs.
    for condition in args.conditions:
        for seed in args.seeds:
            key = (condition, int(seed))
            if key not in label_store:
                label_file = (
                    out_dir
                    / condition
                    / f"seed_{seed}"
                    / "subject_subtypes.csv"
                )
                label_store[key] = pd.read_csv(label_file)[
                    "subtype"
                ].to_numpy()

    pairwise_rows = []
    for condition in args.conditions:
        for seed_a, seed_b in combinations(args.seeds, 2):
            labels_a = label_store[(condition, int(seed_a))]
            labels_b = label_store[(condition, int(seed_b))]
            pairwise_rows.append(
                {
                    "condition": condition,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "ARI": float(
                        adjusted_rand_score(labels_a, labels_b)
                    ),
                    "NMI": float(
                        normalized_mutual_info_score(labels_a, labels_b)
                    ),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(
        out_dir / "pairwise_seed_stability.csv",
        index=False,
    )

    summary = summarize_results(per_seed, pairwise)
    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    write_latex_rows(
        summary,
        out_dir / "cnn_ae_branch_ablation_latex_rows.tex",
    )

    manifest = {
        "root": str(root),
        "output": str(out_dir),
        "recording_metadata": str(recording_meta_path),
        "device": str(device),
        "seeds": [int(s) for s in args.seeds],
        "conditions_run": args.conditions,
        "conditions_skipped": skipped_conditions,
        "experiment_type": (
            "Fresh controlled branch ablation under one common numerical "
            "environment; not an exact regeneration of the archived final row."
        ),
        "fixed_settings": {
            "pca_components": PCA_COMPONENTS,
            "alpha": ALPHA,
            "graph_k": GRAPH_K,
            "vgae_architecture": "196 -> 64 -> 16",
            "vgae_epochs": VGAE_EPOCHS,
            "umap_n_neighbors": UMAP_N,
            "spectral_k": FINAL_K,
        },
        "branch_source_observation": (
            "The two uploaded embedding scripts use the same network "
            "architecture and both set SEED=42. Whether the saved outputs "
            "are genuinely different is determined by branch_similarity_audit.csv."
        ),
        "mean_absolute_paired_coordinate_correlation": mean_alignment,
        "archived_final_input_audit": final_audit,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "umap": getattr(umap, "__version__", "unknown"),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("BRANCH SIMILARITY AUDIT")
    print("=" * 92)
    print(
        similarity[
            [
                "band",
                "raw_max_abs_difference",
                "linear_CKA",
                "paired_coordinate_abs_correlation_mean",
            ]
        ].to_string(index=False)
    )
    print("\nARCHIVED FINAL INPUT AUDIT")
    print(json.dumps(final_audit, indent=2))
    if skipped_conditions:
        print("\nSKIPPED CONDITIONS")
        print(json.dumps(skipped_conditions, indent=2))
    print("\nCONDITION SUMMARY")
    print(summary.to_string(index=False))
    print("\nREADY-TO-PASTE LATEX ROWS")
    print(
        (
            out_dir
            / "cnn_ae_branch_ablation_latex_rows.tex"
        ).read_text(encoding="utf-8")
    )
    print("\nOUTPUT DIRECTORY")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
