#!/usr/bin/env python3
"""
Phenotype-similarity sensitivity analysis for EPG-SD.

This script reconstructs the archived final cosine graph exactly and then changes
ONLY the phenotype similarity metric for the three alternative conditions.

Archived-pipeline alignment:
1. Uses the archived final phenotype column order:
   internalizing, attention, externalizing, p_factor.
2. Uses raw cosine similarities (no shift, clipping, or min-max scaling).
3. Builds the binary subject graph exactly as the archived final run:
   kneighbors_graph on rows of the fused similarity matrix, mode='connectivity',
   Euclidean row distance, include_self=False, followed by symmetrization.
4. Uses the archived final VGAE architecture and training setup:
   196 -> 64 -> 16, 500 epochs, posterior mean for clustering.
5. Uses the archived final KL normalization (divide by number of subjects) and
   total loss = reconstruction BCE + KL.
6. Uses UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
   followed by SpectralClustering(K=5, nearest-neighbor affinity, 12 neighbors).
7. Recomputes the six-variable clinical table with ANOVA, Kruskal-Wallis,
   eta-squared, omega-squared, and Benjamini-Hochberg FDR.
8. Audits the cosine graph against the archived final fused_similarity.npy,
   adjacency.npy, and node_features.npy before running the ablation.

The script produces a fresh four-row table. It does not hard-code the headline
metrics. The archived final metrics are read only for an explicit reproduction
comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
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
except Exception as exc:  # pragma: no cover
    raise ImportError("Install UMAP with: pip install umap-learn") from exc


# -----------------------------------------------------------------------------
# Fixed final-model settings
# -----------------------------------------------------------------------------
SEED = 42
GRAPH_K = 12
ALPHA = 0.6
UMAP_N = 30
FINAL_K = 5
EPOCHS = 500
LR = 1e-3
HIDDEN_DIM = 64
LATENT_DIM = 16

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


@dataclass(frozen=True)
class Paths:
    root: Path
    eeg_file: Path
    meta_file: Path
    final_graph_dir: Path
    final_umap_dir: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("TECHNICAL_FIX_SPECTRAL_UMAP"),
        help="TECHNICAL_FIX_SPECTRAL_UMAP directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/phenotype_similarity_sensitivity",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["cosine", "euclidean_rbf", "mahalanobis_rbf", "correlation"],
        default=["cosine", "euclidean_rbf", "mahalanobis_rbf", "correlation"],
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--retrain-cosine",
        action="store_true",
        help="Retrain the cosine condition instead of reusing the archived final "
        "EPG-SD control. By default the archived final cosine artifacts are used, "
        "so the control row is exactly the headline final model.",
    )
    parser.add_argument(
        "--strict-control",
        action="store_true",
        help="Exit nonzero unless the freshly retrained cosine condition matches "
        "the archived final clustering metrics within the stated tolerances.",
    )
    return parser.parse_args()


def make_paths(root: Path, out: Path | None) -> Paths:
    root = root.expanduser().resolve()
    final_graph_dir = (
        root
        / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
        / "graphK12_alpha0p6"
    )
    final_umap_dir = final_graph_dir / "UMAP30_K5"
    out_dir = (
        out.expanduser().resolve()
        if out is not None
        else root / "phenotype_similarity_sensitivity"
    )
    return Paths(
        root=root,
        eeg_file=root
        / "subject_embeddings_norm_fixed"
        / "subject_embedding_mean_std_median_norm_fixed.npy",
        meta_file=root
        / "subject_embeddings_norm_fixed"
        / "subject_metadata_norm_fixed.csv",
        final_graph_dir=final_graph_dir,
        final_umap_dir=final_umap_dir,
        out_dir=out_dir,
    )


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_cosine_similarity(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    return S


def euclidean_rbf_similarity(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    D = cdist(X, X, metric="euclidean")
    nonzero = D[D > 0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    S = np.exp(-(D**2) / (2.0 * sigma**2 + 1e-9))
    np.fill_diagonal(S, 1.0)
    return S


def mahalanobis_rbf_similarity(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    cov = np.cov(X, rowvar=False)
    cov = cov + 1e-4 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    D = cdist(X, X, metric="mahalanobis", VI=inv_cov)
    nonzero = D[D > 0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    S = np.exp(-(D**2) / (2.0 * sigma**2 + 1e-9))
    np.fill_diagonal(S, 1.0)
    return S


def correlation_similarity(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    Xn = Xc / np.maximum(norms, 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    return S


def build_binary_graph(fused_similarity: np.ndarray, k: int = GRAPH_K) -> np.ndarray:
    """Exact graph recipe recovered from the archived final artifacts."""
    directed = kneighbors_graph(
        fused_similarity,
        n_neighbors=k,
        mode="connectivity",
        metric="euclidean",
        include_self=False,
    ).toarray()
    A = np.maximum(directed, directed.T).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A_tilde = np.asarray(A, dtype=np.float32).copy()
    np.fill_diagonal(A_tilde, 1.0)
    degree = A_tilde.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(degree + 1e-9)
    return inv_sqrt[:, None] * A_tilde * inv_sqrt[None, :]


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        return self.linear(A_norm @ X)


class FinalVGAE(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.gcn1 = DenseGraphConv(in_dim, HIDDEN_DIM)
        self.gcn_mu = DenseGraphConv(HIDDEN_DIM, LATENT_DIM)
        self.gcn_logstd = DenseGraphConv(HIDDEN_DIM, LATENT_DIM)

    def encode(
        self, X: torch.Tensor, A_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = F.relu(self.gcn1(X, A_norm))
        mu = self.gcn_mu(H, A_norm)
        logstd = self.gcn_logstd(H, A_norm).clamp(-4.0, 4.0)
        return mu, logstd

    @staticmethod
    def reparameterize(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(logstd)

    def forward(
        self, X: torch.Tensor, A_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.encode(X, A_norm)
        Z = self.reparameterize(mu, logstd)
        logits = Z @ Z.T
        return logits, mu, logstd


def train_vgae(
    X_node: np.ndarray,
    adjacency: np.ndarray,
    out_dir: Path,
    device: torch.device,
) -> np.ndarray:
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_np = (adjacency > 0).astype(np.float32)
    A_norm_np = normalize_adjacency(adjacency).astype(np.float32)

    X = torch.tensor(X_node, dtype=torch.float32, device=device)
    A_norm = torch.tensor(A_norm_np, dtype=torch.float32, device=device)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)

    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = (negatives / (positives + 1e-9)).clamp(min=1.0, max=50.0)

    model = FinalVGAE(in_dim=X_node.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits, mu, logstd = model(X, A_norm)
        recon = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )

        # Archived-final normalization: average node-wise KL, then divide by N.
        raw_kl = -0.5 * torch.mean(
            torch.sum(
                1.0
                + 2.0 * logstd
                - mu.pow(2)
                - torch.exp(2.0 * logstd),
                dim=1,
            )
        )
        kl = raw_kl / X.shape[0]
        loss = recon + kl
        loss.backward()
        optimizer.step()

        rows.append(
            {
                "epoch": epoch,
                "recon_loss": float(recon.detach().cpu()),
                "kl_loss": float(kl.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
        )
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"    epoch={epoch:03d} "
                f"recon={recon.item():.6f} "
                f"kl={kl.item():.6f} total={loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X, A_norm)
        Z = mu.detach().cpu().numpy()

    pd.DataFrame(rows).to_csv(out_dir / "vgae_training_log.csv", index=False)
    np.save(out_dir / "latent_embeddings.npy", Z)
    pd.DataFrame(Z).to_csv(out_dir / "latent_embeddings.csv", index=False)
    return Z


def run_umap(Z: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=UMAP_N,
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED,
    )
    return reducer.fit_transform(Z)


def run_spectral(U: np.ndarray) -> np.ndarray:
    return SpectralClustering(
        n_clusters=FINAL_K,
        affinity="nearest_neighbors",
        n_neighbors=GRAPH_K,
        assign_labels="kmeans",
        random_state=SEED,
    ).fit_predict(U)


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


def one_variable_stats(values: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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

    anova_F, anova_p = stats.f_oneway(*groups)
    kruskal_H, kruskal_p = stats.kruskal(*groups)

    all_valid = np.concatenate(groups)
    grand_mean = float(np.mean(all_valid))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups)
    ss_within = sum(float(np.sum((g - float(np.mean(g))) ** 2)) for g in groups)
    ss_total = ss_between + ss_within
    df_between = len(groups) - 1
    df_within = len(all_valid) - len(groups)
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    omega_num = ss_between - df_between * ms_within
    omega_den = ss_total + ms_within
    omega_squared = max(0.0, omega_num / omega_den) if omega_den > 0 else 0.0

    return {
        "anova_F": float(anova_F),
        "anova_p": float(anova_p),
        "kruskal_H": float(kruskal_H),
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
    return table


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def audit_cosine_graph(
    paths: Paths,
    fused: np.ndarray,
    adjacency: np.ndarray,
    node_features: np.ndarray,
) -> Dict[str, object]:
    archived_fused = np.load(paths.final_graph_dir / "fused_similarity.npy")
    archived_adjacency = np.load(paths.final_graph_dir / "adjacency.npy")
    archived_nodes = np.load(paths.final_graph_dir / "node_features.npy")

    audit = {
        "fused_similarity_max_abs_error": max_abs_error(fused, archived_fused),
        "adjacency_exact_match": bool(np.array_equal(adjacency, archived_adjacency)),
        "adjacency_different_entries": int(np.sum(adjacency != archived_adjacency)),
        "node_features_max_abs_error": max_abs_error(node_features, archived_nodes),
    }
    audit["pass"] = bool(
        audit["fused_similarity_max_abs_error"] <= 1e-6
        and audit["adjacency_exact_match"]
        and audit["node_features_max_abs_error"] <= 1e-6
    )
    return audit


def summarize(
    metric_name: str,
    U: np.ndarray,
    labels: np.ndarray,
    clinical: pd.DataFrame,
    archived_labels: np.ndarray | None,
) -> Dict[str, object]:
    counts = np.bincount(labels, minlength=FINAL_K)
    row: Dict[str, object] = {
        "phenotype_similarity": metric_name,
        "alpha": ALPHA,
        "graph_k": GRAPH_K,
        "umap_n_neighbors": UMAP_N,
        "final_k": FINAL_K,
        "silhouette": float(silhouette_score(U, labels)),
        "davies_bouldin": float(davies_bouldin_score(U, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(U, labels)),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "cluster_size_ratio_max_min": float(counts.max() / max(counts.min(), 1)),
        "mean_eta_squared": float(clinical["eta_squared"].mean()),
        "fdr_sig_count": f"{int((clinical['anova_p_fdr_bh'] < 0.05).sum())}/6",
    }
    if archived_labels is not None:
        row["ARI_vs_archived_final"] = float(
            adjusted_rand_score(archived_labels, labels)
        )
        row["NMI_vs_archived_final"] = float(
            normalized_mutual_info_score(archived_labels, labels)
        )
    return row


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Sensitivity of EPG-SD to the phenotype similarity metric. The cosine row is the archived final EPG-SD control. For the alternative rows, the subject-level EEG representation, phenotype variables, fusion weight ($\alpha=0.6$), graph neighborhood size ($K_g=12$), VGAE architecture and training settings, UMAP configuration ($n_{\mathrm{neighbors}}=30$), subtype number ($K=5$), and random seed were held fixed; only the phenotype-similarity function was changed.}",
        r"\label{tab:phenotype_similarity_ablation}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Phenotype similarity & Sil. $\uparrow$ & DB $\downarrow$ & CH $\uparrow$ & Min. $\uparrow$ & Bal. $\downarrow$ & FDR \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        name = str(row["phenotype_similarity"]).replace("_", r"\_")
        lines.append(
            f"{name} & {row['silhouette']:.4f} & "
            f"{row['davies_bouldin']:.4f} & "
            f"{row['calinski_harabasz']:.2f} & "
            f"{int(row['min_cluster_size'])} & "
            f"{row['cluster_size_ratio_max_min']:.2f} & "
            f"{row['fdr_sig_count']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = make_paths(args.root, args.out)
    device = choose_device(args.device)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        paths.eeg_file,
        paths.meta_file,
        paths.final_graph_dir / "fused_similarity.npy",
        paths.final_graph_dir / "adjacency.npy",
        paths.final_graph_dir / "node_features.npy",
        paths.final_umap_dir / "main_quality.csv",
        paths.final_umap_dir / "subject_subtypes.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("=" * 88)
    print("CORRECTED TABLE 14 PHENOTYPE-SIMILARITY ABLATION")
    print("=" * 88)
    print("Root:", paths.root)
    print("Output:", paths.out_dir)
    print("Device:", device)

    X_eeg_raw = np.load(paths.eeg_file)
    meta = pd.read_csv(paths.meta_file)

    X_pheno_raw = (
        meta[PHENO_COLS_FINAL_ORDER]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(meta[PHENO_COLS_FINAL_ORDER].median(numeric_only=True))
        .to_numpy()
    )

    X_eeg = StandardScaler().fit_transform(np.nan_to_num(X_eeg_raw))
    X_pheno = StandardScaler().fit_transform(np.nan_to_num(X_pheno_raw))
    X_node = StandardScaler().fit_transform(
        np.concatenate([X_eeg, X_pheno], axis=1)
    ).astype(np.float32)

    A_eeg = raw_cosine_similarity(X_eeg)

    archived_label_table = pd.read_csv(paths.final_umap_dir / "subject_subtypes.csv")
    archived_labels = archived_label_table["subtype"].to_numpy()
    archived_quality = pd.read_csv(paths.final_umap_dir / "main_quality.csv").iloc[0]

    builders: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "cosine": raw_cosine_similarity,
        "euclidean_rbf": euclidean_rbf_similarity,
        "mahalanobis_rbf": mahalanobis_rbf_similarity,
        "correlation": correlation_similarity,
    }

    # Mandatory cosine graph audit before any model is trained.
    A_phe_cosine = raw_cosine_similarity(X_pheno)
    fused_cosine = ALPHA * A_eeg + (1.0 - ALPHA) * A_phe_cosine
    adjacency_cosine = build_binary_graph(fused_cosine, GRAPH_K)
    graph_audit = audit_cosine_graph(
        paths, fused_cosine, adjacency_cosine, X_node
    )
    (paths.out_dir / "cosine_graph_reconstruction_audit.json").write_text(
        json.dumps(graph_audit, indent=2)
    )
    print("\nCosine graph reconstruction audit:")
    print(json.dumps(graph_audit, indent=2))
    if not graph_audit["pass"]:
        raise RuntimeError(
            "Cosine graph reconstruction does not match the archived final graph. "
            "A controlled ablation cannot proceed."
        )

    rows = []
    for metric_name in args.metrics:
        print("\n" + "=" * 88)
        print("Metric:", metric_name)
        print("=" * 88)
        metric_dir = paths.out_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        # The cosine row is the already finalized EPG-SD control unless the user
        # explicitly asks for a fresh stochastic retraining. This reads the
        # archived artifacts; it does not hard-code any metric value.
        if metric_name == "cosine" and not args.retrain_cosine:
            A_phe = A_phe_cosine
            np.save(metric_dir / "A_phenotype.npy", A_phe)
            shutil.copy2(
                paths.final_graph_dir / "fused_similarity.npy",
                metric_dir / "fused_similarity.npy",
            )
            shutil.copy2(
                paths.final_graph_dir / "adjacency.npy",
                metric_dir / "adjacency.npy",
            )
            shutil.copy2(
                paths.final_graph_dir / "node_features.npy",
                metric_dir / "node_features.npy",
            )
            if (metric_dir / "vgae").exists():
                shutil.rmtree(metric_dir / "vgae")
            shutil.copytree(paths.final_graph_dir / "vgae", metric_dir / "vgae")

            for filename in [
                "umap_coordinates.csv",
                "subject_subtypes.csv",
                "cluster_sizes.csv",
                "clinical_stats.csv",
            ]:
                shutil.copy2(paths.final_umap_dir / filename, metric_dir / filename)

            U = pd.read_csv(metric_dir / "umap_coordinates.csv")[["UMAP1", "UMAP2"]].to_numpy()
            label_table = pd.read_csv(metric_dir / "subject_subtypes.csv")
            labels = label_table["subtype"].to_numpy()
            clinical = pd.read_csv(metric_dir / "clinical_stats.csv")

            row = {
                "phenotype_similarity": "cosine",
                "alpha": ALPHA,
                "graph_k": GRAPH_K,
                "umap_n_neighbors": UMAP_N,
                "final_k": FINAL_K,
                "silhouette": float(archived_quality["silhouette"]),
                "davies_bouldin": float(archived_quality["davies_bouldin"]),
                "calinski_harabasz": float(archived_quality["calinski_harabasz"]),
                "min_cluster_size": int(archived_quality["min_cluster_size"]),
                "max_cluster_size": int(archived_quality["max_cluster_size"]),
                "cluster_size_ratio_max_min": float(
                    archived_quality["cluster_size_ratio_max_min"]
                ),
                "mean_eta_squared": float(clinical["eta_squared"].mean()),
                "fdr_sig_count": f"{int((clinical['anova_p_fdr_bh'] < 0.05).sum())}/6",
                "ARI_vs_archived_final": 1.0,
                "NMI_vs_archived_final": 1.0,
                "control_source": "archived_final_EPGSD",
            }
            pd.DataFrame([row]).to_csv(metric_dir / "main_quality.csv", index=False)
            rows.append(row)
            print(pd.Series(row).to_string())
            continue

        A_phe = builders[metric_name](X_pheno)
        A_fused = ALPHA * A_eeg + (1.0 - ALPHA) * A_phe
        adjacency = build_binary_graph(A_fused, GRAPH_K)

        np.save(metric_dir / "A_phenotype.npy", A_phe)
        np.save(metric_dir / "fused_similarity.npy", A_fused)
        np.save(metric_dir / "adjacency.npy", adjacency)
        np.save(metric_dir / "node_features.npy", X_node)

        Z = train_vgae(X_node, adjacency, metric_dir / "vgae", device)
        U = run_umap(Z)
        labels = run_spectral(U)

        pd.DataFrame(U, columns=["UMAP1", "UMAP2"]).to_csv(
            metric_dir / "umap_coordinates.csv", index=False
        )
        pd.DataFrame(
            {"subject_index": np.arange(len(labels)), "subtype": labels}
        ).to_csv(metric_dir / "subject_subtypes.csv", index=False)

        counts = np.bincount(labels, minlength=FINAL_K)
        pd.DataFrame(
            {"subtype": np.arange(FINAL_K), "n_subjects": counts}
        ).to_csv(metric_dir / "cluster_sizes.csv", index=False)

        clinical = clinical_stats(meta, labels)
        clinical.to_csv(metric_dir / "clinical_stats.csv", index=False)

        row = summarize(metric_name, U, labels, clinical, archived_labels)
        row["control_source"] = "fresh_fixed_seed_rerun"
        pd.DataFrame([row]).to_csv(metric_dir / "main_quality.csv", index=False)
        rows.append(row)

        print(pd.Series(row).to_string())

    summary = pd.DataFrame(rows)
    summary.to_csv(
        paths.out_dir / "phenotype_similarity_metric_ablation_summary.csv",
        index=False,
    )

    # Fresh cosine-vs-archived control report.
    cosine_rows = summary[summary["phenotype_similarity"] == "cosine"]
    if args.retrain_cosine and not cosine_rows.empty:
        fresh = cosine_rows.iloc[0]
        comparison = pd.DataFrame(
            [
                {
                    "source": "archived_final",
                    "silhouette": archived_quality["silhouette"],
                    "davies_bouldin": archived_quality["davies_bouldin"],
                    "calinski_harabasz": archived_quality["calinski_harabasz"],
                    "min_cluster_size": archived_quality["min_cluster_size"],
                    "cluster_size_ratio_max_min": archived_quality[
                        "cluster_size_ratio_max_min"
                    ],
                },
                {
                    "source": "fresh_corrected_cosine",
                    "silhouette": fresh["silhouette"],
                    "davies_bouldin": fresh["davies_bouldin"],
                    "calinski_harabasz": fresh["calinski_harabasz"],
                    "min_cluster_size": fresh["min_cluster_size"],
                    "cluster_size_ratio_max_min": fresh[
                        "cluster_size_ratio_max_min"
                    ],
                },
            ]
        )
        comparison.to_csv(paths.out_dir / "cosine_control_comparison.csv", index=False)

        metric_match = bool(
            abs(float(fresh["silhouette"]) - float(archived_quality["silhouette"]))
            <= 1e-4
            and abs(
                float(fresh["davies_bouldin"])
                - float(archived_quality["davies_bouldin"])
            )
            <= 1e-4
            and abs(
                float(fresh["calinski_harabasz"])
                - float(archived_quality["calinski_harabasz"])
            )
            <= 0.1
            and int(fresh["min_cluster_size"])
            == int(archived_quality["min_cluster_size"])
            and abs(
                float(fresh["cluster_size_ratio_max_min"])
                - float(archived_quality["cluster_size_ratio_max_min"])
            )
            <= 1e-4
        )
        control = {
            "graph_reconstruction_pass": graph_audit["pass"],
            "fresh_cosine_metrics_match_archived_final": metric_match,
            "fresh_cosine_ARI_vs_archived_final": float(
                fresh.get("ARI_vs_archived_final", np.nan)
            ),
            "fresh_cosine_NMI_vs_archived_final": float(
                fresh.get("NMI_vs_archived_final", np.nan)
            ),
            "note": (
                "Exact graph reconstruction is mandatory. Fresh downstream metrics "
                "may differ across CUDA/PyTorch/UMAP versions unless the original "
                "software environment is reproduced."
            ),
        }
        (paths.out_dir / "cosine_control_status.json").write_text(
            json.dumps(control, indent=2)
        )
        print("\nCosine control status:")
        print(json.dumps(control, indent=2))
        if args.strict_control and not metric_match:
            print(
                "STRICT CONTROL FAILED: no final LaTeX table will be produced.",
                file=sys.stderr,
            )
            return 2

    write_latex(
        summary,
        paths.out_dir / "phenotype_similarity_metric_ablation_latex.txt",
    )

    run_manifest = {
        "seed": SEED,
        "graph_k": GRAPH_K,
        "alpha": ALPHA,
        "umap_n_neighbors": UMAP_N,
        "final_k": FINAL_K,
        "vgae_epochs": EPOCHS,
        "vgae_hidden_dim": HIDDEN_DIM,
        "vgae_latent_dim": LATENT_DIM,
        "phenotype_column_order": PHENO_COLS_FINAL_ORDER,
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (paths.out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print("\nSaved to:", paths.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
