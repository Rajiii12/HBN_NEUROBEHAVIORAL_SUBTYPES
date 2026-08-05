#!/usr/bin/env python3
"""
27_nested_alpha_k_subject_resampling.py

Nested subject-level resampling for validation of the
EEG--phenotype fusion weight alpha and subtype count K.

Design
------
1. Five outer subject folds are created once.
2. For each outer fold, only the outer-development subjects are used for
   model selection.
3. Within the outer-development set, repeated 80% subject subsamples are
   used to evaluate each candidate (alpha, K) by internal validity,
   graph-Laplacian eigengap, cluster balance/size, and resampling stability.
4. Age, EHQ/handedness, phenotype-association p-values, and all clinical
   interpretation variables are excluded from selection.
5. The selected alpha and K are locked and evaluated on the untouched outer
   test subjects as a held-out structural replication.

Important scope
---------------
The script starts from saved 192-dimensional subject-level EEG embeddings.
It therefore nests the downstream standardization, phenotype preparation,
graph construction, VGAE, UMAP, and spectral-clustering stages. It does not
retrain the upstream CNN-autoencoder streams inside every fold.

The default graph builder reproduces the archived implementation, in which
kneighbors_graph is applied to rows of the fused similarity matrix.

Typical final run
-----------------
python -u 27_nested_alpha_k_subject_resampling.py \
  --project-root /path/to/HBN_NEUROBEHAVIORAL_SUBTYPES \
  --output-root /path/to/private_nested_results \
  --outer-folds 5 \
  --inner-repeats 10 \
  --inner-fraction 0.80 \
  --alpha-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --k-grid 2,3,4,5,6,7,8,9,10 \
  --vgae-epochs 500

Parallel outer-fold execution is supported with --outer-fold-index.
Run --finalize-only after all fold jobs have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BEHAVIOR_COLS = ["internalizing", "attention", "externalizing", "p_factor"]
EXCLUDED_SELECTION_VARS = [
    "age",
    "ehq_total",
    "handedness",
    "P-factor/attention/internalizing/externalizing association p-values",
    "FDR counts",
    "effect sizes used for post-clustering interpretation",
]


@dataclass(frozen=True)
class Config:
    seed: int
    outer_folds: int
    inner_repeats: int
    inner_fraction: float
    alpha_grid: Tuple[float, ...]
    k_grid: Tuple[int, ...]
    graph_k: int
    umap_neighbors: int
    umap_min_dist: float
    umap_metric: str
    vgae_hidden_dim: int
    vgae_latent_dim: int
    vgae_epochs: int
    learning_rate: float
    dropout: float
    min_cluster_fraction: float
    outer_eval_seeds: int
    graph_builder: str
    device: str


class FoldPreprocessor:
    """Development-fitted imputation and scaling for EEG and phenotypes."""

    def __init__(self) -> None:
        self.eeg_scaler = StandardScaler()
        self.pheno_scaler = StandardScaler()
        self.pheno_medians: Optional[np.ndarray] = None

    def fit(self, eeg: np.ndarray, pheno: np.ndarray) -> "FoldPreprocessor":
        self.eeg_scaler.fit(eeg)
        med = np.nanmedian(pheno, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self.pheno_medians = med.astype(np.float64)
        self.pheno_scaler.fit(self._impute(pheno))
        return self

    def _impute(self, pheno: np.ndarray) -> np.ndarray:
        if self.pheno_medians is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        out = np.asarray(pheno, dtype=np.float64).copy()
        missing = ~np.isfinite(out)
        if missing.any():
            out[missing] = np.take(self.pheno_medians, np.where(missing)[1])
        return out

    def transform(self, eeg: np.ndarray, pheno: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eeg_t = self.eeg_scaler.transform(eeg).astype(np.float32)
        pheno_t = self.pheno_scaler.transform(self._impute(pheno)).astype(np.float32)
        return eeg_t, pheno_t

    def fit_transform(self, eeg: np.ndarray, pheno: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(eeg, pheno).transform(eeg, pheno)


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.lin(adj_norm @ x)


class DenseVGAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gc1 = DenseGCNLayer(input_dim, hidden_dim)
        self.gc_mu = DenseGCNLayer(hidden_dim, latent_dim)
        self.gc_logstd = DenseGCNLayer(hidden_dim, latent_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.gc1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.gc_mu(h, adj_norm)
        logstd = torch.clamp(self.gc_logstd(h, adj_norm), min=-10.0, max=10.0)
        return mu, logstd

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(logstd)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(
        self, x: torch.Tensor, adj_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.encode(x, adj_norm)
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd, z


def parse_float_grid(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("Grid cannot be empty.")
    if any(x < 0.0 or x > 1.0 for x in values):
        raise argparse.ArgumentTypeError("All alpha values must be in [0, 1].")
    return values


def parse_int_grid(text: str) -> Tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(x < 2 for x in values):
        raise argparse.ArgumentTypeError("K values must be integers >= 2.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def find_first_existing(candidates: Iterable[Path], label: str) -> Path:
    checked: List[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {label}. Checked:\n" + "\n".join(checked))


def resolve_inputs(
    project_root: Path,
    eeg_file_arg: Optional[str],
    metadata_file_arg: Optional[str],
) -> Tuple[Path, Path]:
    eeg_candidates: List[Path] = []
    meta_candidates: List[Path] = []
    if eeg_file_arg:
        eeg_candidates.append(Path(eeg_file_arg))
    if metadata_file_arg:
        meta_candidates.append(Path(metadata_file_arg))

    eeg_candidates.extend(
        [
            project_root / "TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_norm_fixed/subject_embedding_mean_std_median_norm_fixed.npy",
            project_root / "TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_fixed/subject_embedding_mean_std_fixed.npy",
            project_root / "subject_embeddings/subject_embedding.npy",
        ]
    )
    meta_candidates.extend(
        [
            project_root / "TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_norm_fixed/subject_metadata_norm_fixed.csv",
            project_root / "TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_fixed/subject_metadata_fixed.csv",
            project_root / "subject_embeddings/subject_metadata.csv",
        ]
    )
    return (
        find_first_existing(eeg_candidates, "subject-level EEG embedding"),
        find_first_existing(meta_candidates, "subject metadata"),
    )


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a = adjacency.astype(np.float32, copy=True)
    a += np.eye(a.shape[0], dtype=np.float32)
    degree = a.sum(axis=1)
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    return (inv_sqrt[:, None] * a * inv_sqrt[None, :]).astype(np.float32)


def build_adjacency(similarity: np.ndarray, k: int, mode: str) -> np.ndarray:
    n = similarity.shape[0]
    k_eff = max(1, min(int(k), n - 1))
    if mode == "archived_similarity_rows":
        adjacency = kneighbors_graph(
            similarity,
            n_neighbors=k_eff,
            mode="connectivity",
            include_self=False,
        ).toarray()
    elif mode == "direct_topk_similarity":
        sim = similarity.copy()
        np.fill_diagonal(sim, -np.inf)
        idx = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
        adjacency = np.zeros_like(similarity, dtype=np.float32)
        rows = np.arange(n)[:, None]
        adjacency[rows, idx] = 1.0
    else:
        raise ValueError(f"Unknown graph builder: {mode}")
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def train_vgae(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    config: Config,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    device = torch.device(config.device)
    x = torch.as_tensor(node_features, dtype=torch.float32, device=device)
    adj_norm = torch.as_tensor(normalize_adjacency(adjacency), dtype=torch.float32, device=device)

    target_np = adjacency.astype(np.float32, copy=True)
    target_np += np.eye(target_np.shape[0], dtype=np.float32)
    target_np[target_np > 0] = 1.0
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)

    n_pos = float(target_np.sum())
    n_total = float(target_np.size)
    pos_weight = torch.tensor((n_total - n_pos) / max(n_pos, 1.0), dtype=torch.float32, device=device)

    model = DenseVGAE(
        input_dim=node_features.shape[1],
        hidden_dim=config.vgae_hidden_dim,
        latent_dim=config.vgae_latent_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    final = {"recon_loss": np.nan, "kl_loss": np.nan, "total_loss": np.nan}
    for _epoch in range(1, config.vgae_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logstd, _z = model(x, adj_norm)
        weights = target * pos_weight + (1.0 - target)
        recon_loss = F.binary_cross_entropy(reconstruction, target, weight=weights, reduction="mean")
        kl_loss = -0.5 / x.shape[0] * torch.mean(
            torch.sum(1.0 + 2.0 * logstd - mu.pow(2) - torch.exp(2.0 * logstd), dim=1)
        )
        loss = recon_loss + kl_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite VGAE loss encountered.")
        loss.backward()
        optimizer.step()
        final = {
            "recon_loss": float(recon_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }

    model.eval()
    with torch.no_grad():
        mu, _logstd = model.encode(x, adj_norm)
    latent = mu.detach().cpu().numpy().astype(np.float32)
    del model, optimizer, x, adj_norm, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latent, final


def run_umap(latent: np.ndarray, config: Config, seed: int) -> np.ndarray:
    n_neighbors = max(2, min(config.umap_neighbors, len(latent) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(latent).astype(np.float32)


def run_spectral(embedding_2d: np.ndarray, k: int, graph_k: int, seed: int) -> np.ndarray:
    if k >= len(embedding_2d):
        raise ValueError(f"K={k} is not valid for n={len(embedding_2d)}.")
    n_neighbors = max(1, min(graph_k, len(embedding_2d) - 1))
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=seed,
    )
    return model.fit_predict(embedding_2d).astype(np.int32)


def spectral_eigengap(embedding_2d: np.ndarray, k: int, graph_k: int) -> float:
    if k >= len(embedding_2d):
        return float("nan")
    n_neighbors = max(1, min(graph_k, len(embedding_2d) - 1))
    affinity = kneighbors_graph(
        embedding_2d,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    ).toarray()
    affinity = np.maximum(affinity, affinity.T)
    lap = csgraph_laplacian(affinity, normed=True)
    eigenvalues = np.linalg.eigvalsh(lap)
    eigenvalues = np.sort(np.real(eigenvalues))
    return float(eigenvalues[k] - eigenvalues[k - 1]) if len(eigenvalues) > k else float("nan")


def safe_cluster_metrics(embedding_2d: np.ndarray, labels: np.ndarray, k: int, graph_k: int) -> Dict[str, float]:
    unique, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    valid = len(unique) >= 2 and len(unique) == k and counts.min() >= 2
    if valid:
        try:
            silhouette = float(silhouette_score(embedding_2d, labels))
        except Exception:
            silhouette = float("nan")
        try:
            db = float(davies_bouldin_score(embedding_2d, labels))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(embedding_2d, labels))
        except Exception:
            ch = float("nan")
    else:
        silhouette = db = ch = float("nan")
    min_size = int(counts.min()) if len(counts) else 0
    max_size = int(counts.max()) if len(counts) else 0
    return {
        "silhouette": silhouette,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "eigengap": spectral_eigengap(embedding_2d, k, graph_k),
        "n_clusters_found": int(len(unique)),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "min_cluster_fraction": float(min_size / n) if n else float("nan"),
        "balance_ratio": float(max_size / min_size) if min_size > 0 else float("inf"),
    }


def build_fused_inputs(
    eeg_scaled: np.ndarray,
    pheno_scaled: np.ndarray,
    alpha: float,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    eeg_similarity = cosine_similarity(eeg_scaled)
    pheno_similarity = cosine_similarity(pheno_scaled)
    fused_similarity = alpha * eeg_similarity + (1.0 - alpha) * pheno_similarity
    adjacency = build_adjacency(fused_similarity, config.graph_k, config.graph_builder)
    node_features = np.concatenate([eeg_scaled, pheno_scaled], axis=1).astype(np.float32)
    return node_features, adjacency


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:.3f}".replace(".", "p")


def run_alpha_pipeline(
    eeg_scaled: np.ndarray,
    pheno_scaled: np.ndarray,
    alpha: float,
    config: Config,
    seed: int,
    cache_dir: Path,
    save_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = cache_dir / f"{alpha_tag(alpha)}_metrics.csv"
    labels_file = cache_dir / f"{alpha_tag(alpha)}_labels.npz"
    loss_file = cache_dir / f"{alpha_tag(alpha)}_vgae_loss.json"

    if metrics_file.exists() and labels_file.exists() and loss_file.exists():
        metrics = pd.read_csv(metrics_file)
        loaded = np.load(labels_file)
        labels = {int(key[1:]): loaded[key] for key in loaded.files if key.startswith("k")}
        with loss_file.open("r", encoding="utf-8") as f:
            losses = json.load(f)
        return metrics, labels, losses

    node_features, adjacency = build_fused_inputs(eeg_scaled, pheno_scaled, alpha, config)
    latent, losses = train_vgae(node_features, adjacency, config, seed)
    embedding_2d = run_umap(latent, config, seed)

    rows: List[Dict[str, float]] = []
    labels_by_k: Dict[int, np.ndarray] = {}
    for k in config.k_grid:
        if k >= len(embedding_2d):
            continue
        labels = run_spectral(embedding_2d, k, config.graph_k, seed)
        labels_by_k[k] = labels
        row: Dict[str, float] = {
            "alpha": float(alpha),
            "K": int(k),
            "n_subjects": int(len(labels)),
            **safe_cluster_metrics(embedding_2d, labels, k, config.graph_k),
        }
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_file, index=False)
    if save_labels:
        np.savez_compressed(labels_file, **{f"k{k}": v for k, v in labels_by_k.items()})
    else:
        np.savez_compressed(labels_file, **{f"k{k}": v for k, v in labels_by_k.items()})
    json_dump(losses, loss_file)
    return metrics, labels_by_k, losses


def pairwise_label_stability(labels_list: Sequence[np.ndarray]) -> Tuple[float, float]:
    if len(labels_list) < 2:
        return float("nan"), float("nan")
    ari: List[float] = []
    nmi: List[float] = []
    for a, b in combinations(labels_list, 2):
        ari.append(adjusted_rand_score(a, b))
        nmi.append(normalized_mutual_info_score(a, b))
    return float(np.mean(ari)), float(np.mean(nmi))


def aggregate_candidate_summary(
    full_metrics: pd.DataFrame,
    inner_rows: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    candidates = full_metrics[["alpha", "K"]].drop_duplicates().sort_values(["alpha", "K"])
    rows: List[Dict[str, float]] = []
    for _, candidate in candidates.iterrows():
        alpha = float(candidate["alpha"])
        k = int(candidate["K"])
        full = full_metrics[np.isclose(full_metrics["alpha"], alpha) & (full_metrics["K"] == k)].iloc[0]
        inner = inner_rows[np.isclose(inner_rows["alpha"], alpha) & (inner_rows["K"] == k)]
        combined_sil = np.concatenate([[full["silhouette"]], inner["silhouette"].to_numpy(dtype=float)])
        combined_db = np.concatenate([[full["davies_bouldin"]], inner["davies_bouldin"].to_numpy(dtype=float)])
        combined_ch = np.concatenate([[full["calinski_harabasz"]], inner["calinski_harabasz"].to_numpy(dtype=float)])
        combined_gap = np.concatenate([[full["eigengap"]], inner["eigengap"].to_numpy(dtype=float)])
        combined_min = np.concatenate([[full["min_cluster_fraction"]], inner["min_cluster_fraction"].to_numpy(dtype=float)])
        combined_balance = np.concatenate([[full["balance_ratio"]], inner["balance_ratio"].to_numpy(dtype=float)])
        all_found = bool(full["n_clusters_found"] == k) and bool((inner["n_clusters_found"] == k).all())
        p10_min = float(np.nanpercentile(combined_min, 10))
        rows.append(
            {
                "alpha": alpha,
                "K": k,
                "n_inner_runs": int(len(inner)),
                "silhouette_mean": float(np.nanmean(combined_sil)),
                "silhouette_sd": float(np.nanstd(combined_sil, ddof=1)) if len(combined_sil) > 1 else 0.0,
                "davies_bouldin_mean": float(np.nanmean(combined_db)),
                "davies_bouldin_sd": float(np.nanstd(combined_db, ddof=1)) if len(combined_db) > 1 else 0.0,
                "calinski_harabasz_mean": float(np.nanmean(combined_ch)),
                "eigengap_mean": float(np.nanmean(combined_gap)),
                "resample_ari_mean": float(np.nanmean(inner["ARI_vs_full"])),
                "resample_nmi_mean": float(np.nanmean(inner["NMI_vs_full"])),
                "min_cluster_fraction_p10": p10_min,
                "balance_ratio_median": float(np.nanmedian(combined_balance)),
                "full_silhouette": float(full["silhouette"]),
                "full_davies_bouldin": float(full["davies_bouldin"]),
                "full_calinski_harabasz": float(full["calinski_harabasz"]),
                "full_eigengap": float(full["eigengap"]),
                "full_min_cluster_size": int(full["min_cluster_size"]),
                "full_balance_ratio": float(full["balance_ratio"]),
                "all_runs_found_K_clusters": all_found,
                "passes_min_cluster_fraction": bool(p10_min >= config.min_cluster_fraction),
            }
        )

    summary = pd.DataFrame(rows)
    summary["feasible"] = summary["all_runs_found_K_clusters"] & summary["passes_min_cluster_fraction"]

    rank_specs = {
        "silhouette_mean": False,
        "davies_bouldin_mean": True,
        "eigengap_mean": False,
        "resample_ari_mean": False,
        "resample_nmi_mean": False,
        "min_cluster_fraction_p10": False,
        "balance_ratio_median": True,
    }
    rank_cols: List[str] = []
    for metric, ascending in rank_specs.items():
        col = f"rank_{metric}"
        summary[col] = summary[metric].rank(method="average", ascending=ascending, na_option="bottom")
        rank_cols.append(col)
    summary["mean_rank"] = summary[rank_cols].mean(axis=1)

    selection_pool = summary[summary["feasible"]].copy()
    if selection_pool.empty:
        selection_pool = summary.copy()
    selected_index = selection_pool.sort_values(
        ["mean_rank", "resample_ari_mean", "silhouette_mean", "K"],
        ascending=[True, False, False, True],
    ).index[0]
    summary["selected"] = False
    summary.loc[selected_index, "selected"] = True
    return summary.sort_values(["selected", "mean_rank"], ascending=[False, True]).reset_index(drop=True)


def evaluate_selected_on_outer_test(
    eeg_scaled: np.ndarray,
    pheno_scaled: np.ndarray,
    alpha: float,
    k: int,
    config: Config,
    seed: int,
    fold_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    eval_dir = fold_dir / "outer_test_selected"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "outer_test_seed_metrics.csv"
    labels_path = eval_dir / "outer_test_labels_private.npz"
    if metrics_path.exists() and labels_path.exists():
        metrics = pd.read_csv(metrics_path)
        labels_npz = np.load(labels_path)
        primary = labels_npz["seed0"]
        return metrics, primary

    node_features, adjacency = build_fused_inputs(eeg_scaled, pheno_scaled, alpha, config)
    latent, losses = train_vgae(node_features, adjacency, config, seed)
    json_dump(losses, eval_dir / "outer_test_vgae_loss.json")

    rows: List[Dict[str, float]] = []
    labels_list: List[np.ndarray] = []
    for i in range(config.outer_eval_seeds):
        eval_seed = seed + i
        embedding_2d = run_umap(latent, config, eval_seed)
        labels = run_spectral(embedding_2d, k, config.graph_k, eval_seed)
        labels_list.append(labels)
        rows.append(
            {
                "eval_seed": eval_seed,
                "alpha": alpha,
                "K": k,
                "n_subjects": len(labels),
                **safe_cluster_metrics(embedding_2d, labels, k, config.graph_k),
            }
        )
    ari, nmi = pairwise_label_stability(labels_list)
    metrics = pd.DataFrame(rows)
    metrics["seed_pairwise_ARI"] = ari
    metrics["seed_pairwise_NMI"] = nmi
    metrics.to_csv(metrics_path, index=False)
    np.savez_compressed(labels_path, **{f"seed{i}": labels for i, labels in enumerate(labels_list)})
    return metrics, labels_list[0]


def run_outer_fold(
    outer_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    eeg_raw: np.ndarray,
    pheno_raw: np.ndarray,
    subject_ids: np.ndarray,
    output_root: Path,
    config: Config,
) -> None:
    fold_dir = output_root / f"outer_fold_{outer_index + 1:02d}"
    done_marker = fold_dir / "FOLD_COMPLETE.json"
    if done_marker.exists():
        print(f"[skip] Outer fold {outer_index + 1} is already complete: {fold_dir}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== OUTER FOLD {outer_index + 1}/{config.outer_folds} ===", flush=True)
    print(f"development={len(train_idx)} held_out={len(test_idx)}", flush=True)

    private_assignments = pd.DataFrame(
        {
            "row_index": np.concatenate([train_idx, test_idx]),
            "subject": np.concatenate([subject_ids[train_idx], subject_ids[test_idx]]),
            "outer_role": ["development"] * len(train_idx) + ["held_out"] * len(test_idx),
        }
    )
    private_assignments.to_csv(fold_dir / "outer_fold_assignments_PRIVATE.csv", index=False)

    dev_preprocessor = FoldPreprocessor()
    eeg_dev, pheno_dev = dev_preprocessor.fit_transform(eeg_raw[train_idx], pheno_raw[train_idx])
    eeg_test, pheno_test = dev_preprocessor.transform(eeg_raw[test_idx], pheno_raw[test_idx])

    full_metrics_parts: List[pd.DataFrame] = []
    full_labels_by_alpha: Dict[float, Dict[int, np.ndarray]] = {}
    full_cache = fold_dir / "private_cache/full_development"
    full_seed = config.seed + outer_index * 10000
    for alpha in config.alpha_grid:
        print(f"[fold {outer_index + 1}] full development alpha={alpha:.3f}", flush=True)
        metrics, labels, _losses = run_alpha_pipeline(
            eeg_dev,
            pheno_dev,
            alpha,
            config,
            full_seed,
            full_cache,
        )
        full_metrics_parts.append(metrics)
        full_labels_by_alpha[alpha] = labels
    full_metrics = pd.concat(full_metrics_parts, ignore_index=True)
    full_metrics.to_csv(fold_dir / "full_development_candidate_metrics.csv", index=False)

    inner_rows: List[pd.DataFrame] = []
    for repeat in range(config.inner_repeats):
        repeat_seed = config.seed + outer_index * 10000 + 100 + repeat
        rng = np.random.default_rng(repeat_seed)
        n_sub = max(20, int(round(config.inner_fraction * len(train_idx))))
        n_sub = min(n_sub, len(train_idx))
        relative_idx = np.sort(rng.choice(len(train_idx), size=n_sub, replace=False))
        absolute_idx = train_idx[relative_idx]

        inner_preprocessor = FoldPreprocessor()
        eeg_inner, pheno_inner = inner_preprocessor.fit_transform(eeg_raw[absolute_idx], pheno_raw[absolute_idx])
        repeat_cache = fold_dir / f"private_cache/inner_repeat_{repeat + 1:02d}"
        repeat_parts: List[pd.DataFrame] = []
        for alpha in config.alpha_grid:
            print(
                f"[fold {outer_index + 1}] inner {repeat + 1}/{config.inner_repeats} alpha={alpha:.3f}",
                flush=True,
            )
            metrics, labels_by_k, _losses = run_alpha_pipeline(
                eeg_inner,
                pheno_inner,
                alpha,
                config,
                repeat_seed,
                repeat_cache,
            )
            metrics = metrics.copy()
            metrics["inner_repeat"] = repeat + 1
            metrics["inner_n_subjects"] = n_sub
            ari_values: List[float] = []
            nmi_values: List[float] = []
            for _, row in metrics.iterrows():
                k = int(row["K"])
                full_labels = full_labels_by_alpha[alpha][k][relative_idx]
                inner_labels = labels_by_k[k]
                ari_values.append(float(adjusted_rand_score(full_labels, inner_labels)))
                nmi_values.append(float(normalized_mutual_info_score(full_labels, inner_labels)))
            metrics["ARI_vs_full"] = ari_values
            metrics["NMI_vs_full"] = nmi_values
            repeat_parts.append(metrics)
        repeat_df = pd.concat(repeat_parts, ignore_index=True)
        repeat_df.to_csv(fold_dir / f"inner_repeat_{repeat + 1:02d}_candidate_metrics.csv", index=False)
        inner_rows.append(repeat_df)

    inner_all = pd.concat(inner_rows, ignore_index=True)
    inner_all.to_csv(fold_dir / "inner_all_candidate_metrics.csv", index=False)
    candidate_summary = aggregate_candidate_summary(full_metrics, inner_all, config)
    candidate_summary.to_csv(fold_dir / "candidate_rank_summary.csv", index=False)
    selected = candidate_summary[candidate_summary["selected"]].iloc[0]
    selected_alpha = float(selected["alpha"])
    selected_k = int(selected["K"])
    json_dump(
        {
            "outer_fold": outer_index + 1,
            "selected_alpha": selected_alpha,
            "selected_K": selected_k,
            "selection_mean_rank": float(selected["mean_rank"]),
            "selection_resample_ARI": float(selected["resample_ari_mean"]),
            "selection_resample_NMI": float(selected["resample_nmi_mean"]),
            "selection_used_clinical_variables": False,
            "excluded_selection_variables": EXCLUDED_SELECTION_VARS,
        },
        fold_dir / "selected_configuration.json",
    )
    print(
        f"[fold {outer_index + 1}] selected alpha={selected_alpha:.3f}, K={selected_k}",
        flush=True,
    )

    outer_metrics, outer_primary_labels = evaluate_selected_on_outer_test(
        eeg_test,
        pheno_test,
        selected_alpha,
        selected_k,
        config,
        config.seed + outer_index * 10000 + 5000,
        fold_dir,
    )
    outer_metrics.to_csv(fold_dir / "outer_test_selected_metrics.csv", index=False)
    pd.DataFrame(
        {
            "row_index": test_idx,
            "subject": subject_ids[test_idx],
            "outer_test_cluster": outer_primary_labels,
        }
    ).to_csv(fold_dir / "outer_test_assignments_PRIVATE.csv", index=False)

    summary_row = {
        "outer_fold": outer_index + 1,
        "n_development": len(train_idx),
        "n_held_out": len(test_idx),
        "selected_alpha": selected_alpha,
        "selected_K": selected_k,
        "selection_mean_rank": float(selected["mean_rank"]),
        "inner_resample_ARI": float(selected["resample_ari_mean"]),
        "inner_resample_NMI": float(selected["resample_nmi_mean"]),
        "outer_silhouette_mean": float(outer_metrics["silhouette"].mean()),
        "outer_silhouette_sd": float(outer_metrics["silhouette"].std(ddof=1)) if len(outer_metrics) > 1 else 0.0,
        "outer_DB_mean": float(outer_metrics["davies_bouldin"].mean()),
        "outer_DB_sd": float(outer_metrics["davies_bouldin"].std(ddof=1)) if len(outer_metrics) > 1 else 0.0,
        "outer_CH_mean": float(outer_metrics["calinski_harabasz"].mean()),
        "outer_eigengap_mean": float(outer_metrics["eigengap"].mean()),
        "outer_min_cluster_size_median": float(outer_metrics["min_cluster_size"].median()),
        "outer_balance_ratio_mean": float(outer_metrics["balance_ratio"].mean()),
        "outer_seed_pairwise_ARI": float(outer_metrics["seed_pairwise_ARI"].iloc[0]),
        "outer_seed_pairwise_NMI": float(outer_metrics["seed_pairwise_NMI"].iloc[0]),
    }
    pd.DataFrame([summary_row]).to_csv(fold_dir / "FOLD_AGGREGATE_SUMMARY.csv", index=False)
    json_dump(summary_row, done_marker)
    print(f"[complete] Outer fold {outer_index + 1}: {fold_dir}", flush=True)


def finalize_results(output_root: Path, config: Config) -> None:
    fold_files = sorted(output_root.glob("outer_fold_*/FOLD_AGGREGATE_SUMMARY.csv"))
    if len(fold_files) != config.outer_folds:
        raise RuntimeError(
            f"Expected {config.outer_folds} completed folds, found {len(fold_files)}. "
            "Complete all folds before finalizing."
        )
    folds = pd.concat([pd.read_csv(path) for path in fold_files], ignore_index=True)
    folds.to_csv(output_root / "NESTED_OUTER_FOLD_RESULTS.csv", index=False)

    selection_frequency_alpha = (
        folds["selected_alpha"].value_counts().sort_index().rename_axis("alpha").reset_index(name="n_outer_folds")
    )
    selection_frequency_k = (
        folds["selected_K"].value_counts().sort_index().rename_axis("K").reset_index(name="n_outer_folds")
    )
    joint = (
        folds.groupby(["selected_alpha", "selected_K"]).size().reset_index(name="n_outer_folds")
        .sort_values("n_outer_folds", ascending=False)
    )
    selection_frequency_alpha.to_csv(output_root / "SELECTION_FREQUENCY_ALPHA.csv", index=False)
    selection_frequency_k.to_csv(output_root / "SELECTION_FREQUENCY_K.csv", index=False)
    joint.to_csv(output_root / "SELECTION_FREQUENCY_JOINT.csv", index=False)

    numeric_metrics = [
        "outer_silhouette_mean",
        "outer_DB_mean",
        "outer_CH_mean",
        "outer_eigengap_mean",
        "outer_min_cluster_size_median",
        "outer_balance_ratio_mean",
        "outer_seed_pairwise_ARI",
        "outer_seed_pairwise_NMI",
        "inner_resample_ARI",
        "inner_resample_NMI",
    ]
    aggregate_rows: List[Dict[str, float]] = []
    for metric in numeric_metrics:
        values = folds[metric].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.nanmedian(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_root / "NESTED_AGGREGATE_METRICS.csv", index=False)

    alpha06_count = int(np.isclose(folds["selected_alpha"], 0.6).sum())
    k5_count = int((folds["selected_K"] == 5).sum())
    joint_count = int((np.isclose(folds["selected_alpha"], 0.6) & (folds["selected_K"] == 5)).sum())

    def metric_value(metric: str, field: str = "mean") -> float:
        return float(aggregate.loc[aggregate["metric"] == metric, field].iloc[0])

    manuscript_text = f"""NESTED SUBJECT-LEVEL RESAMPLING SUMMARY\n\nOuter folds: {config.outer_folds}\nInner repeats per outer fold: {config.inner_repeats}\nInner fraction: {config.inner_fraction:.2f}\nAlpha grid: {list(config.alpha_grid)}\nK grid: {list(config.k_grid)}\n\nalpha=0.6 selected: {alpha06_count}/{config.outer_folds} outer folds\nK=5 selected: {k5_count}/{config.outer_folds} outer folds\nJoint (alpha=0.6, K=5): {joint_count}/{config.outer_folds} outer folds\n\nHeld-out outer-fold structural metrics (mean +/- SD across outer folds):\nSilhouette: {metric_value('outer_silhouette_mean'):.4f} +/- {metric_value('outer_silhouette_mean', 'sd'):.4f}\nDavies-Bouldin: {metric_value('outer_DB_mean'):.4f} +/- {metric_value('outer_DB_mean', 'sd'):.4f}\nCalinski-Harabasz: {metric_value('outer_CH_mean'):.2f} +/- {metric_value('outer_CH_mean', 'sd'):.2f}\nSeed stability ARI: {metric_value('outer_seed_pairwise_ARI'):.4f} +/- {metric_value('outer_seed_pairwise_ARI', 'sd'):.4f}\nSeed stability NMI: {metric_value('outer_seed_pairwise_NMI'):.4f} +/- {metric_value('outer_seed_pairwise_NMI', 'sd'):.4f}\nMedian minimum cluster size across folds: {metric_value('outer_min_cluster_size_median', 'median'):.1f}\n\nSelection variables audit:\nClinical/demographic variables used for selection: NO\nExcluded: {', '.join(EXCLUDED_SELECTION_VARS)}\n\nInterpretation boundary:\nThis is nested within-cohort downstream model-selection validation using saved subject-level EEG embeddings. It is not independent-cohort replication and not a fully nested retraining of the upstream CNN-autoencoder stages.\n"""
    (output_root / "NESTED_MANUSCRIPT_READY_SUMMARY.txt").write_text(manuscript_text, encoding="utf-8")

    public_dir = output_root / "PUBLIC_AGGREGATE_RESULTS"
    public_dir.mkdir(parents=True, exist_ok=True)
    folds.drop(columns=[], errors="ignore").to_csv(public_dir / "nested_outer_fold_aggregate_results.csv", index=False)
    aggregate.to_csv(public_dir / "nested_aggregate_metrics.csv", index=False)
    selection_frequency_alpha.to_csv(public_dir / "nested_alpha_selection_frequency.csv", index=False)
    selection_frequency_k.to_csv(public_dir / "nested_k_selection_frequency.csv", index=False)
    joint.to_csv(public_dir / "nested_joint_selection_frequency.csv", index=False)
    (public_dir / "nested_summary.txt").write_text(manuscript_text, encoding="utf-8")

    print("\n=== FINALIZED NESTED RESAMPLING ===")
    print(manuscript_text)
    print("Aggregate outputs:", output_root)
    print("Participant-free public outputs:", public_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-root", required=True, help="Root of the private NVIDIA project/results tree.")
    parser.add_argument("--eeg-file", default=None, help="Optional explicit subject-level EEG .npy path.")
    parser.add_argument("--metadata-file", default=None, help="Optional explicit subject metadata CSV path.")
    parser.add_argument("--output-root", required=True, help="New non-overwriting output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-fold-index", type=int, default=None, help="Zero-based outer fold to run; omit to run all sequentially.")
    parser.add_argument("--inner-repeats", type=int, default=10)
    parser.add_argument("--inner-fraction", type=float, default=0.80)
    parser.add_argument("--alpha-grid", type=parse_float_grid, default=parse_float_grid("0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"))
    parser.add_argument("--k-grid", type=parse_int_grid, default=parse_int_grid("2,3,4,5,6,7,8,9,10"))
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--vgae-hidden-dim", type=int, default=64)
    parser.add_argument("--vgae-latent-dim", type=int, default=16)
    parser.add_argument("--vgae-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--outer-eval-seeds", type=int, default=5)
    parser.add_argument(
        "--graph-builder",
        choices=["archived_similarity_rows", "direct_topk_similarity"],
        default="archived_similarity_rows",
        help="Use archived mode for exact consistency with the submitted pipeline.",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to CUDA when available.")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not (0.0 < args.inner_fraction <= 1.0):
        raise ValueError("--inner-fraction must be in (0, 1].")
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2.")
    if args.inner_repeats < 1:
        raise ValueError("--inner-repeats must be at least 1.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    config = Config(
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_repeats=args.inner_repeats,
        inner_fraction=args.inner_fraction,
        alpha_grid=tuple(args.alpha_grid),
        k_grid=tuple(args.k_grid),
        graph_k=args.graph_k,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        vgae_hidden_dim=args.vgae_hidden_dim,
        vgae_latent_dim=args.vgae_latent_dim,
        vgae_epochs=args.vgae_epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        min_cluster_fraction=args.min_cluster_fraction,
        outer_eval_seeds=args.outer_eval_seeds,
        graph_builder=args.graph_builder,
        device=device,
    )
    json_dump(asdict(config), output_root / "RUN_CONFIG.json")
    (output_root / "SELECTION_VARIABLE_AUDIT.txt").write_text(
        "Clinical/demographic variables used in alpha/K selection: NO\nExcluded variables:\n- "
        + "\n- ".join(EXCLUDED_SELECTION_VARS)
        + "\n",
        encoding="utf-8",
    )

    if args.finalize_only:
        finalize_results(output_root, config)
        return

    eeg_file, metadata_file = resolve_inputs(project_root, args.eeg_file, args.metadata_file)
    eeg_raw = np.load(eeg_file).astype(np.float32)
    metadata = pd.read_csv(metadata_file)
    if len(eeg_raw) != len(metadata):
        raise ValueError(f"EEG rows ({len(eeg_raw)}) do not match metadata rows ({len(metadata)}).")
    missing = [col for col in BEHAVIOR_COLS if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required phenotype columns: {missing}")
    pheno_raw = metadata[BEHAVIOR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if "subject" in metadata.columns:
        subject_ids = metadata["subject"].astype(str).to_numpy()
    else:
        subject_ids = np.array([f"row_{i:04d}" for i in range(len(metadata))])
    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError("Subject-level metadata contains duplicate subject identifiers.")

    audit = {
        "project_root": str(project_root),
        "eeg_file": str(eeg_file),
        "eeg_sha256": sha256_file(eeg_file),
        "metadata_file": str(metadata_file),
        "metadata_sha256": sha256_file(metadata_file),
        "n_subjects": int(len(eeg_raw)),
        "eeg_shape": list(eeg_raw.shape),
        "phenotype_shape": list(pheno_raw.shape),
        "phenotype_columns": BEHAVIOR_COLS,
        "excluded_selection_variables": EXCLUDED_SELECTION_VARS,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    json_dump(audit, output_root / "INPUT_AUDIT.json")
    print(json.dumps(audit, indent=2), flush=True)

    kfold = KFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)
    splits = list(kfold.split(np.arange(len(eeg_raw))))
    if args.outer_fold_index is not None:
        if args.outer_fold_index < 0 or args.outer_fold_index >= config.outer_folds:
            raise ValueError("--outer-fold-index is outside the valid range.")
        selected_splits = [(args.outer_fold_index, *splits[args.outer_fold_index])]
    else:
        selected_splits = [(i, train_idx, test_idx) for i, (train_idx, test_idx) in enumerate(splits)]

    start = time.time()
    for outer_index, train_idx, test_idx in selected_splits:
        run_outer_fold(
            outer_index,
            train_idx,
            test_idx,
            eeg_raw,
            pheno_raw,
            subject_ids,
            output_root,
            config,
        )
    elapsed = time.time() - start
    print(f"Completed requested fold work in {elapsed / 60.0:.2f} minutes.", flush=True)

    if args.outer_fold_index is None:
        finalize_results(output_root, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise
