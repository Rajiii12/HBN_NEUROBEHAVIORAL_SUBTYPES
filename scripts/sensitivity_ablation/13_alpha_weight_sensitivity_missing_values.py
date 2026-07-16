#!/usr/bin/env python3
"""
Fixed-setting fusion-weight sensitivity for the EPG-SD pipeline.

This script computes the alpha values that were not present in the archived
joint-selection grid: 0.0, 0.1, 0.2, 0.3, 0.4, 0.9, and 1.0. The reported
alpha=0.6 control is taken from the archived final EPG-SD run when assembling
the manuscript table.

Fixed settings: graph K=12, UMAP neighbors=30, and subtype count K=5.
Participant-level outputs are generated locally and must not be committed.
"""

from pathlib import Path
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap
except Exception as exc:
    raise ImportError("Install umap-learn before running this analysis.") from exc

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Fixed settings
# ============================================================
SEED = 42
GRAPH_K = 12
UMAP_N = 30
FORCED_K = 5
EPOCHS = 300
LR = 1e-3

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.9, 1.0]

ROOT = Path(
    os.environ.get("EPGSD_ANALYSIS_ROOT", "TECHNICAL_FIX_SPECTRAL_UMAP")
)

# Inputs from the fixed subject-level pipeline.
EEG_FILE = ROOT / "subject_embeddings_norm_fixed" / "subject_embedding_mean_std_median_norm_fixed.npy"
META_FILE = ROOT / "subject_embeddings_norm_fixed" / "subject_metadata_norm_fixed.csv"

OUT_ROOT = Path(
    os.environ.get(
        "EPGSD_SENSITIVITY_OUT",
        str(ROOT / "alpha_weight_sensitivity_missing_values"),
    )
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Helpers
# ============================================================
def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = pvals[order[i]] * n / rank
        prev = min(prev, val)
        adj[order[i]] = min(prev, 1.0)
    return adj


def eta_squared_anova(y, labels):
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)

    groups = []
    for g in np.unique(labels):
        vals = y[labels == g]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            groups.append(vals)

    if len(groups) < 2:
        return 1.0, 0.0

    Fval, pval = stats.f_oneway(*groups)

    gm = np.nanmean(y)
    sst = np.nansum((y - gm) ** 2)
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    eta2 = ssb / sst if sst > 0 else 0.0
    return float(pval), float(eta2)


def cosine_affinity(X):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    A = Xn @ Xn.T
    A = np.clip(A, 0, None)
    np.fill_diagonal(A, 1.0)
    return A


def knn_symmetric(A, k=12):
    n = A.shape[0]
    W = np.zeros_like(A)
    for i in range(n):
        idx = np.argsort(-A[i])[:k + 1]
        W[i, idx] = A[i, idx]
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 1.0)
    return W


def normalize_adj_dense(A):
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-9)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


# ============================================================
# Simple VGAE
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        return self.lin(adj_norm @ x)


class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn_mu = GCNLayer(hidden_dim, latent_dim)
        self.gcn_logstd = GCNLayer(hidden_dim, latent_dim)

    def encode(self, x, adj_norm):
        h = F.relu(self.gcn1(x, adj_norm))
        mu = self.gcn_mu(h, adj_norm)
        logstd = self.gcn_logstd(h, adj_norm).clamp(-4, 4)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(logstd)

    def decode_logits(self, z):
        return z @ z.T

    def forward(self, x, adj_norm):
        mu, logstd = self.encode(x, adj_norm)
        z = self.reparameterize(mu, logstd)
        logits = self.decode_logits(z)
        return logits, mu, logstd, z


def train_vgae(X_node, A, out_dir):
    set_seed(SEED)

    A_bin = (A > 0).astype(np.float32)
    adj_norm = normalize_adj_dense(A).astype(np.float32)

    x = torch.tensor(X_node, dtype=torch.float32, device=DEVICE)
    adj_t = torch.tensor(adj_norm, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(A_bin, dtype=torch.float32, device=DEVICE)

    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1e-9)).clamp(min=1.0, max=50.0)

    model = VGAE(in_dim=X_node.shape[1], hidden_dim=128, latent_dim=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        logits, mu, logstd, z = model(x, adj_t)

        recon = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )

        # KL divergence
        kl = -0.5 * torch.mean(torch.sum(
            1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd),
            dim=1
        ))

        loss = recon + 0.01 * kl
        loss.backward()
        opt.step()

        if ep % 25 == 0 or ep == 1:
            logs.append({
                "epoch": ep,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu())
            })
            print(f"    epoch={ep:03d} loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")

    model.eval()
    with torch.no_grad():
        mu, logstd = model.encode(x, adj_t)
        z = mu.detach().cpu().numpy()

    pd.DataFrame(logs).to_csv(out_dir / "vgae_training_log.csv", index=False)
    np.save(out_dir / "latent_embeddings.npy", z)
    pd.DataFrame(z).to_csv(out_dir / "latent_embeddings.csv", index=False)
    return z


def run_umap(Z):
    reducer = umap.UMAP(
        n_neighbors=UMAP_N,
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED,
    )
    return reducer.fit_transform(Z)


def clinical_stats(meta, labels, clinical_cols):
    rows = []
    pvals = []
    tmp = []

    for name, col in clinical_cols.items():
        y = pd.to_numeric(meta[col], errors="coerce").values
        p, e2 = eta_squared_anova(y, labels)
        pvals.append(p)
        tmp.append((name, col, p, e2))

    fdr = bh_fdr(pvals)

    for i, (name, col, p, e2) in enumerate(tmp):
        rows.append({
            "variable": name,
            "column": col,
            "anova_p": p,
            "anova_p_fdr": float(fdr[i]),
            "eta_squared": e2,
            "significant_fdr_0p05": bool(fdr[i] < 0.05)
        })

    return pd.DataFrame(rows)


# ============================================================
# Load data
# ============================================================
print("Device:", DEVICE)
print("EEG_FILE:", EEG_FILE)
print("META_FILE:", META_FILE)

if not EEG_FILE.exists():
    raise FileNotFoundError(f"Missing EEG file: {EEG_FILE}")
if not META_FILE.exists():
    raise FileNotFoundError(f"Missing metadata file: {META_FILE}")

X_eeg = np.load(EEG_FILE)
meta = pd.read_csv(META_FILE)

print("X_eeg shape:", X_eeg.shape)
print("metadata shape:", meta.shape)
print("metadata columns:", list(meta.columns))

# Detect phenotype columns
col_pfactor = find_col(meta, ["p_factor", "pfactor", "p factor", "p-factor"])
col_attention = find_col(meta, ["attention", "attention_score"])
col_internal = find_col(meta, ["internalizing", "internal"])
col_external = find_col(meta, ["externalizing", "external"])
col_age = find_col(meta, ["age"])
col_ehq = find_col(meta, ["ehq_total", "ehq", "handedness"])

needed = {
    "p_factor": col_pfactor,
    "attention": col_attention,
    "internalizing": col_internal,
    "externalizing": col_external,
    "age": col_age,
    "ehq_total": col_ehq,
}

print("\nDetected columns:")
for k, v in needed.items():
    print(f"  {k}: {v}")

missing = [k for k, v in needed.items() if v is None]
if missing:
    raise ValueError(f"Could not detect columns: {missing}")

PHENO_COLS = [col_pfactor, col_attention, col_internal, col_external]

CLINICAL_COLS = {
    "p_factor": col_pfactor,
    "attention": col_attention,
    "internalizing": col_internal,
    "externalizing": col_external,
    "age": col_age,
    "ehq_total": col_ehq,
}

X_phe = meta[PHENO_COLS].apply(pd.to_numeric, errors="coerce")
X_phe = X_phe.fillna(X_phe.median(numeric_only=True)).values

Xe = StandardScaler().fit_transform(np.nan_to_num(X_eeg))
Xp = StandardScaler().fit_transform(np.nan_to_num(X_phe))

X_node = StandardScaler().fit_transform(np.concatenate([Xe, Xp], axis=1))

A_eeg = cosine_affinity(Xe)
A_phe = cosine_affinity(Xp)

summary_rows = []


# ============================================================
# Run missing alphas
# ============================================================
for alpha in ALPHAS:
    print("\n" + "=" * 90)
    print(f"Running fixed alpha sweep: alpha={alpha:.1f}, graph_k={GRAPH_K}, UMAP={UMAP_N}, K={FORCED_K}")
    print("=" * 90)

    alpha_tag = str(alpha).replace(".", "p")
    out_dir = OUT_ROOT / f"graphK{GRAPH_K}_alpha{alpha_tag}" / f"UMAP{UMAP_N}_K{FORCED_K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    A_fused = alpha * A_eeg + (1.0 - alpha) * A_phe
    W = knn_symmetric(A_fused, GRAPH_K)

    np.save(out_dir.parent / "fused_similarity.npy", A_fused)
    np.save(out_dir.parent / "adjacency.npy", W)
    np.save(out_dir.parent / "node_features.npy", X_node)

    # VGAE
    vgae_dir = out_dir.parent / "vgae"
    vgae_dir.mkdir(parents=True, exist_ok=True)
    Z = train_vgae(X_node, W, vgae_dir)

    # UMAP
    U = run_umap(Z)
    pd.DataFrame(U, columns=["UMAP1", "UMAP2"]).to_csv(out_dir / "umap_coordinates.csv", index=False)

    # Spectral clustering on UMAP embedding
    labels = SpectralClustering(
        n_clusters=FORCED_K,
        affinity="nearest_neighbors",
        n_neighbors=min(12, len(U) - 1),
        assign_labels="kmeans",
        random_state=SEED
    ).fit_predict(U)

    # Quality metrics
    counts = np.bincount(labels, minlength=FORCED_K)
    sil = silhouette_score(U, labels)
    db = davies_bouldin_score(U, labels)
    ch = calinski_harabasz_score(U, labels)
    min_cluster = int(counts.min())
    max_cluster = int(counts.max())
    balance = float(max_cluster / max(min_cluster, 1))

    # Clinical / profiling stats
    clin = clinical_stats(meta, labels, CLINICAL_COLS)
    clin.to_csv(out_dir / "clinical_stats.csv", index=False)

    fdr_sig = int(clin["significant_fdr_0p05"].sum())
    mean_eta2 = float(clin["eta_squared"].mean())

    # Save labels and quality
    pd.DataFrame({
        "subject_index": np.arange(len(labels)),
        "subtype": labels
    }).to_csv(out_dir / "subject_subtypes.csv", index=False)

    pd.DataFrame({
        "cluster": np.arange(FORCED_K),
        "size": counts
    }).to_csv(out_dir / "cluster_sizes.csv", index=False)

    q = {
        "graph_k": GRAPH_K,
        "alpha": alpha,
        "umap_n_neighbors": UMAP_N,
        "forced_k": FORCED_K,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "n_clusters_found": int(len(np.unique(labels))),
        "n_subjects": int(len(labels)),
        "min_cluster_size": min_cluster,
        "max_cluster_size": max_cluster,
        "cluster_size_ratio_max_min": balance,
        "mean_eta_squared": mean_eta2,
        "fdr_sig_count": f"{fdr_sig}/{len(CLINICAL_COLS)}",
        "output_dir": str(out_dir)
    }

    # Add individual important validation fields
    for _, row in clin.iterrows():
        var = row["variable"]
        q[f"{var}_anova_p_fdr"] = row["anova_p_fdr"]
        q[f"{var}_eta_squared"] = row["eta_squared"]

    pd.DataFrame([q]).to_csv(out_dir / "main_quality.csv", index=False)
    summary_rows.append(q)

    print(f"\nRESULT alpha={alpha:.1f}")
    print(f"  silhouette={sil:.4f}")
    print(f"  DB={db:.4f}")
    print(f"  CH={ch:.2f}")
    print(f"  min_cluster={min_cluster}")
    print(f"  balance={balance:.3f}")
    print(f"  mean_eta2={mean_eta2:.4f}")
    print(f"  FDR={fdr_sig}/{len(CLINICAL_COLS)}")
    print(f"  saved={out_dir}")


summary = pd.DataFrame(summary_rows).sort_values("alpha")
summary_path = OUT_ROOT / "alpha_fixed_clean_sweep_missing_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 90)
print("DONE")
print("Saved summary:", summary_path)
print(summary[[
    "alpha", "silhouette", "davies_bouldin", "calinski_harabasz",
    "min_cluster_size", "cluster_size_ratio_max_min",
    "mean_eta_squared", "fdr_sig_count",
    "age_anova_p_fdr", "age_eta_squared",
    "ehq_total_anova_p_fdr", "ehq_total_eta_squared"
]].to_string(index=False))
