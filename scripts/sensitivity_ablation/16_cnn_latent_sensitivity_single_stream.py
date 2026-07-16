#!/usr/bin/env python3
"""CNN-AE latent-dimension sensitivity using one saved embedding stream.

Latent dimensions 64, 128, and 256 are evaluated with the same downstream
configuration. This archived experiment uses the dynamic CNN-AE stream for the
latent-size comparison; it is not a two-stream retraining experiment.
Participant-level outputs must not be committed.
"""
import warnings
import os
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import umap


# =========================
# PATHS
# =========================
ROOT = Path(os.environ.get("EPGSD_PROJECT_ROOT", "."))
BASE = Path(
    os.environ.get(
        "EPGSD_CNN_LATENT_ROOT",
        str(ROOT / "TECHNICAL_FIX_SPECTRAL_UMAP" / "cnn_latent_sensitivity"),
    )
)
OUT = Path(os.environ.get("EPGSD_SENSITIVITY_OUT", str(BASE / "downstream_results")))
OUT.mkdir(parents=True, exist_ok=True)

META_FILE = ROOT / "TECHNICAL_FIX_SPECTRAL_UMAP/multiview_tensors_FS500/band_decomposition_metadata_FS500.csv"
SUBJECT_PHENO_FILE = ROOT / "TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_norm_fixed/subject_metadata_norm_fixed.csv"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
LATENTS = [64, 128, 256]

PCA_DIM = 64
ALPHA = 0.6
GRAPH_K = 12
UMAP_NEIGHBORS = 30
VGAE_LATENT = 16
N_CLUSTERS = 5
SEED = 42

PHENO_CANDIDATES = {
    "p_factor": ["p_factor", "pfactor", "P_factor", "P-Factor", "p-factor", "Pfactor"],
    "attention": ["attention", "Attention"],
    "internalizing": ["internalizing", "Internalizing"],
    "externalizing": ["externalizing", "Externalizing"],
}

CLINICAL_CANDIDATES = {
    "p_factor": ["p_factor", "pfactor", "P_factor", "P-Factor", "p-factor", "Pfactor"],
    "attention": ["attention", "Attention"],
    "internalizing": ["internalizing", "Internalizing"],
    "externalizing": ["externalizing", "Externalizing"],
    "age": ["age", "Age", "interview_age", "participant_age"],
    "ehq": ["ehq", "EHQ", "EHQ_Total", "ehq_total", "EHI", "handedness"],
}


# =========================
# UTILS
# =========================
def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh

    reject = np.zeros(n, dtype=bool)
    if np.any(passed):
        max_i = np.where(passed)[0].max()
        reject[order[:max_i + 1]] = True

    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.minimum(q_ranked, 1.0)
    return reject, qvals


def clinical_stats(labels, meta):
    pvals, etas, names = [], [], []

    for name, candidates in CLINICAL_CANDIDATES.items():
        col = find_col(meta, candidates)
        if col is None:
            continue

        y = pd.to_numeric(meta[col], errors="coerce").values
        mask = np.isfinite(y)
        if mask.sum() < 20:
            continue

        yv = y[mask]
        lv = labels[mask]

        groups = [yv[lv == k] for k in np.unique(lv)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue

        try:
            _, p = stats.kruskal(*groups)
        except Exception:
            continue

        grand = np.mean(yv)
        ss_total = np.sum((yv - grand) ** 2)
        ss_between = 0.0
        for g in groups:
            ss_between += len(g) * (np.mean(g) - grand) ** 2
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0

        names.append(name)
        pvals.append(p)
        etas.append(eta2)

    if len(pvals) == 0:
        return 0.0, 0, {}

    reject, qvals = fdr_bh(pvals)
    details = {
        n: {"p": float(p), "q": float(q), "eta2": float(e), "sig": bool(r)}
        for n, p, q, e, r in zip(names, pvals, qvals, etas, reject)
    }
    return float(np.mean(etas)), int(np.sum(reject)), details


def get_subject_column(meta):
    candidates = [
        "subject", "Subject", "participant_id", "participant", "sub",
        "subject_id", "Subject_ID", "sub_id"
    ]
    col = find_col(meta, candidates)
    if col is None:
        raise ValueError(f"Cannot find subject column. Available columns: {list(meta.columns)[:50]}")
    return col


def get_pheno_matrix(meta):
    cols = []
    for key, candidates in PHENO_CANDIDATES.items():
        col = find_col(meta, candidates)
        if col is None:
            raise ValueError(f"Cannot find phenotype column for {key}. Columns: {list(meta.columns)[:80]}")
        cols.append(col)

    P = meta[cols].apply(pd.to_numeric, errors="coerce")
    P = P.fillna(P.median(numeric_only=True))
    return P.values.astype(np.float32), cols


# =========================
# VGAE
# =========================
class DenseGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        return adj_norm @ self.lin(x)


class DenseVGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, dropout=0.0):
        super().__init__()
        self.gc1 = DenseGraphConv(input_dim, hidden_dim)
        self.gc_mu = DenseGraphConv(hidden_dim, latent_dim)
        self.gc_logstd = DenseGraphConv(hidden_dim, latent_dim)
        self.dropout = dropout

    def encode(self, x, adj_norm):
        h = F.relu(self.gc1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.gc_mu(h, adj_norm)
        logstd = self.gc_logstd(h, adj_norm)
        return mu, logstd

    def reparam(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(logstd)
        return mu

    def forward(self, x, adj_norm):
        mu, logstd = self.encode(x, adj_norm)
        z = self.reparam(mu, logstd)
        recon = torch.sigmoid(z @ z.t())
        return recon, mu, logstd


def normalize_adj(A):
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]


def train_vgae(X, A, latent_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    A_norm_t = torch.tensor(normalize_adj(A), dtype=torch.float32, device=device)

    model = DenseVGAE(X.shape[1], hidden_dim=64, latent_dim=latent_dim, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = A.shape[0]

    for epoch in range(500):
        model.train()
        recon, mu, logstd = model(X_t, A_norm_t)

        bce = F.binary_cross_entropy(recon, A_t)
        kl = -0.5 / n * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1))
        loss = bce + kl

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t, A_norm_t)

    return mu.detach().cpu().numpy()


# =========================
# MAIN PIPELINE
# =========================
def run_one(latent_dim):
    print("\n" + "=" * 90)
    print(f"RUNNING CNN LATENT DIM = {latent_dim}")
    print("=" * 90)

    emb_dir = BASE / f"latent{latent_dim}" / "autoencoder_embeddings_dynamic_norm_fixed"

    band_embs = []
    for band in BANDS:
        f = emb_dir / f"{band}_embedding.npy"
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        Z = np.load(f).astype(np.float32)
        print(f"{band}: {Z.shape}")
        Z = StandardScaler().fit_transform(Z)
        band_embs.append(Z)

    H = np.concatenate(band_embs, axis=1)
    print("Concatenated recording embedding:", H.shape)

    meta = pd.read_csv(META_FILE)
    print("Metadata:", meta.shape)

    if len(meta) != H.shape[0]:
        raise ValueError(f"Metadata rows {len(meta)} != embeddings rows {H.shape[0]}")

    pca_dim = min(PCA_DIM, H.shape[1], H.shape[0] - 1)
    U = PCA(n_components=pca_dim, random_state=SEED).fit_transform(H)
    print("PCA recording embedding:", U.shape)

    subj_col = get_subject_column(meta)
    subjects = meta[subj_col].astype(str).values
    unique_subjects = sorted(pd.unique(subjects))
    print("Unique subjects:", len(unique_subjects))

    rows = []
    feats = []
    meta_rows = []

    for s in unique_subjects:
        idx = np.where(subjects == s)[0]
        if len(idx) == 0:
            continue

        Xs = U[idx]
        mean = Xs.mean(axis=0)
        std = Xs.std(axis=0)
        med = np.median(Xs, axis=0)
        feats.append(np.concatenate([mean, std, med]))

        first = meta.iloc[idx[0]].copy()
        first["subject_for_analysis"] = s
        first["n_recordings_used"] = len(idx)
        meta_rows.append(first)

    E = np.vstack(feats).astype(np.float32)
    subject_meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    # Merge subject-level phenotype/clinical metadata.
    # The FS500 band-decomposition metadata is recording-level and contains only
    # file/subject information, so phenotype variables are taken from the final
    # 430-subject metadata used in the original EPG-SD pipeline.
    pheno_meta = pd.read_csv(SUBJECT_PHENO_FILE)
    pheno_subject_col = get_subject_column(pheno_meta)

    subject_meta["subject_merge_key"] = subject_meta["subject_for_analysis"].astype(str)
    pheno_meta["subject_merge_key"] = pheno_meta[pheno_subject_col].astype(str)

    before_cols = set(subject_meta.columns)
    subject_meta = subject_meta.merge(
        pheno_meta.drop_duplicates("subject_merge_key"),
        on="subject_merge_key",
        how="left",
        suffixes=("", "_pheno")
    )

    print("Subject EEG embedding:", E.shape)
    print("Subject metadata after phenotype merge:", subject_meta.shape)
    print("Merged phenotype columns added:", [c for c in subject_meta.columns if c not in before_cols][:30])

    P, pheno_cols = get_pheno_matrix(subject_meta)
    print("Phenotype columns used:", pheno_cols)
    Pz = StandardScaler().fit_transform(P)

    Ez = StandardScaler().fit_transform(E)

    A_eeg = cosine_similarity(Ez)
    A_phe = cosine_similarity(Pz)
    A_fused = ALPHA * A_eeg + (1.0 - ALPHA) * A_phe

    # kNN sparsification
    n = A_fused.shape[0]
    A = np.zeros_like(A_fused, dtype=np.float32)
    for i in range(n):
        idx = np.argsort(A_fused[i])[-(GRAPH_K + 1):]
        idx = idx[idx != i]
        A[i, idx] = 1.0
    A = np.maximum(A, A.T)

    X_node = np.concatenate([Ez, Pz], axis=1).astype(np.float32)

    Z_vgae = train_vgae(X_node, A, latent_dim=VGAE_LATENT)

    Y = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=2,
        random_state=SEED,
        min_dist=0.1
    ).fit_transform(Z_vgae)

    labels = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity="nearest_neighbors",
        n_neighbors=12,
        assign_labels="kmeans",
        random_state=SEED
    ).fit_predict(Y)

    counts = Counter(labels)
    min_size = min(counts.values())
    max_size = max(counts.values())
    balance = max_size / min_size

    sil = silhouette_score(Y, labels)
    db = davies_bouldin_score(Y, labels)
    ch = calinski_harabasz_score(Y, labels)

    mean_eta2, fdr_count, clin_details = clinical_stats(labels, subject_meta)

    run_out = OUT / f"latent{latent_dim}"
    run_out.mkdir(parents=True, exist_ok=True)

    np.save(run_out / "subject_embedding_mean_std_median.npy", E)
    np.save(run_out / "vgae_embedding.npy", Z_vgae)
    np.save(run_out / "umap_embedding.npy", Y)

    subject_meta["cluster"] = labels
    subject_meta.to_csv(run_out / "subject_subtypes.csv", index=False)

    row = {
        "cnn_latent_dim": latent_dim,
        "recording_embedding_dim": H.shape[1],
        "pca_dim": pca_dim,
        "subject_embedding_dim": E.shape[1],
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "balance_ratio": balance,
        "mean_eta2": mean_eta2,
        "fdr_count": fdr_count,
        "cluster_sizes": dict(counts),
        "clinical_details": clin_details,
    }

    pd.DataFrame([row]).to_csv(run_out / "summary.csv", index=False)

    print("RESULT:", row)
    return row


def main():
    rows = []
    for dim in LATENTS:
        rows.append(run_one(dim))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cnn_latent_sensitivity_results.csv", index=False)

    # LaTeX table
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Sensitivity of EPG-SD to the CNN autoencoder latent dimension. Each latent size was retrained at the band-wise CNN autoencoder stage and evaluated using the same downstream configuration: PCA dimension 64, graph $K_g=12$, fusion weight $\alpha=0.6$, UMAP neighbors 30, VGAE latent dimension 16, and subtype number $K=5$.}")
    lines.append(r"\label{tab:cnn_latent_sensitivity}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"CNN latent & Sil. $\uparrow$ & DB $\downarrow$ & CH $\uparrow$ & Min. $\uparrow$ & Mean $\eta^2$ $\uparrow$ & FDR \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        dim = int(r["cnn_latent_dim"])
        dim_txt = rf"\textbf{{{dim}}}" if dim == 128 else str(dim)
        vals = [
            f"{r['silhouette']:.3f}",
            f"{r['davies_bouldin']:.3f}",
            f"{r['calinski_harabasz']:.1f}",
            f"{int(r['min_cluster_size'])}",
            f"{r['mean_eta2']:.3f}",
            f"{int(r['fdr_count'])}/6",
        ]
        if dim == 128:
            vals = [rf"\textbf{{{v}}}" for v in vals]
        lines.append(dim_txt + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    (OUT / "cnn_latent_sensitivity_latex_table.txt").write_text(latex)

    print("\nDONE.")
    print("Saved:", OUT / "cnn_latent_sensitivity_results.csv")
    print("Saved:", OUT / "cnn_latent_sensitivity_latex_table.txt")
    print("\nLATEX TABLE:\n")
    print(latex)


if __name__ == "__main__":
    main()
