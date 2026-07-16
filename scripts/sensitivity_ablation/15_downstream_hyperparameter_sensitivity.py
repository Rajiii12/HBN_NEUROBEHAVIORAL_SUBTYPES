#!/usr/bin/env python3
"""Controlled downstream sensitivity for PCA, graph, UMAP, and VGAE dimensions.

The numerical implementation is preserved from the experiment that generated
the manuscript sensitivity table. Each block changes one setting at a time.
Participant-level outputs must not be committed.
"""
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh

    reject = np.zeros(n, dtype=bool)
    if np.any(passed):
        max_i = np.where(passed)[0].max()
        reject[order[:max_i + 1]] = True

    qvals_ranked = ranked * n / np.arange(1, n + 1)
    qvals_ranked = np.minimum.accumulate(qvals_ranked[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.minimum(qvals_ranked, 1.0)
    return reject, qvals


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import umap
except Exception as e:
    raise RuntimeError("Install umap-learn first: pip install umap-learn") from e


# =========================
# CONFIG
# =========================
ROOT = Path(os.environ.get("EPGSD_PROJECT_ROOT", "."))
OUT = Path(
    os.environ.get(
        "EPGSD_SENSITIVITY_OUT",
        str(ROOT / "TECHNICAL_FIX_SPECTRAL_UMAP" / "hyperparameter_sensitivity"),
    )
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLUSTERS = 5
ALPHA = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINAL = {
    "pca_dim": 64,
    "graph_k": 12,
    "umap_neighbors": 30,
    "vgae_latent": 16,
}

PCA_DIMS = [32, 64, 96]
GRAPH_KS = [8, 12, 16]
UMAP_NEIGHBORS = [15, 30, 50]
VGAE_LATENTS = [8, 16, 32]

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# FILE DISCOVERY
# =========================
def find_first(patterns):
    for pat in patterns:
        hits = list(ROOT.rglob(pat))
        hits = [h for h in hits if h.is_file()]
        if hits:
            hits = sorted(hits, key=lambda x: len(str(x)))
            return hits[0]
    return None


embedding_override = os.environ.get("EPGSD_SUBJECT_EMBEDDING")
metadata_override = os.environ.get("EPGSD_SUBJECT_METADATA")

EMB_PATH = (
    Path(embedding_override)
    if embedding_override
    else find_first([
        "subject_embedding_mean_std_median_norm_fixed.npy",
        "*subject*embedding*mean*std*median*.npy",
        "*subject_embedding*.npy",
    ])
)

META_PATH = (
    Path(metadata_override)
    if metadata_override
    else find_first([
        "subject_metadata_norm_fixed.csv",
        "*subject*metadata*.csv",
    ])
)

if EMB_PATH is None:
    raise FileNotFoundError("Could not find subject embedding .npy file.")
if META_PATH is None:
    raise FileNotFoundError("Could not find subject metadata .csv file.")

print("================================================================================")
print("Downstream hyperparameter sensitivity")
print("================================================================================")
print("Embedding:", EMB_PATH)
print("Metadata :", META_PATH)
print("Output   :", OUT)
print("Device   :", DEVICE)

X_eeg_raw = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)

print("X_eeg_raw shape:", X_eeg_raw.shape)
print("metadata shape :", meta.shape)


# =========================
# CLINICAL COLUMNS
# =========================
def norm_col(c):
    return str(c).lower().replace(" ", "").replace("_", "").replace("-", "")

col_map = {norm_col(c): c for c in meta.columns}

candidate_map = {
    "p_factor": ["pfactor", "p", "p_factor", "generalpsychopathology"],
    "attention": ["attention", "adhd", "attentionscore"],
    "internalizing": ["internalizing", "internal"],
    "externalizing": ["externalizing", "external"],
    "age": ["age"],
    "ehq": ["ehq", "ehqtotal", "handedness"],
}

clinical_cols = []
for name, keys in candidate_map.items():
    found = None
    for k in keys:
        nk = norm_col(k)
        for cc_norm, cc in col_map.items():
            if nk == cc_norm or nk in cc_norm:
                found = cc
                break
        if found:
            break
    if found and found not in clinical_cols:
        clinical_cols.append(found)

if len(clinical_cols) == 0:
    raise RuntimeError(f"No clinical columns found. Metadata columns: {list(meta.columns)}")

print("Clinical columns:", clinical_cols)


# phenotype columns used for graph fusion
phenotype_keys = ["pfactor", "attention", "internalizing", "externalizing"]
phenotype_cols = []
for key in phenotype_keys:
    for cc_norm, cc in col_map.items():
        if key in cc_norm:
            phenotype_cols.append(cc)
            break

# fallback: use first four clinical behavioral columns excluding age/ehq
if len(phenotype_cols) < 4:
    phenotype_cols = [c for c in clinical_cols if "age" not in norm_col(c) and "ehq" not in norm_col(c)][:4]

if len(phenotype_cols) < 4:
    raise RuntimeError(f"Could not identify 4 phenotype columns. Found: {phenotype_cols}")

print("Phenotype graph columns:", phenotype_cols)


# align N
N = min(X_eeg_raw.shape[0], len(meta))
X_eeg_raw = X_eeg_raw[:N]
meta = meta.iloc[:N].reset_index(drop=True)

X_pheno_raw = meta[phenotype_cols].apply(pd.to_numeric, errors="coerce").fillna(meta[phenotype_cols].median(numeric_only=True)).values
X_pheno_raw = StandardScaler().fit_transform(X_pheno_raw)


# =========================
# MODEL
# =========================
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A_norm):
        return A_norm @ self.W(X)


class DenseVGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, dropout=0.0):
        super().__init__()
        self.gc1 = DenseGCNLayer(input_dim, hidden_dim)
        self.gc_mu = DenseGCNLayer(hidden_dim, latent_dim)
        self.gc_logvar = DenseGCNLayer(hidden_dim, latent_dim)
        self.dropout = dropout

    def encode(self, X, A_norm):
        h = F.relu(self.gc1(X, A_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.gc_mu(h, A_norm)
        logvar = self.gc_logvar(h, A_norm)
        return mu, logvar

    def reparam(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, X, A_norm):
        mu, logvar = self.encode(X, A_norm)
        z = self.reparam(mu, logvar)
        A_pred = torch.sigmoid(z @ z.T)
        return A_pred, mu, logvar


def normalize_adj(A):
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_fused_graph(X_eeg, X_pheno, graph_k=12, alpha=0.6):
    X_eeg = StandardScaler().fit_transform(X_eeg)
    X_pheno = StandardScaler().fit_transform(X_pheno)

    S_eeg = cosine_similarity(X_eeg)
    S_eeg = (S_eeg + 1.0) / 2.0

    S_pheno = rbf_kernel(X_pheno, gamma=1.0 / max(1, X_pheno.shape[1]))

    S = alpha * S_eeg + (1.0 - alpha) * S_pheno
    np.fill_diagonal(S, 0.0)

    A = np.zeros_like(S)
    for i in range(S.shape[0]):
        idx = np.argsort(S[i])[-graph_k:]
        A[i, idx] = S[i, idx]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    return A


def train_vgae(X, A, latent_dim=16, epochs=500, lr=1e-3):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    A_bin = (A > 0).astype(np.float32)
    A_norm = normalize_adj(A_bin)

    A_t = torch.tensor(A_bin, dtype=torch.float32, device=DEVICE)
    A_norm_t = torch.tensor(A_norm, dtype=torch.float32, device=DEVICE)

    model = DenseVGAE(input_dim=X.shape[1], hidden_dim=64, latent_dim=latent_dim, dropout=0.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        A_pred, mu, logvar = model(X_t, A_norm_t)

        recon = F.binary_cross_entropy(A_pred, A_t)
        kl = -0.5 / X.shape[0] * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon + kl

        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_t, A_norm_t)
    return mu.detach().cpu().numpy()


def cluster_and_score(Z, umap_neighbors=30):
    Y = umap.UMAP(
        n_neighbors=umap_neighbors,
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED,
    ).fit_transform(Z)

    labels = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity="nearest_neighbors",
        n_neighbors=12,
        assign_labels="kmeans",
        random_state=SEED,
    ).fit_predict(Y)

    sil = silhouette_score(Y, labels)
    db = davies_bouldin_score(Y, labels)
    ch = calinski_harabasz_score(Y, labels)

    counts = np.bincount(labels, minlength=N_CLUSTERS)
    min_size = int(counts.min())
    max_size = int(counts.max())
    balance = float(max_size / max(1, min_size))

    return labels, Y, sil, db, ch, min_size, balance


def clinical_stats(labels):
    eta_vals = []
    p_vals = []
    used_cols = []

    for col in clinical_cols:
        y = pd.to_numeric(meta[col], errors="coerce")
        valid = y.notna().values
        if valid.sum() < 20:
            continue

        groups = [y[(labels == k) & valid].values for k in np.unique(labels)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        try:
            f, p = stats.f_oneway(*groups)
        except Exception:
            continue

        overall = y[valid].values
        grand_mean = np.mean(overall)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((overall - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        if not np.isnan(eta2):
            eta_vals.append(float(eta2))
            p_vals.append(float(p))
            used_cols.append(col)

    if len(p_vals) > 0:
        reject, q = fdr_bh(p_vals, alpha=0.05)
        fdr_count = int(np.sum(reject))
    else:
        fdr_count = 0

    return float(np.mean(eta_vals)) if eta_vals else np.nan, fdr_count


def evaluate(config_name, hyperparameter, value, pca_dim=64, graph_k=12, umap_neighbors=30, vgae_latent=16):
    print(f"\nRUN: {config_name} | {hyperparameter}={value}")

    # PCA compression of existing EEG subject embedding
    dim = min(pca_dim, X_eeg_raw.shape[1], N - 1)
    X_eeg = PCA(n_components=dim, random_state=SEED).fit_transform(StandardScaler().fit_transform(X_eeg_raw))

    X_node = np.concatenate([StandardScaler().fit_transform(X_eeg), X_pheno_raw], axis=1)
    A = build_fused_graph(X_eeg, X_pheno_raw, graph_k=graph_k, alpha=ALPHA)

    Z = train_vgae(X_node, A, latent_dim=vgae_latent, epochs=500, lr=1e-3)
    labels, Y, sil, db, ch, min_size, balance = cluster_and_score(Z, umap_neighbors=umap_neighbors)
    mean_eta2, fdr_count = clinical_stats(labels)

    row = {
        "config_name": config_name,
        "hyperparameter": hyperparameter,
        "value": value,
        "pca_dim": pca_dim,
        "graph_k": graph_k,
        "umap_neighbors": umap_neighbors,
        "vgae_latent": vgae_latent,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "min_cluster_size": min_size,
        "balance_ratio": balance,
        "mean_eta2": mean_eta2,
        "fdr_count": fdr_count,
    }

    print(row)
    return row


rows = []

# PCA sensitivity
for d in PCA_DIMS:
    rows.append(evaluate(f"pca_dim_{d}", "PCA dimension", d, pca_dim=d, graph_k=FINAL["graph_k"], umap_neighbors=FINAL["umap_neighbors"], vgae_latent=FINAL["vgae_latent"]))

# Graph K sensitivity
for k in GRAPH_KS:
    rows.append(evaluate(f"graph_k_{k}", "Graph K", k, pca_dim=FINAL["pca_dim"], graph_k=k, umap_neighbors=FINAL["umap_neighbors"], vgae_latent=FINAL["vgae_latent"]))

# UMAP neighbors sensitivity
for umap_nn in UMAP_NEIGHBORS:
    rows.append(evaluate(f"umap_neighbors_{umap_nn}", "UMAP neighbors", umap_nn, pca_dim=FINAL["pca_dim"], graph_k=FINAL["graph_k"], umap_neighbors=umap_nn, vgae_latent=FINAL["vgae_latent"]))

# VGAE latent sensitivity
for zdim in VGAE_LATENTS:
    rows.append(evaluate(f"vgae_latent_{zdim}", "VGAE latent dimension", zdim, pca_dim=FINAL["pca_dim"], graph_k=FINAL["graph_k"], umap_neighbors=FINAL["umap_neighbors"], vgae_latent=zdim))


df = pd.DataFrame(rows)
df.to_csv(OUT / "hyperparameter_sensitivity_results.csv", index=False)

# Build LaTeX table
def bold_if_final(h, v, text):
    final_map = {
        "PCA dimension": 64,
        "Graph K": 12,
        "UMAP neighbors": 30,
        "VGAE latent dimension": 16,
    }
    return f"\\textbf{{{text}}}" if final_map.get(h) == v else str(text)

latex_lines = []
latex_lines.append(r"\begin{table*}[!t]")
latex_lines.append(r"	\caption{Hyperparameter sensitivity analysis for the final EPG-SD configuration. Each row varies one hyperparameter while keeping the other components fixed. The final selected setting is highlighted in bold.}")
latex_lines.append(r"	\label{tab:hyperparameter_sensitivity}")
latex_lines.append(r"	\centering")
latex_lines.append(r"	\footnotesize")
latex_lines.append(r"	\setlength{\tabcolsep}{4pt}")
latex_lines.append(r"	\begin{tabular}{llcccccc}")
latex_lines.append(r"		\toprule")
latex_lines.append(r"		Hyperparameter & Value & Sil. $\uparrow$ & DB $\downarrow$ & CH $\uparrow$ & Min. size $\uparrow$ & Mean $\eta^2$ $\uparrow$ & FDR \\")
latex_lines.append(r"		\midrule")

last_h = None
for _, r in df.iterrows():
    h = r["hyperparameter"]
    if last_h is not None and h != last_h:
        latex_lines.append(r"		\midrule")
    last_h = h

    val = int(r["value"])
    sil = f"{r['silhouette']:.3f}"
    db = f"{r['davies_bouldin']:.3f}"
    ch = f"{r['calinski_harabasz']:.1f}"
    mn = f"{int(r['min_cluster_size'])}"
    eta = f"{r['mean_eta2']:.3f}"
    fdr = f"{int(r['fdr_count'])}/{len(clinical_cols)}"

    final_map = {
        "PCA dimension": 64,
        "Graph K": 12,
        "UMAP neighbors": 30,
        "VGAE latent dimension": 16,
    }
    if final_map.get(h) == val:
        line = rf"		{h} & \textbf{{{val}}} & \textbf{{{sil}}} & \textbf{{{db}}} & \textbf{{{ch}}} & \textbf{{{mn}}} & \textbf{{{eta}}} & \textbf{{{fdr}}} \\"
    else:
        line = rf"		{h} & {val} & {sil} & {db} & {ch} & {mn} & {eta} & {fdr} \\"
    latex_lines.append(line)

latex_lines.append(r"		\bottomrule")
latex_lines.append(r"	\end{tabular}")
latex_lines.append(r"\end{table*}")

latex_text = "\n".join(latex_lines)
(OUT / "hyperparameter_sensitivity_latex_table.txt").write_text(latex_text)

summary = {
    "embedding_path": str(EMB_PATH),
    "metadata_path": str(META_PATH),
    "clinical_cols": clinical_cols,
    "phenotype_cols": phenotype_cols,
    "output": str(OUT),
}
(OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))

print("\n================================================================================")
print("DONE")
print("================================================================================")
print(df)
print("\nSaved:")
print(" ", OUT / "hyperparameter_sensitivity_results.csv")
print(" ", OUT / "hyperparameter_sensitivity_latex_table.txt")
print(" ", OUT / "run_summary.json")
print("\nLATEX TABLE:")
print(latex_text)
