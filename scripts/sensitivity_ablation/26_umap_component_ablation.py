# Public reproducibility version.
# Generated participant-level labels and embeddings must remain private.
# Only participant-free aggregate summaries should be committed.

#!/usr/bin/env python3
"""
Reviewer Comment 5:
Why apply Spectral Clustering after UMAP instead of directly on VGAE embeddings?

This script compares:
1. VGAE + Spectral
2. VGAE + UMAP + Spectral

using the final paper configuration:
graphK12, alpha=0.6, UMAP30, K=5.
"""

from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# ============================================================
# Paths
# ============================================================
ROOT = Path(os.environ.get(
    "EPGSD_ANALYSIS_ROOT",
    "TECHNICAL_FIX_SPECTRAL_UMAP",
))

FINAL_ROOT = ROOT / "FINAL_best_graphK12_alpha06_umap30_forcedK5"
GRAPH_DIR = FINAL_ROOT / "graphK12_alpha0p6"
UMAP_DIR = GRAPH_DIR / "UMAP30_K5"

VGAE_CSV = GRAPH_DIR / "vgae" / "latent_embeddings.csv"
VGAE_NPY = GRAPH_DIR / "vgae" / "latent_embeddings.npy"
UMAP_CSV = UMAP_DIR / "umap_coordinates.csv"
FINAL_LABELS = UMAP_DIR / "subject_subtypes.csv"

META_FILE = ROOT / "subject_embeddings_norm_fixed" / "subject_metadata_norm_fixed.csv"

OUT = Path(os.environ.get(
    "EPGSD_T26_OUT",
    "results/sensitivity_ablation/table26_umap_ablation",
))
OUT.mkdir(parents=True, exist_ok=True)

K = 5
NN = 12
SEED = 42

print("=" * 90)
print("Reviewer Comment 5: VGAE vs VGAE+UMAP Spectral Ablation")
print("=" * 90)
print("VGAE_CSV:", VGAE_CSV)
print("VGAE_NPY:", VGAE_NPY)
print("UMAP_CSV:", UMAP_CSV)
print("FINAL_LABELS:", FINAL_LABELS)
print("META_FILE:", META_FILE)
print("OUT:", OUT)


# ============================================================
# Helper functions
# ============================================================
def load_matrix(csv_path=None, npy_path=None):
    if npy_path is not None and npy_path.exists():
        X = np.load(npy_path)
        return np.asarray(X, dtype=float)

    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        num = df.select_dtypes(include=[np.number])
        return num.values.astype(float)

    raise FileNotFoundError(f"Could not find {csv_path} or {npy_path}")


def spectral_labels(X, k=5, nn=12):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    nn = min(nn, len(Xs) - 1)

    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=nn,
        assign_labels="kmeans",
        random_state=SEED
    )
    labels = model.fit_predict(Xs)
    return Xs, labels


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = pvals[order[i]] * n / rank
        prev = min(prev, val)
        adj[order[i]] = min(prev, 1.0)
    return adj


def eta_squared(y, labels):
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

    _, p = stats.f_oneway(*groups)
    gm = np.nanmean(y)
    sst = np.nansum((y - gm) ** 2)
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    eta2 = ssb / sst if sst > 0 else 0.0
    return float(p), float(eta2)


def clinical_stats(meta, labels):
    cols = {
        "p_factor": "p_factor",
        "attention": "attention",
        "internalizing": "internalizing",
        "externalizing": "externalizing",
        "age": "age",
        "ehq_total": "ehq_total",
    }

    rows = []
    pvals = []

    for var, col in cols.items():
        y = pd.to_numeric(meta[col], errors="coerce").values
        p, e2 = eta_squared(y, labels)
        rows.append({
            "variable": var,
            "anova_p": p,
            "eta_squared": e2
        })
        pvals.append(p)

    fdr = bh_fdr(pvals)
    for i in range(len(rows)):
        rows[i]["anova_p_fdr"] = float(fdr[i])
        rows[i]["significant_fdr_0p05"] = bool(fdr[i] < 0.05)

    return pd.DataFrame(rows)


def summarize_method(name, X, labels, meta, final_labels=None):
    counts = np.bincount(labels, minlength=K)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    min_cluster = int(counts.min())
    max_cluster = int(counts.max())
    balance = float(max_cluster / max(min_cluster, 1))

    clin = clinical_stats(meta, labels)
    clin.to_csv(OUT / f"{name}_clinical_stats.csv", index=False)

    row = {
        "method": name,
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "min_cluster_size": min_cluster,
        "max_cluster_size": max_cluster,
        "cluster_size_ratio_max_min": balance,
        "mean_eta_squared": float(clin["eta_squared"].mean()),
        "fdr_sig_count": int(clin["significant_fdr_0p05"].sum()),
        "age_fdr_p": float(clin.loc[clin["variable"] == "age", "anova_p_fdr"].values[0]),
        "age_eta_squared": float(clin.loc[clin["variable"] == "age", "eta_squared"].values[0]),
        "ehq_fdr_p": float(clin.loc[clin["variable"] == "ehq_total", "anova_p_fdr"].values[0]),
        "ehq_eta_squared": float(clin.loc[clin["variable"] == "ehq_total", "eta_squared"].values[0]),
    }

    if final_labels is not None:
        row["ARI_vs_final_EPGSD"] = float(adjusted_rand_score(final_labels, labels))
        row["NMI_vs_final_EPGSD"] = float(normalized_mutual_info_score(final_labels, labels))

    pd.DataFrame({
        "subject_index": np.arange(len(labels)),
        "subtype": labels
    }).to_csv(OUT / f"{name}_labels.csv", index=False)

    pd.DataFrame({
        "cluster": np.arange(K),
        "size": counts
    }).to_csv(OUT / f"{name}_cluster_sizes.csv", index=False)

    return row


# ============================================================
# Load data
# ============================================================
X_vgae = load_matrix(csv_path=VGAE_CSV, npy_path=VGAE_NPY)
X_umap = load_matrix(csv_path=UMAP_CSV, npy_path=None)

meta = pd.read_csv(META_FILE)

if FINAL_LABELS.exists():
    final_df = pd.read_csv(FINAL_LABELS)
    if "subtype" in final_df.columns:
        final_labels = final_df["subtype"].values
    else:
        final_labels = final_df.select_dtypes(include=[np.number]).iloc[:, -1].values
else:
    final_labels = None

print("\nLoaded:")
print("X_vgae:", X_vgae.shape)
print("X_umap:", X_umap.shape)
print("meta:", meta.shape)
print("final_labels:", None if final_labels is None else final_labels.shape)


# ============================================================
# Run ablations
# ============================================================
Xv, lab_vgae = spectral_labels(X_vgae, k=K, nn=NN)
Xu, lab_umap = spectral_labels(X_umap, k=K, nn=NN)

rows = []
rows.append(summarize_method(
    "VGAE_Spectral_direct",
    Xv,
    lab_vgae,
    meta,
    final_labels=final_labels
))

rows.append(summarize_method(
    "VGAE_UMAP_Spectral_final",
    Xu,
    lab_umap,
    meta,
    final_labels=final_labels
))

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "comment5_umap_ablation_summary.csv", index=False)

print("\n" + "=" * 90)
print("COMMENT 5 UMAP ABLATION SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))


# ============================================================
# LaTeX table
# ============================================================
latex = r"""
\begin{table}[!t]
	\caption{Ablation of the UMAP projection stage. Spectral clustering was applied either directly to the VGAE embeddings or after UMAP projection.}
	\label{tab:umap_ablation}
	\centering
	\footnotesize
	\setlength{\tabcolsep}{3.5pt}
	\begin{tabular}{lcccccc}
		\toprule
		Method & Sil. $\uparrow$ & DB $\downarrow$ & CH $\uparrow$ & Min. $\uparrow$ & Bal. $\downarrow$ & FDR \\
		\midrule
"""

for _, r in summary.iterrows():
    method = r["method"].replace("_", r"\_")
    latex += (
        f"\t\t{method} & "
        f"{r['silhouette']:.3f} & "
        f"{r['davies_bouldin']:.3f} & "
        f"{r['calinski_harabasz']:.1f} & "
        f"{int(r['min_cluster_size'])} & "
        f"{r['cluster_size_ratio_max_min']:.2f} & "
        f"{int(r['fdr_sig_count'])}/6 \\\\\n"
    )

latex += r"""		\bottomrule
	\end{tabular}
\end{table}
"""

(OUT / "comment5_umap_ablation_latex.txt").write_text(latex)

print("\n" + "=" * 90)
print("LATEX TABLE")
print("=" * 90)
print(latex)

print("\nSaved files:")
for p in sorted(OUT.iterdir()):
    print(" ", p)

print("\nDONE.")
