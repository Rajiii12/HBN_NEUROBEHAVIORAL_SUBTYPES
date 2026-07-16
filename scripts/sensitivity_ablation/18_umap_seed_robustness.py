#!/usr/bin/env python3
"""
UMAP seed-robustness analysis for the final saved VGAE representation.

This script repeats:
VGAE embedding -> UMAP with different random_state -> spectral clustering K=5

It reports internal metrics, cluster balance, ARI/NMI versus final labels,
and clinical FDR count for every seed.
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

import umap


# ============================================================
# Settings
# ============================================================
ROOT = Path(
    os.environ.get("EPGSD_ANALYSIS_ROOT", "TECHNICAL_FIX_SPECTRAL_UMAP")
)

FINAL_DIR = ROOT / "FINAL_best_graphK12_alpha06_umap30_forcedK5" / "graphK12_alpha0p6"
VGAE_EMB = Path(
    os.environ.get(
        "EPGSD_VGAE_EMBEDDING",
        str(
            ROOT
            / "hyper_tune_old_best_forced_k5_no_overwrite"
            / "graphK12_alpha0p6"
            / "vgae"
            / "latent_embeddings.npy"
        ),
    )
)
FINAL_LABELS = Path(
    os.environ.get(
        "EPGSD_FINAL_LABELS",
        str(FINAL_DIR / "UMAP30_K5" / "subject_subtypes.csv"),
    )
)
META_FILE = Path(
    os.environ.get(
        "EPGSD_SUBJECT_METADATA",
        str(ROOT / "subject_embeddings_norm_fixed" / "subject_metadata_norm_fixed.csv"),
    )
)

OUT = Path(
    os.environ.get(
        "EPGSD_SENSITIVITY_OUT",
        str(ROOT / "umap_seed_robustness"),
    )
)
OUT.mkdir(parents=True, exist_ok=True)

K = 5
UMAP_NEIGHBORS = 30
UMAP_COMPONENTS = 2
SEEDS = [0, 1, 2, 3, 4, 5, 10, 11, 21, 42, 77, 100, 123, 202, 999, 1001, 2024, 2026, 3001, 5000]


# ============================================================
# Helper functions
# ============================================================
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

    _, p = stats.f_oneway(*groups)
    gm = np.nanmean(y)
    sst = np.nansum((y - gm) ** 2)
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    eta2 = ssb / sst if sst > 0 else 0.0

    return float(p), float(eta2)


def clinical_stats(meta, labels):
    vars_cols = {
        "p_factor": "p_factor",
        "attention": "attention",
        "internalizing": "internalizing",
        "externalizing": "externalizing",
        "age": "age",
        "ehq_total": "ehq_total",
    }

    rows = []
    pvals = []

    for var, col in vars_cols.items():
        y = pd.to_numeric(meta[col], errors="coerce").values
        p, e2 = eta_squared_anova(y, labels)
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


def load_final_labels(path):
    df = pd.read_csv(path)
    if "subtype" in df.columns:
        return df["subtype"].values
    numeric = df.select_dtypes(include=[np.number])
    return numeric.iloc[:, -1].values


def find_vgae_embedding():
    candidates = [
        VGAE_EMB,
        FINAL_DIR / "vgae_embeddings.npy",
        FINAL_DIR / "embeddings.npy",
        FINAL_DIR / "latent_embeddings.npy",
        ROOT / "vgae_three_experiments_logged" / "EEG_PHENOTYPE" / "latent_embeddings.npy",
        ROOT / "vgae_three_experiments_logged" / "EEG_PHENOTYPE" / "vgae_embeddings.npy",
        ROOT / "vgae_three_experiments_logged" / "EEG_PHENOTYPE" / "embeddings.npy",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(ROOT.rglob("*embedding*.npy"))
    print("Could not find expected VGAE embedding. Candidate embedding files:")
    for p in found[:50]:
        print(" ", p)

    raise FileNotFoundError("Could not find VGAE embedding file. Please check final folder.")


# ============================================================
# Load data
# ============================================================
print("=" * 90)
print("UMAP seed robustness")
print("=" * 90)

emb_path = find_vgae_embedding()

print("VGAE embedding:", emb_path)
print("FINAL_LABELS:", FINAL_LABELS)
print("META_FILE:", META_FILE)
print("OUT:", OUT)

Z = np.load(emb_path)
Z = np.asarray(Z, dtype=float)
Z = StandardScaler().fit_transform(np.nan_to_num(Z))

meta = pd.read_csv(META_FILE)
final_labels = load_final_labels(FINAL_LABELS)

print("Z shape:", Z.shape)
print("final labels:", final_labels.shape)
print("Seeds:", SEEDS)


# ============================================================
# Run seed robustness
# ============================================================
all_rows = []

for seed in SEEDS:
    print("\nRunning seed:", seed)

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed
    )

    Y = reducer.fit_transform(Z)
    Y_eval = StandardScaler().fit_transform(Y)

    labels = SpectralClustering(
        n_clusters=K,
        affinity="nearest_neighbors",
        n_neighbors=min(12, len(Y_eval) - 1),
        assign_labels="kmeans",
        random_state=seed
    ).fit_predict(Y_eval)

    counts = np.bincount(labels, minlength=K)

    clin = clinical_stats(meta, labels)
    fdr_count = int(clin["significant_fdr_0p05"].sum())
    mean_eta = float(clin["eta_squared"].mean())

    row = {
        "seed": seed,
        "silhouette": float(silhouette_score(Y_eval, labels)),
        "davies_bouldin": float(davies_bouldin_score(Y_eval, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Y_eval, labels)),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "cluster_size_ratio_max_min": float(counts.max() / max(counts.min(), 1)),
        "mean_eta_squared": mean_eta,
        "fdr_sig_count": f"{fdr_count}/6",
        "fdr_sig_count_num": fdr_count,
        "ARI_vs_final": float(adjusted_rand_score(final_labels, labels)),
        "NMI_vs_final": float(normalized_mutual_info_score(final_labels, labels))
    }

    all_rows.append(row)

    pd.DataFrame({
        "subject_index": np.arange(len(labels)),
        "subtype": labels
    }).to_csv(OUT / f"labels_seed_{seed}.csv", index=False)

    pd.DataFrame(Y, columns=["UMAP1", "UMAP2"]).to_csv(OUT / f"umap_seed_{seed}.csv", index=False)
    clin.to_csv(OUT / f"clinical_seed_{seed}.csv", index=False)

    print(row)


df = pd.DataFrame(all_rows)
df.to_csv(OUT / "umap_seed_robustness_all_runs.csv", index=False)


# ============================================================
# Summary statistics
# ============================================================
metric_cols = [
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "min_cluster_size",
    "cluster_size_ratio_max_min",
    "mean_eta_squared",
    "fdr_sig_count_num",
    "ARI_vs_final",
    "NMI_vs_final"
]

summary_rows = []
for col in metric_cols:
    vals = df[col].astype(float).values
    summary_rows.append({
        "metric": col,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals))
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "umap_seed_robustness_summary.csv", index=False)

print("\n" + "=" * 90)
print("ALL RUNS")
print("=" * 90)
print(df.to_string(index=False))

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))


# ============================================================
# LaTeX table
# ============================================================
latex = r"""
\begin{table}[!t]
	\caption{Robustness of the EPG-SD subtype solution to UMAP random initialization. UMAP and spectral clustering were repeated across 20 random seeds while keeping graph neighborhood size, fusion weight, UMAP neighborhood size, and subtype number fixed.}
	\label{tab:umap_seed_robustness}
	\centering
	\footnotesize
	\setlength{\tabcolsep}{4pt}
	\begin{tabular}{lcccc}
		\toprule
		Metric & Mean & SD & Min & Max \\
		\midrule
"""

name_map = {
    "silhouette": "Silhouette",
    "davies_bouldin": "DB",
    "calinski_harabasz": "CH",
    "min_cluster_size": "Min. size",
    "cluster_size_ratio_max_min": "Balance",
    "mean_eta_squared": "Mean $\\eta^2$",
    "fdr_sig_count_num": "FDR count",
    "ARI_vs_final": "ARI vs. final",
    "NMI_vs_final": "NMI vs. final"
}

for _, r in summary.iterrows():
    name = name_map.get(r["metric"], r["metric"])
    if r["metric"] in ["calinski_harabasz"]:
        latex += f"\t\t{name} & {r['mean']:.1f} & {r['std']:.1f} & {r['min']:.1f} & {r['max']:.1f} \\\\\n"
    elif r["metric"] in ["min_cluster_size", "fdr_sig_count_num"]:
        latex += f"\t\t{name} & {r['mean']:.1f} & {r['std']:.1f} & {r['min']:.0f} & {r['max']:.0f} \\\\\n"
    else:
        latex += f"\t\t{name} & {r['mean']:.3f} & {r['std']:.3f} & {r['min']:.3f} & {r['max']:.3f} \\\\\n"

latex += r"""		\bottomrule
	\end{tabular}
\end{table}
"""

(OUT / "umap_seed_robustness_latex.txt").write_text(latex)

print("\n" + "=" * 90)
print("LATEX TABLE")
print("=" * 90)
print(latex)

print("\nSaved files:")
for p in sorted(OUT.iterdir()):
    print(" ", p)

print("\nDONE.")
