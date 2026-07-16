"""Reproduce the archived final fused K=5 analysis and validation workflow.

This result-producing script rebuilds the fused similarity-profile graph and
trains its own dense VGAE before UMAP-30 and spectral clustering with K=5.
It therefore does not consume the PyTorch Geometric VGAE output produced by
08_train_vgae.py. The distinction is retained explicitly for provenance.

The script also runs 100 subsamples, 100 bootstrap analyses, 1000 label
permutations, and clinical-profile tests. Generated subject-level files must
remain local and must not be committed to a public repository.
"""

from pathlib import Path
import os
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import umap
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    pairwise_distances,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAPH_ROOT = Path(os.getenv(
    "EPGSD_GRAPH_ROOT",
    "TECHNICAL_FIX_SPECTRAL_UMAP/graphs_MONTAGE_FIXED_graphK12_alpha06",
))
EEG_FILE = GRAPH_ROOT / "EEG_ONLY" / "node_features.npy"
PHENO_FILE = GRAPH_ROOT / "PHENOTYPE_ONLY" / "node_features.npy"
META_FILE = GRAPH_ROOT / "subject_metadata_for_all_experiments.csv"

OUTPUT_ROOT = Path(os.getenv(
    "EPGSD_FINAL_OUTPUT_ROOT",
    "TECHNICAL_FIX_SPECTRAL_UMAP/FINAL_MONTAGE_FIXED_graphK12_alpha06_umap30_forcedK5",
))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GRAPH_K_GRID = [12]
ALPHA_GRID = [0.6]
UMAP_N_GRID = [30]

FORCED_K = 5
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

VGAE_EPOCHS = 500
LR = 0.001
HIDDEN_DIM = 64
LATENT_DIM = 16
DROPOUT = 0.0

# Screening values.
# After finding best setting, we can rerun top setting with 100/100/1000.
N_SUBSAMPLE = 100
N_BOOTSTRAP = 100
N_PERMUTATIONS = 1000

CLINICAL_VARS = [
    "p_factor",
    "attention",
    "internalizing",
    "externalizing",
    "ehq_total",
    "age"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# GRAPH HELPERS
# ============================================================

def build_knn_adjacency_from_similarity(similarity, k):
    adj = kneighbors_graph(
        similarity,
        n_neighbors=k,
        mode="connectivity",
        include_self=False
    ).toarray()

    adj = np.maximum(adj, adj.T)
    return adj.astype(np.float32)


def dense_to_edge_index(adj):
    src, dst = np.where(adj > 0)
    edge_index = np.vstack([src, dst])
    return torch.tensor(edge_index, dtype=torch.long)


# ============================================================
# ARCHIVED DENSE VGAE IMPLEMENTATION
# ============================================================

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        return self.lin(adj_norm @ x)


class DenseVGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, dropout=0.0):
        super().__init__()
        self.gc1 = DenseGCNLayer(input_dim, hidden_dim)
        self.gc_mu = DenseGCNLayer(hidden_dim, latent_dim)
        self.gc_logstd = DenseGCNLayer(hidden_dim, latent_dim)
        self.dropout = dropout

    def encode(self, x, adj_norm):
        h = self.gc1(x, adj_norm)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.gc_mu(h, adj_norm)
        logstd = self.gc_logstd(h, adj_norm)
        logstd = torch.clamp(logstd, min=-10, max=10)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(logstd)
        return mu

    def decode(self, z):
        return torch.sigmoid(z @ z.t())

    def forward(self, x, adj_norm):
        mu, logstd = self.encode(x, adj_norm)
        z = self.reparameterize(mu, logstd)
        recon = self.decode(z)
        return recon, mu, logstd, z


def normalize_adjacency(adj):
    A = adj.copy().astype(np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    D = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def train_vgae_dense(node_features, adjacency, out_dir):
    set_seed(SEED)

    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
    adj = adjacency.astype(np.float32)

    adj_norm_np = normalize_adjacency(adj)
    adj_norm = torch.tensor(adj_norm_np, dtype=torch.float32).to(DEVICE)

    A = adj.copy()
    A = A + np.eye(A.shape[0], dtype=np.float32)
    A[A > 0] = 1.0

    target = torch.tensor(A, dtype=torch.float32).to(DEVICE)

    n_pos = float(A.sum())
    n_total = float(A.size)
    pos_weight = torch.tensor((n_total - n_pos) / max(n_pos, 1.0), dtype=torch.float32).to(DEVICE)

    model = DenseVGAE(
        input_dim=node_features.shape[1],
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []

    for epoch in range(1, VGAE_EPOCHS + 1):
        model.train()
        opt.zero_grad()

        recon, mu, logstd, z = model(x, adj_norm)

        recon_loss = F.binary_cross_entropy(
            recon,
            target,
            weight=target * pos_weight + (1 - target),
            reduction="mean"
        )

        kl = -0.5 / x.shape[0] * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        )

        loss = recon_loss + kl
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | recon={recon_loss.item():.6f} "
                f"kl={kl.item():.6f} total={loss.item():.6f}"
            )

        logs.append({
            "epoch": epoch,
            "recon_loss": float(recon_loss.item()),
            "kl_loss": float(kl.item()),
            "total_loss": float(loss.item())
        })

    model.eval()
    with torch.no_grad():
        mu, logstd = model.encode(x, adj_norm)
        z = mu.detach().cpu().numpy()
        reconstructed = torch.sigmoid(mu @ mu.t()).detach().cpu().numpy()

    np.save(out_dir / "latent_embeddings.npy", z)
    pd.DataFrame(z).to_csv(out_dir / "latent_embeddings.csv", index=False)
    np.save(out_dir / "reconstructed_graph.npy", reconstructed)
    pd.DataFrame(logs).to_csv(out_dir / "vgae_training_log.csv", index=False)

    return z


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_metric(func, X, labels):
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(func(X, labels))
    except Exception:
        return np.nan


def dunn_index(X, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)

    intra = []
    for c in unique:
        pts = X[labels == c]
        if len(pts) > 1:
            intra.append(np.max(pairwise_distances(pts)))

    if len(intra) == 0:
        return np.nan

    max_intra = np.max(intra)
    if max_intra == 0:
        return np.nan

    inter = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            a = X[labels == unique[i]]
            b = X[labels == unique[j]]
            if len(a) > 0 and len(b) > 0:
                inter.append(np.min(cdist(a, b)))

    if len(inter) == 0:
        return np.nan

    return float(np.min(inter) / max_intra)


def run_umap_projection(X, seed, n_neighbors):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=seed
    )
    return reducer.fit_transform(X)


def run_spectral(X2, k, seed, n_neighbors):
    n_neighbors = min(n_neighbors, len(X2) - 1)
    return SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=seed
    ).fit_predict(X2)


def cluster_quality(X2, labels, graph_k, alpha, umap_n):
    labels = np.asarray(labels)
    sizes = pd.Series(labels).value_counts().sort_index()

    return {
        "graph_k": graph_k,
        "alpha": alpha,
        "umap_n_neighbors": umap_n,
        "forced_k": FORCED_K,
        "silhouette": safe_metric(silhouette_score, X2, labels),
        "davies_bouldin": safe_metric(davies_bouldin_score, X2, labels),
        "calinski_harabasz": safe_metric(calinski_harabasz_score, X2, labels),
        "dunn_index": dunn_index(X2, labels),
        "n_clusters_found": int(len(np.unique(labels))),
        "n_subjects": int(len(labels)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_size_ratio_max_min": float(sizes.max() / sizes.min()) if sizes.min() > 0 else np.inf
    }


def pairwise_jaccard_same_cluster(labels_a, labels_b):
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    n = len(labels_a)
    A = labels_a[:, None] == labels_a[None, :]
    B = labels_b[:, None] == labels_b[None, :]

    iu = np.triu_indices(n, 1)
    A = A[iu]
    B = B[iu]

    inter = np.sum(A & B)
    union = np.sum(A | B)

    return float(inter / union) if union > 0 else np.nan


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "median": float(np.nanmedian(values)),
        "ci95_low": float(np.nanpercentile(values, 2.5)),
        "ci95_high": float(np.nanpercentile(values, 97.5))
    }


# ============================================================
# STABILITY / PERMUTATION / CLINICAL
# ============================================================

def subsample_stability(X_latent, ref_labels, umap_n, graph_k):
    rows = []
    n = len(ref_labels)

    for seed in range(N_SUBSAMPLE):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, int(0.8 * n), replace=False)

        X_sub = X_latent[idx]
        ref_sub = ref_labels[idx]

        X2_sub = run_umap_projection(
            X_sub,
            seed=seed,
            n_neighbors=min(umap_n, len(idx) - 1)
        )

        pred = run_spectral(
            X2_sub,
            k=FORCED_K,
            seed=seed,
            n_neighbors=min(graph_k, len(idx) - 1)
        )

        rows.append({
            "run": seed,
            "ARI": adjusted_rand_score(ref_sub, pred),
            "NMI": normalized_mutual_info_score(ref_sub, pred),
            "V_measure": v_measure_score(ref_sub, pred),
            "Fowlkes_Mallows": fowlkes_mallows_score(ref_sub, pred),
            "pairwise_jaccard": pairwise_jaccard_same_cluster(ref_sub, pred),
            "silhouette": safe_metric(silhouette_score, X2_sub, pred)
        })

    return pd.DataFrame(rows)


def bootstrap_stability(X_latent, ref_labels, umap_n, graph_k):
    rows = []
    n = len(ref_labels)

    for seed in range(N_BOOTSTRAP):
        rng = np.random.default_rng(1000 + seed)
        sample_idx = rng.choice(n, n, replace=True)
        unique_idx = np.unique(sample_idx)

        if len(unique_idx) <= FORCED_K + 2:
            continue

        X_sub = X_latent[unique_idx]
        ref_sub = ref_labels[unique_idx]

        X2_sub = run_umap_projection(
            X_sub,
            seed=seed,
            n_neighbors=min(umap_n, len(unique_idx) - 1)
        )

        pred = run_spectral(
            X2_sub,
            k=FORCED_K,
            seed=seed,
            n_neighbors=min(graph_k, len(unique_idx) - 1)
        )

        rows.append({
            "run": seed,
            "n_unique": len(unique_idx),
            "ARI": adjusted_rand_score(ref_sub, pred),
            "NMI": normalized_mutual_info_score(ref_sub, pred),
            "V_measure": v_measure_score(ref_sub, pred),
            "Fowlkes_Mallows": fowlkes_mallows_score(ref_sub, pred),
            "pairwise_jaccard": pairwise_jaccard_same_cluster(ref_sub, pred),
            "silhouette": safe_metric(silhouette_score, X2_sub, pred)
        })

    return pd.DataFrame(rows)


def permutation_test(X2, labels, real_quality):
    rows = []
    rng = np.random.default_rng(SEED)

    metric_funcs = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }

    for metric_name, func in metric_funcs.items():
        null = []

        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(labels)
            null.append(safe_metric(func, X2, perm))

        null = np.asarray(null, dtype=float)
        real = real_quality[metric_name]

        if metric_name == "davies_bouldin":
            p = (np.sum(null <= real) + 1) / (len(null) + 1)
        else:
            p = (np.sum(null >= real) + 1) / (len(null) + 1)

        z = (real - np.nanmean(null)) / np.nanstd(null)

        rows.append({
            "metric": metric_name,
            "real_value": real,
            "null_mean": float(np.nanmean(null)),
            "null_std": float(np.nanstd(null)),
            "z_score": float(z),
            "permutation_p": float(p)
        })

    return pd.DataFrame(rows)


def eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    grand = np.nanmean(values)
    ss_total = np.nansum((values - grand) ** 2)

    if ss_total == 0:
        return np.nan

    ss_between = 0.0
    for c in np.unique(labels):
        g = values[labels == c]
        ss_between += len(g) * (np.nanmean(g) - grand) ** 2

    return float(ss_between / ss_total)


def omega_squared(groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    all_values = np.concatenate(groups)

    grand = np.nanmean(all_values)
    k = len(groups)
    n = len(all_values)

    ss_between = sum(len(g) * (np.nanmean(g) - grand) ** 2 for g in groups)
    ss_within = sum(np.nansum((g - np.nanmean(g)) ** 2) for g in groups)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n - k

    if df_within <= 0:
        return np.nan

    ms_within = ss_within / df_within
    val = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    return float(max(0.0, val)) if np.isfinite(val) else np.nan


def clinical_stats(meta, labels):
    df = meta.copy()
    df["subtype"] = labels

    rows = []

    for var in CLINICAL_VARS:
        if var not in df.columns:
            continue

        df[var] = pd.to_numeric(df[var], errors="coerce")
        df[var] = df[var].fillna(df[var].median())

        groups = [g[var].values for _, g in df.groupby("subtype")]

        try:
            Fv, p_anova = f_oneway(*groups)
        except Exception:
            Fv, p_anova = np.nan, np.nan

        try:
            H, p_kw = kruskal(*groups)
        except Exception:
            H, p_kw = np.nan, np.nan

        rows.append({
            "variable": var,
            "anova_F": Fv,
            "anova_p": p_anova,
            "kruskal_H": H,
            "kruskal_p": p_kw,
            "eta_squared": eta_squared(df[var].values, labels),
            "omega_squared": omega_squared(groups)
        })

    stats = pd.DataFrame(rows)

    if len(stats) > 0:
        stats["anova_p_fdr_bh"] = multipletests(
            stats["anova_p"].fillna(1.0),
            method="fdr_bh"
        )[1]

        stats["kruskal_p_fdr_bh"] = multipletests(
            stats["kruskal_p"].fillna(1.0),
            method="fdr_bh"
        )[1]

    return stats


# ============================================================
# ONE GRID RUN
# ============================================================

def run_one_setting(X_eeg, X_pheno, meta, graph_k, alpha):
    setting_name = f"graphK{graph_k}_alpha{alpha:.1f}".replace(".", "p")
    setting_dir = OUTPUT_ROOT / setting_name
    setting_dir.mkdir(parents=True, exist_ok=True)

    print("\n================================================")
    print("SETTING:", setting_name)
    print("================================================")

    eeg_sim = cosine_similarity(X_eeg)
    pheno_sim = cosine_similarity(X_pheno)
    fused_sim = alpha * eeg_sim + (1 - alpha) * pheno_sim

    adjacency = build_knn_adjacency_from_similarity(fused_sim, graph_k)
    node_features = np.concatenate([X_eeg, X_pheno], axis=1).astype(np.float32)

    np.save(setting_dir / "adjacency.npy", adjacency)
    np.save(setting_dir / "node_features.npy", node_features)
    np.save(setting_dir / "fused_similarity.npy", fused_sim.astype(np.float32))

    vgae_dir = setting_dir / "vgae"
    X_latent = train_vgae_dense(node_features, adjacency, vgae_dir)

    all_rows = []

    for umap_n in UMAP_N_GRID:
        print(f"\nUMAP n_neighbors={umap_n}")

        out_dir = setting_dir / f"UMAP{umap_n}_K5"
        out_dir.mkdir(parents=True, exist_ok=True)

        X2 = run_umap_projection(
            X_latent,
            seed=SEED,
            n_neighbors=umap_n
        )

        labels = run_spectral(
            X2,
            k=FORCED_K,
            seed=SEED,
            n_neighbors=graph_k
        )

        quality = cluster_quality(
            X2,
            labels,
            graph_k=graph_k,
            alpha=alpha,
            umap_n=umap_n
        )

        assign = meta.copy()
        assign["subtype"] = labels
        assign["UMAP1"] = X2[:, 0]
        assign["UMAP2"] = X2[:, 1]

        cluster_sizes = (
            assign["subtype"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        cluster_sizes.columns = ["subtype", "n_subjects"]

        subsample = subsample_stability(X_latent, labels, umap_n, graph_k)
        bootstrap = bootstrap_stability(X_latent, labels, umap_n, graph_k)

        stability_rows = []
        for metric in ["ARI", "NMI", "V_measure", "Fowlkes_Mallows", "pairwise_jaccard", "silhouette"]:
            stability_rows.append({
                "analysis": "80_percent_subsample",
                "metric": metric,
                **summarize_values(subsample[metric])
            })
            stability_rows.append({
                "analysis": "bootstrap_unique_subjects",
                "metric": metric,
                **summarize_values(bootstrap[metric])
            })

        stability_summary = pd.DataFrame(stability_rows)
        permutation = permutation_test(X2, labels, quality)
        clinical = clinical_stats(meta, labels)

        pd.DataFrame([quality]).to_csv(out_dir / "main_quality.csv", index=False)
        cluster_sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)
        assign.to_csv(out_dir / "subject_subtypes.csv", index=False)
        pd.DataFrame(X2, columns=["UMAP1", "UMAP2"]).to_csv(out_dir / "umap_coordinates.csv", index=False)
        pd.DataFrame(X_latent).to_csv(out_dir / "latent_embeddings_used.csv", index=False)
        subsample.to_csv(out_dir / "subsample_runs.csv", index=False)
        bootstrap.to_csv(out_dir / "bootstrap_runs.csv", index=False)
        stability_summary.to_csv(out_dir / "stability_summary.csv", index=False)
        permutation.to_csv(out_dir / "permutation_test.csv", index=False)
        clinical.to_csv(out_dir / "clinical_stats.csv", index=False)

        row = dict(quality)

        for analysis_name, df_stab in [
            ("subsample", subsample),
            ("bootstrap", bootstrap)
        ]:
            for metric in ["ARI", "NMI", "pairwise_jaccard", "silhouette"]:
                row[f"{analysis_name}_{metric}_mean"] = float(df_stab[metric].mean())
                row[f"{analysis_name}_{metric}_std"] = float(df_stab[metric].std())

        for _, r in clinical.iterrows():
            var = r["variable"]
            row[f"{var}_anova_p_fdr"] = r["anova_p_fdr_bh"]
            row[f"{var}_eta_squared"] = r["eta_squared"]

        row["output_dir"] = str(out_dir)
        all_rows.append(row)

        print(
            f"Result | graph_k={graph_k} alpha={alpha} umap={umap_n} "
            f"sil={quality['silhouette']:.4f} "
            f"DB={quality['davies_bouldin']:.4f} "
            f"CH={quality['calinski_harabasz']:.2f} "
            f"min_cluster={quality['min_cluster_size']} "
            f"sub_ARI={row['subsample_ARI_mean']:.4f} "
            f"boot_ARI={row['bootstrap_ARI_mean']:.4f}"
        )

    return all_rows


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("\n====================================================")
    print("FINAL FORCED K=5 ANALYSIS")
    print("====================================================")
    print("Device:", DEVICE)
    print("EEG file:", EEG_FILE)
    print("Phenotype file:", PHENO_FILE)
    print("Metadata:", META_FILE)
    print("Output:", OUTPUT_ROOT)
    print("Graph K grid:", GRAPH_K_GRID)
    print("Alpha grid:", ALPHA_GRID)
    print("UMAP n grid:", UMAP_N_GRID)
    print("Forced K:", FORCED_K)

    if not EEG_FILE.exists():
        raise FileNotFoundError(EEG_FILE)
    if not PHENO_FILE.exists():
        raise FileNotFoundError(PHENO_FILE)
    if not META_FILE.exists():
        raise FileNotFoundError(META_FILE)

    X_eeg = np.load(EEG_FILE).astype(np.float32)
    X_pheno = np.load(PHENO_FILE).astype(np.float32)
    meta = pd.read_csv(META_FILE)

    print("\nLoaded:")
    print("X_eeg:", X_eeg.shape)
    print("X_pheno:", X_pheno.shape)
    print("meta:", meta.shape)

    if len(X_eeg) != len(X_pheno) or len(X_eeg) != len(meta):
        raise ValueError("Row mismatch between EEG, phenotype, and metadata.")

    X_eeg = StandardScaler().fit_transform(X_eeg).astype(np.float32)
    X_pheno = StandardScaler().fit_transform(X_pheno).astype(np.float32)

    master_rows = []

    for graph_k in GRAPH_K_GRID:
        for alpha in ALPHA_GRID:
            rows = run_one_setting(
                X_eeg=X_eeg,
                X_pheno=X_pheno,
                meta=meta,
                graph_k=graph_k,
                alpha=alpha
            )
            master_rows.extend(rows)

            master_df = pd.DataFrame(master_rows)
            master_df.to_csv(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS_RUNNING.csv", index=False)

    master = pd.DataFrame(master_rows)

    # Selection target:
    # Archived quality thresholds used for result ranking.
    master["passes_archived_quality_threshold"] = (
        (master["silhouette"] > 0.4810) &
        (master["davies_bouldin"] < 0.6490) &
        (master["calinski_harabasz"] > 596.0) &
        (master["min_cluster_size"] > 40)
    )

    master_sorted = master.sort_values(
        [
            "passes_archived_quality_threshold",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "min_cluster_size",
            "subsample_ARI_mean",
            "bootstrap_ARI_mean"
        ],
        ascending=[False, False, True, False, False, False, False]
    )

    master.to_csv(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS.csv", index=False)
    master_sorted.to_csv(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS_SORTED.csv", index=False)

    with pd.ExcelWriter(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS.xlsx") as writer:
        master.to_excel(writer, sheet_name="AllResults", index=False)
        master_sorted.to_excel(writer, sheet_name="SortedResults", index=False)

    print("\n====================================================")
    print("DONE HYPERPARAMETER TUNING")
    print("====================================================")
    print("Saved:")
    print(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS.csv")
    print(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS_SORTED.csv")
    print(OUTPUT_ROOT / "MASTER_HYPER_TUNE_RESULTS.xlsx")

    print("\nTOP 10 RESULTS:")
    show_cols = [
        "graph_k",
        "alpha",
        "umap_n_neighbors",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "dunn_index",
        "min_cluster_size",
        "max_cluster_size",
        "cluster_size_ratio_max_min",
        "subsample_ARI_mean",
        "subsample_NMI_mean",
        "bootstrap_ARI_mean",
        "bootstrap_NMI_mean",
        "passes_archived_quality_threshold",
        "output_dir"
    ]
    print(master_sorted[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
