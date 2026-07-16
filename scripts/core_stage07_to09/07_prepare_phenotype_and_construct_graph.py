"""Prepare phenotype features and construct the three subject graphs.

The archived manuscript configuration uses graph neighborhood k=12 and
EEG-phenotype fusion weight alpha=0.6. Phenotype preparation is included in
this stage rather than implemented as a separate script.

The adjacency is constructed by applying Euclidean k-nearest neighbors to
rows of the cosine-similarity matrix (similarity-profile kNN). This preserves
the archived result-producing implementation and is not direct top-k
selection from each similarity row.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

SUBJECT_EMBED_CANDIDATES = [
    os.getenv("EPGSD_SUBJECT_EMBED_FILE", ""),
    "./TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_MONTAGE_FIXED/subject_embedding_mean_std_median_norm_fixed.npy",
    "./TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_fixed/subject_embedding_mean_std_fixed.npy",
    "./subject_embeddings/subject_embedding.npy",
]

META_CANDIDATES = [
    os.getenv("EPGSD_SUBJECT_META_FILE", ""),
    "./TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_MONTAGE_FIXED/subject_metadata_norm_fixed.csv",
    "./TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_fixed/subject_metadata_fixed.csv",
    "./subject_embeddings/subject_metadata.csv",
]
OUTPUT_ROOT = Path(os.getenv(
    "EPGSD_GRAPH_OUTPUT_ROOT",
    "./TECHNICAL_FIX_SPECTRAL_UMAP/graphs_MONTAGE_FIXED_graphK12_alpha06",
))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

KNN_K = 12
ALPHA = 0.6

# Column order is retained from the archived result-producing script.
BEHAVIOR_COLS = ["internalizing", "attention", "externalizing", "p_factor"]

def find_existing(cands):
    for p in cands:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("\n".join(cands))

def build_knn_adjacency(similarity, k=10):
    # Archived implementation: kNN is applied to similarity-profile rows.
    adj = kneighbors_graph(similarity, n_neighbors=k, mode="connectivity", include_self=False)
    adj = adj.toarray()
    adj = np.maximum(adj, adj.T)
    return adj.astype(np.float32)

def save_experiment(name, node_features, adjacency, extra_dict=None):
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "node_features.npy", node_features.astype(np.float32))
    np.save(out / "adjacency.npy", adjacency.astype(np.float32))
    if extra_dict:
        for k, v in extra_dict.items():
            np.save(out / f"{k}.npy", v.astype(np.float32))
    pd.DataFrame({
        "metric": ["n_subjects", "feature_dim", "knn_k", "n_edges"],
        "value": [node_features.shape[0], node_features.shape[1], KNN_K, int(adjacency.sum())]
    }).to_csv(out / "graph_summary.csv", index=False)
    print(name, "node_features", node_features.shape, "adjacency", adjacency.shape, "edges", int(adjacency.sum()))

def main():
    print("\n===================================")
    print("THREE-EXPERIMENT GRAPH CONSTRUCTION")
    print("===================================")

    eeg_file = find_existing(SUBJECT_EMBED_CANDIDATES)
    meta_file = find_existing(META_CANDIDATES)

    X_eeg = np.load(eeg_file)
    meta = pd.read_csv(meta_file)

    print("EEG file:", eeg_file, X_eeg.shape)
    print("Meta file:", meta_file, meta.shape)

    if len(X_eeg) != len(meta):
        raise ValueError("EEG subject features and metadata mismatch.")

    X_eeg = StandardScaler().fit_transform(X_eeg)

    missing = [c for c in BEHAVIOR_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing behavior columns: {missing}")

    Xb = meta[BEHAVIOR_COLS].copy()
    for c in BEHAVIOR_COLS:
        Xb[c] = Xb[c].fillna(Xb[c].median())
    X_behavior = StandardScaler().fit_transform(Xb.values)

    eeg_sim = cosine_similarity(X_eeg)
    beh_sim = cosine_similarity(X_behavior)
    fused_sim = ALPHA * eeg_sim + (1 - ALPHA) * beh_sim

    # EEG-only graph.
    adj_eeg = build_knn_adjacency(eeg_sim, KNN_K)
    save_experiment("EEG_ONLY", X_eeg, adj_eeg, {"eeg_similarity": eeg_sim})

    # Phenotype-only graph.
    adj_pheno = build_knn_adjacency(beh_sim, KNN_K)
    save_experiment("PHENOTYPE_ONLY", X_behavior, adj_pheno, {"behavior_similarity": beh_sim})

    # EEG-plus-phenotype graph.
    node_full = np.concatenate([X_eeg, X_behavior], axis=1)
    adj_full = build_knn_adjacency(fused_sim, KNN_K)
    save_experiment("EEG_PHENOTYPE", node_full, adj_full, {
        "eeg_similarity": eeg_sim,
        "behavior_similarity": beh_sim,
        "fused_similarity": fused_sim
    })

    # Save aligned metadata locally for downstream validation. Do not commit it.
    meta.to_csv(OUTPUT_ROOT / "subject_metadata_for_all_experiments.csv", index=False)
    print("\nDONE. Saved:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
