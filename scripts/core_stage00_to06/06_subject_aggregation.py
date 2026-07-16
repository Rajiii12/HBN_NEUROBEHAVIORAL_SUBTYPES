"""Aggregate recording-level PCA features into subject descriptors.

For each subject, the script concatenates the mean, standard deviation,
and median of the 64-dimensional recording descriptors, producing a
192-dimensional EEG representation.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd

META_FILE = Path(os.environ.get(
    "EPGSD_MASTER_METADATA",
    "./data/preprocessed_HBN_v2/master_metadata.csv",
))

FUSION_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/tensor_fusion_norm_fixed")
OUTPUT_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/subject_embeddings_norm_fixed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate filenames are retained for compatibility with archived outputs.
EMBED_CANDIDATES = [
    FUSION_DIR / "AE_PLUS_CNN_NORM_FIXED_fused_embedding.npy",
    FUSION_DIR / "NORM_FIXED_fused_embedding.npy",
    FUSION_DIR / "fused_embedding_norm_fixed.npy",
    FUSION_DIR / "fused_embedding.npy",
    FUSION_DIR / "AE_CNN_NORM_FIXED_fused_embedding.npy",
    FUSION_DIR / "CNN_AE_NORM_FIXED_fused_embedding.npy",
]


def find_embedding_file():
    for fp in EMBED_CANDIDATES:
        if fp.exists():
            return fp

    raise FileNotFoundError(
        "Could not find final fused embedding file. Checked:\n"
        + "\n".join(str(p) for p in EMBED_CANDIDATES)
        + "\n\nRun this command to see the actual filename:\n"
        + "ls -lh TECHNICAL_FIX_SPECTRAL_UMAP/tensor_fusion_norm_fixed"
    )


def main():
    embed_file = find_embedding_file()

    X = np.load(embed_file).astype(np.float32)
    meta = pd.read_csv(META_FILE)

    print("\nFINAL NORMALIZED SUBJECT AGGREGATION")
    print("Embedding file:", embed_file)
    print("Embeddings:", X.shape)
    print("Metadata:", meta.shape)
    print("Output:", OUTPUT_DIR)

    if len(X) != len(meta):
        raise ValueError(
            f"Embedding count {len(X)} != metadata count {len(meta)}"
        )

    if X.ndim != 2 or X.shape[1] != 64:
        raise ValueError(
            f"Expected recording-level PCA64 features; received {X.shape}"
        )

    if "subject" not in meta.columns:
        raise ValueError("metadata must contain a 'subject' column")

    subjects = meta["subject"].values

    feats = []
    rows = []

    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        Xi = X[idx]

        agg = np.concatenate(
            [
                Xi.mean(axis=0),
                Xi.std(axis=0),
                np.median(Xi, axis=0),
            ],
            axis=0,
        )

        feats.append(agg)

        row0 = meta.iloc[idx[0]].to_dict()

        rows.append(
            {
                "subject": sid,
                "n_samples": len(idx),
                "age": row0.get("age", np.nan),
                "sex": row0.get("sex", np.nan),
                "p_factor": row0.get("p_factor", np.nan),
                "attention": row0.get("attention", np.nan),
                "internalizing": row0.get("internalizing", np.nan),
                "externalizing": row0.get("externalizing", np.nan),
                "ehq_total": row0.get("ehq_total", np.nan),
            }
        )

    feats = np.asarray(feats, dtype=np.float32)
    table = pd.DataFrame(rows)

    np.save(
        OUTPUT_DIR / "subject_embedding_mean_std_median_norm_fixed.npy",
        feats,
    )

    pd.DataFrame(feats).to_csv(
        OUTPUT_DIR / "subject_embedding_mean_std_median_norm_fixed.csv",
        index=False,
    )

    table.to_csv(
        OUTPUT_DIR / "subject_metadata_norm_fixed.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "metric": [
                "embedding_file",
                "n_recording_samples",
                "n_subjects",
                "subject_feature_dim",
                "aggregation",
            ],
            "value": [
                str(embed_file),
                X.shape[0],
                feats.shape[0],
                feats.shape[1],
                "mean + std + median",
            ],
        }
    ).to_csv(
        OUTPUT_DIR / "subject_aggregation_summary.csv",
        index=False,
    )

    print("Subject features:", feats.shape)
    print("Subjects:", len(table))
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
