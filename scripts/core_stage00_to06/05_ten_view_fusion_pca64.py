"""Fuse ten band-specific embeddings and reduce them to PCA-64.

Five embeddings from each of two saved, same-architecture Conv1D
autoencoder streams are standardized separately, concatenated, and reduced
to a 64-dimensional recording-level representation.
"""

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

AE_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/autoencoder_embeddings_dynamic_norm_fixed")
CNN_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/cnn_autoencoder_embeddings_norm_fixed")

OUTPUT_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/tensor_fusion_norm_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_COMPONENTS = 64
SEED = 42

# Existing outputs are preserved unless replacement is explicitly enabled.
FORCE_OVERWRITE = False


def assert_can_write(path):
    if path.exists() and not FORCE_OVERWRITE:
        raise FileExistsError(
            f"\nFile already exists and FORCE_OVERWRITE=False:\n{path}\n\n"
            "Set FORCE_OVERWRITE=True to replace existing outputs."
        )


def main():
    views = []
    names = []

    print("\nFINAL DUAL-BRANCH CNN-AE NORMALIZED FUSION")
    print("AE_DIR:", AE_DIR)
    print("CNN_DIR:", CNN_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    # ------------------------------------------------------------
    # Load embedding stream 1; the legacy directory name is retained.
    # ------------------------------------------------------------
    for b in BANDS:
        fp = AE_DIR / f"{b}_embedding.npy"
        if not fp.exists():
            raise FileNotFoundError(fp)

        emb = np.load(fp).astype(np.float32)
        if emb.ndim != 2 or emb.shape[1] != 128:
            raise ValueError(
                f"Expected recording-by-128 embeddings for {fp}; got {emb.shape}"
            )
        views.append(emb)
        names.append(f"AE_{b}")

    # ------------------------------------------------------------
    # Load embedding stream 2; the legacy directory name is retained.
    # ------------------------------------------------------------
    for b in BANDS:
        fp = CNN_DIR / f"{b}_cnn_embedding_fixed.npy"
        if not fp.exists():
            raise FileNotFoundError(fp)

        emb = np.load(fp).astype(np.float32)
        if emb.ndim != 2 or emb.shape[1] != 128:
            raise ValueError(
                f"Expected recording-by-128 embeddings for {fp}; got {emb.shape}"
            )
        views.append(emb)
        names.append(f"CNN_AE_{b}")

    # ------------------------------------------------------------
    # Check sample count
    # ------------------------------------------------------------
    n_samples = [v.shape[0] for v in views]
    if len(set(n_samples)) != 1:
        raise ValueError(f"Embedding sample mismatch across views: {n_samples}")

    # ------------------------------------------------------------
    # Standardize each view separately
    # ------------------------------------------------------------
    scaled = []

    for name, view in zip(names, views):
        print(name, view.shape, "std before:", float(view.std()))

        view_scaled = StandardScaler().fit_transform(view)
        scaled.append(view_scaled)

        print(name, view_scaled.shape, "std after:", float(view_scaled.std()))

    # ------------------------------------------------------------
    # Concatenate all views
    # 10 views = 5 AE band views + 5 CNN-AE band views
    # Each view = 128 dimensions
    # Total input dimension = 10 x 128 = 1280
    # ------------------------------------------------------------
    concat = np.concatenate(scaled, axis=1)

    # ------------------------------------------------------------
    # PCA fusion
    # ------------------------------------------------------------
    n_comp = min(PCA_COMPONENTS, concat.shape[0] - 1, concat.shape[1])

    pca = PCA(n_components=n_comp, random_state=SEED)
    fused = pca.fit_transform(concat)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    npy_path = OUTPUT_DIR / "AE_PLUS_CNN_NORM_FIXED_fused_embedding.npy"
    csv_path = OUTPUT_DIR / "AE_PLUS_CNN_NORM_FIXED_fused_embedding.csv"
    summary_path = OUTPUT_DIR / "AE_PLUS_CNN_NORM_FIXED_pca_summary.csv"
    ev_path = OUTPUT_DIR / "AE_PLUS_CNN_NORM_FIXED_pca_explained_variance.csv"
    view_path = OUTPUT_DIR / "AE_PLUS_CNN_NORM_FIXED_view_summary.csv"

    for p in [npy_path, csv_path, summary_path, ev_path, view_path]:
        assert_can_write(p)

    np.save(npy_path, fused)

    pd.DataFrame(fused).to_csv(csv_path, index=False)

    pd.DataFrame({
        "metric": [
            "n_samples",
            "n_views",
            "input_dim",
            "pca_components",
            "explained_variance_ratio_sum"
        ],
        "value": [
            fused.shape[0],
            len(views),
            concat.shape[1],
            n_comp,
            float(pca.explained_variance_ratio_.sum())
        ]
    }).to_csv(summary_path, index=False)

    pd.DataFrame({
        "component": np.arange(1, n_comp + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(ev_path, index=False)

    pd.DataFrame({
        "view_name": names,
        "input_shape": [str(v.shape) for v in views],
        "scaled_shape": [str(v.shape) for v in scaled]
    }).to_csv(view_path, index=False)

    print("\nConcat shape:", concat.shape)
    print("Fused shape:", fused.shape)
    print("Explained variance ratio sum:", float(pca.explained_variance_ratio_.sum()))
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
