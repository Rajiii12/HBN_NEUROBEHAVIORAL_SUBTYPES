# EPG-SD baseline analyses

This folder contains the baseline experiments used to contextualize the final
EPG-SD solution. Generated participant-level labels, latent arrays, model
outputs, and checkpoints are not included.

## Script-to-analysis mapping

- `19_native_dbscan.py`: native-variable-K DBSCAN and the diagnostic search for
  a naturally occurring, non-degenerate K=5 solution. This supports the native
  DBSCAN analysis reported in Table 28.
- `20_jiman_pca95_umap_kmeans.py`: Jiman-style PCA95–UMAP–KMeans at K=5.
- `21_pan_leal_adapted_eeg_benchmarks.py`: Pan-style covariance/Riemannian-AE
  and Leal-style UMAP multi-clustering adaptations. Together with scripts 19
  and 20, these methods form the adapted EEG benchmark in Table 29.
- `22_tuned_conventional_clustering.py`: full tuned KMeans, GMM,
  hierarchical, and spectral-clustering grid on the saved EEG-only and
  EEG–phenotype representations. This supports Table 30.
- `23_deep_clustering_baselines.py`: DEC, IDEC, DeepCluster-style,
  VAE-GMM/VaDE-style, and GAE-KMeans baseline rows for Table 32.
- `24_dec_idec_latent_diagnostics.py`: saved-latent single-run diagnostics for
  DEC and IDEC, supporting Table 31.

## Provenance boundaries

The exact aggregate CSVs for Tables 28, 30, and 31 were independently matched
to the manuscript values. The five non-EPG-SD rows in the deep-clustering
summary also match. The archived combined Table 32 CSV used a non-canonical
EPG-SD control row, so script 23 deliberately emits only the five baseline
rows. The EPG-SD row must be taken from the archived final Stage 09 result.

The exact consolidated five-row adapted-benchmark CSV for Table 29 was not
located. The method implementations are included, but a fresh aggregate output
must be generated and checked before publishing Table 29 result files.

## Typical inputs

The scripts operate on saved subject-level arrays and metadata, not on raw EEG.
Provide paths explicitly through each script's command-line arguments. Run
outputs outside the repository or under an ignored output directory.

## Data policy

Do not commit participant labels, subject metadata, latent arrays, adjacency
arrays, checkpoints, or per-subject clinical outputs. Only source code and
small aggregate, verified result tables belong in the public repository.
