# HBN Neurobehavioral Subtype Discovery

This repository contains the analysis code, aggregate result tables, and
figures for the EPG-SD EEG–phenotype subtype-discovery study.

The repository does not contain raw EEG data, participant metadata,
participant-level subtype assignments, latent embeddings, trained model
weights, or other identifiable subject-level outputs.

## Repository structure

### `scripts/core_stage00_to06/`

Early EEG-processing and representation-learning stages:

- `00_sampling_rate_and_input_shape_audit.py`
  Audits sampling rate, channel count, epoch shape, and input consistency.

- `01_preprocessing.ipynb`
  Performs EEG preprocessing, filtering, ICA/ICLabel artifact handling,
  channel standardization, epoching, rejection, and shape normalization.

- `02_band_decomposition.py`
  Decomposes the preprocessed EEG into delta, theta, alpha, beta, and
  gamma frequency-band representations.

- `03_embedding_stream_1.py`
  Generates the first band-wise CNN-autoencoder embedding stream.

- `04_embedding_stream_2.py`
  Generates the second separately saved same-architecture embedding stream.

- `05_ten_view_fusion_pca64.py`
  Standardizes and concatenates the ten band/stream views and reduces the
  combined representation to 64 PCA components.

- `06_subject_aggregation.py`
  aggregates window-level embeddings into 192 subject-level EEG features
  using the mean, standard deviation, and median.

### `scripts/core_stage07_to09/`

Graph construction, graph representation learning, and final clustering:

- `07_prepare_phenotype_and_construct_graph.py`
  Prepares the four phenotype variables, computes EEG and phenotype
  similarities, combines them using alpha 0.6, constructs the graph with
  12 neighbors, and forms 196-dimensional node features.

- `vgae_model.py`
  Defines the variational graph autoencoder architecture
  196 → 64 → 16.

- `08_train_vgae.py`
  Trains EEG-only, phenotype-only, and fused VGAE experiments.

- `09_final_k5_validation_archived.py`
  Runs the archived final fused analysis using UMAP with 30 neighbors and
  spectral clustering with K=5. It also contains final quality metrics,
  permutation testing, subsampling, bootstrap stability, and clinical
  analyses.

### `scripts/validation/`

Additional validation analyses:

- `10_k_selection_k2_to_k10.py`
  Evaluates K=2 through K=10 using internal metrics, eigengaps, cluster
  sizes, balance, and repeated-subset stability.

- `11_k456_heldout_age_ehq.py`
  Evaluates held-out Age and EHQ effects for K=4, K=5, and K=6 using
  ANOVA, Kruskal–Wallis tests, effect sizes, and FDR correction.

### `scripts/sensitivity_ablation/`

Controlled sensitivity and representation-ablation analyses:

- `12_alpha_joint_selection.py`
  Joint alpha and neighborhood configuration analysis.

- `13_alpha_weight_sensitivity_missing_values.py`
  Fixed-pipeline alpha-weight sensitivity analysis.

- `14_phenotype_similarity_sensitivity.py`
  Compares cosine, correlation, Euclidean-RBF, and Mahalanobis-RBF
  phenotype similarities.

- `15_downstream_hyperparameter_sensitivity.py`
  Evaluates PCA dimension, graph neighbors, UMAP neighbors, and VGAE
  latent dimension.

- `16_cnn_latent_sensitivity_single_stream.py`
  Evaluates CNN-autoencoder latent dimensions 64, 128, and 256 using the
  available single-stream latent experiments.

- `17_two_stream_ablation.py`
  Compares single-stream and two-stream CNN-autoencoder representations.

- `18_umap_seed_robustness.py`
  Evaluates robustness across UMAP random initializations.

### `scripts/baselines/`

Conventional, adapted EEG, and deep-clustering comparisons:

- `19_native_dbscan.py`
  Evaluates native DBSCAN solutions and coverage-aware selection.

- `20_jiman_pca95_umap_kmeans.py`
  Implements the adapted PCA95–UMAP–KMeans benchmark.

- `21_pan_leal_adapted_eeg_benchmarks.py`
  Implements the adapted Pan-style and Leal-style EEG clustering
  comparisons.

- `22_tuned_conventional_clustering.py`
  Evaluates tuned KMeans, Gaussian mixture, hierarchical, and spectral
  clustering configurations.

- `23_deep_clustering_baselines.py`
  Evaluates DEC, IDEC, DeepCluster-style, VAE-GMM/VaDE-style, and
  GAE-KMeans baselines.

- `24_dec_idec_latent_diagnostics.py`
  Examines saved DEC and IDEC latent geometry, effective rank, active
  dimensions, duplicate embeddings, and cluster balance.

### `results/table/`

Contains participant-free aggregate CSV evidence for manuscript tables.

Verified table evidence currently covers Tables:

`9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, and 32`.

Table proofs still requiring source reconciliation are:

`13 and 26`.

### `figures/`

Contains the final manuscript figures in PNG and PDF formats.

## Analysis order

The intended execution order is:

1. Sampling-rate and input-shape audit
2. EEG preprocessing
3. Frequency-band decomposition
4. CNN-autoencoder embedding streams
5. Ten-view fusion and PCA
6. Subject-level aggregation
7. Phenotype preparation and graph construction
8. VGAE training
9. UMAP and spectral clustering
10. Validation analyses
11. Sensitivity and ablation analyses
12. Baseline comparisons

## Main archived configuration

The final archived EPG-SD configuration uses:

- 430 subjects
- 128 EEG channels
- 500 Hz sampling rate
- 1-second windows with 50% overlap
- Five frequency bands
- Two separately generated same-architecture CNN-autoencoder streams
- 128-dimensional embedding per stream and band
- Ten-view concatenation followed by PCA to 64 dimensions
- 192 subject-level EEG features
- Four standardized phenotype variables
- Fusion weight alpha = 0.6
- Graph neighbors = 12
- VGAE architecture 196 → 64 → 16
- UMAP neighbors = 30
- Spectral clustering K = 5
- 1,000 permutation tests
- Subsampling and bootstrap stability analyses

## Data and privacy

The public repository intentionally excludes:

- Raw or preprocessed participant EEG arrays
- Participant metadata
- Subject-level phenotype records
- Participant subtype assignments
- Latent embeddings and UMAP coordinates
- NPY, PT, and PTH files
- Model checkpoints
- Private local filesystem paths

Only code, figures, and aggregate participant-free result tables are
included.
