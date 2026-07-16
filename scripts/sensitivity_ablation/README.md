# EPG-SD sensitivity and ablation scripts

This folder contains the paper-linked sensitivity analyses that are separate
from the core Stage 00–09 pipeline.

## Script-to-analysis mapping

| Script | Analysis | Manuscript evidence |
|---|---|---|
| `12_alpha_joint_selection.py` | Joint fusion-weight, graph-neighbor, and UMAP-neighbor selection at K=5 | Table 13 |
| `13_alpha_weight_sensitivity_missing_values.py` | Fixed-setting alpha sweep for 0.0–0.4, 0.9, and 1.0; the alpha=0.6 control comes from the archived final run | Table 14 |
| `14_phenotype_similarity_sensitivity.py` | Cosine, Euclidean-RBF, Mahalanobis-RBF, and correlation phenotype similarity | Table 15 |
| `15_downstream_hyperparameter_sensitivity.py` | PCA dimension, graph neighbors, UMAP neighbors, and VGAE latent dimension | Table 16 |
| `16_cnn_latent_sensitivity_single_stream.py` | CNN-AE latent sizes 64, 128, and 256 using one saved embedding stream | Table 17 |
| `17_two_stream_ablation.py` | Stream 1, stream 2, standardized stream average, and two-stream concatenation | Table 23 |
| `18_umap_seed_robustness.py` | Twenty UMAP random seeds with the VGAE representation held fixed | Table 27 |

PCA dimension, graph neighbors, UMAP neighbors, and VGAE latent dimension are
combined in one script because they were evaluated as four blocks in the same
controlled downstream experiment.

## Input paths

The scripts do not contain private absolute paths. Configure paths with command
line arguments where available or with these environment variables:

```bash
export EPGSD_PROJECT_ROOT=/path/to/project
export EPGSD_ANALYSIS_ROOT=/path/to/TECHNICAL_FIX_SPECTRAL_UMAP
export EPGSD_SENSITIVITY_OUT=/path/to/separate/output/folder
```

Optional explicit overrides:

```bash
export EPGSD_SUBJECT_EMBEDDING=/path/to/subject_embedding.npy
export EPGSD_SUBJECT_METADATA=/path/to/subject_metadata.csv
export EPGSD_CNN_LATENT_ROOT=/path/to/cnn_latent_input_folder
export EPGSD_VGAE_EMBEDDING=/path/to/latent_embeddings.npy
export EPGSD_FINAL_LABELS=/path/to/subject_subtypes.csv
```

Use a different output folder for each script. Reusing one
`EPGSD_SENSITIVITY_OUT` path across multiple scripts can overwrite outputs.

## Important run details

For the phenotype-similarity table in the latest manuscript, retrain all four
conditions rather than reusing the archived cosine row:

```bash
python 14_phenotype_similarity_sensitivity.py \
  --root "$EPGSD_ANALYSIS_ROOT" \
  --out /path/to/phenotype_similarity_output \
  --retrain-cosine
```

The alpha-weight sensitivity script intentionally computes only the values that
were absent from the earlier joint grid. The final alpha=0.6 row must be taken
from the archived final EPG-SD result when assembling the table.

The two-stream script requires an explicit output folder:

```bash
python 17_two_stream_ablation.py \
  --root "$EPGSD_ANALYSIS_ROOT" \
  --recording-meta /path/to/band_decomposition_metadata_FS500.csv \
  --out /path/to/two_stream_output
```

## Data policy

Do not commit participant-level labels, subject metadata, embeddings, graph
arrays, checkpoints, clinical-detail files, or per-seed outputs. Commit only
source code, documentation, figures approved for release, and aggregate tables
whose provenance has been verified.
