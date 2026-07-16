# EPG-SD validation scripts

This folder contains the validation analyses that are not cleanly represented
as separate files in the core Stage 00–09 folder.

## Included analyses

### `10_k_selection_k2_to_k10.py`

Uses the saved final two-dimensional UMAP representation and the saved fused
graph adjacency. It compares K=2 through K=10 using:

- silhouette, Davies–Bouldin, and Calinski–Harabasz indices;
- graph-Laplacian eigengap;
- minimum and maximum subtype size;
- maximum/minimum size ratio; and
- 50 repeated 80% subsampling runs using ARI and NMI.

The UMAP axes are standardized before candidate-K spectral clustering because
that is the operation used in the archived K-selection analysis supporting
Tables 9 and 10. This is distinct from the archived final K=5 run in the core
pipeline, whose reported main metrics come from its own final execution.

The adjacency input is mandatory. The script deliberately does not construct a
fallback adjacency, because a fallback would change the eigengap analysis.

Example:

```bash
python 10_k_selection_k2_to_k10.py \
  --umap-csv /path/to/umap_coordinates.csv \
  --adjacency /path/to/adjacency.npy \
  --out /path/to/k_selection_outputs
```

### `11_k456_heldout_age_ehq.py`

Reads saved K=4, K=5, and K=6 labels and computes held-out Age and EHQ
associations without rerunning preprocessing, representation learning, graph
construction, VGAE, UMAP, or clustering.

The script:

- requires saved partitions matching the expected 430-subject signatures;
- checks subject identifiers or verifies the saved row order;
- computes ANOVA and Kruskal–Wallis tests;
- reports eta-squared and omega-squared;
- applies Benjamini–Hochberg FDR correction over the six-variable family; and
- stops rather than guessing when candidate files or subject order are
  ambiguous.

Example:

```bash
python 11_k456_heldout_age_ehq.py \
  --root /path/to/project_outputs \
  --out /path/to/k456_heldout_outputs
```

## Analyses already in the core Stage 09 script

The archived final K=5 script already includes:

- final K=5 permutation testing;
- repeated 80% subsampling;
- bootstrap stability;
- final clinical statistics; and
- final K=5 Age/EHQ association testing.

They are not duplicated here.

## Data policy

Do not commit participant-level label files, subject metadata, latent arrays, or
model checkpoints. Generated `labels_K*.csv` files contain participant-level
assignments and must remain outside version control.

Recommended `.gitignore` rules:

```gitignore
**/labels_K*.csv
**/selected_saved_label_sources.csv
**/verification_manifest.json
```
