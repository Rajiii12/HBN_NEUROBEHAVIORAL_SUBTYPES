# Stage 07–10 final code audit

## Files inspected

- `07_graph_construction_montage_fixed_graphK12_alpha06.py`
- `vgae_models_logged_fixed.py`
- `08_train_vgae_montage_fixed_graphK12_alpha06.py`
- `09_final_forced_k5_validation_MONTAGE_FIXED_graphK12_alpha06_umap30.py`

All four source files compile successfully.

## Scientific findings

### Stage 07: phenotype preparation and graph construction

- Phenotype preparation is implemented inside graph construction; there is no separate phenotype-preparation script in the archived workflow.
- Phenotype columns are `internalizing`, `attention`, `externalizing`, and `p_factor`.
- Missing phenotype values are median-imputed, then standardized.
- EEG and phenotype cosine-similarity matrices are combined with `alpha = 0.6`.
- Graph neighborhood size is `k = 12`.
- The graph uses Euclidean kNN on similarity-matrix rows. This is similarity-profile kNN, not direct top-k selection by fused similarity.
- The fused node feature dimension is 192 EEG features plus 4 phenotype features = 196.

### VGAE model and training

- The model dependency is present and correctly importable after standardizing the filename.
- Architecture: graph convolution 196 -> 64, then separate 16-dimensional posterior mean and log-standard-deviation heads.
- Training: 500 epochs, Adam learning rate 0.001, KL weight 1.0, weight decay 1e-5.
- After `model.eval()`, `VGAE.encode()` returns the posterior mean, so the saved representation is deterministic posterior mean.

### Final K=5 workflow

- The archived final script does not consume the output of the PyTorch Geometric VGAE training script.
- It reconstructs the fused graph and trains its own dense VGAE before UMAP-30 and spectral clustering with K=5.
- It performs 100 subsamples, 100 bootstrap analyses, 1000 label permutations, and clinical analyses.
- It writes participant-level metadata and subtype assignments; those generated outputs must remain local and must not be committed.

## GitHub decision

The original uploaded ZIP should not be committed unchanged because it contains development-facing comments and hard-coded project-relative assumptions. A cleaned package was created that preserves the archived numerical logic while adding path configuration and provenance documentation.

## Cleaned package

`core_stage07_10_github_ready.zip`

Contents:

- `07_prepare_phenotype_and_construct_graph.py`
- `vgae_model.py`
- `08_train_vgae.py`
- `09_final_k5_validation_archived.py`
- `README.md`
- `GITIGNORE_SNIPPET.txt`
- `SHA256SUMS.txt`

The cleaned scripts compile and contain no private absolute paths, AI references, or conversational development comments.
