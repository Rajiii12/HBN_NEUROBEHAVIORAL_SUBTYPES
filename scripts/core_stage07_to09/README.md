# Core graph, VGAE, and final K=5 scripts

This folder contains the archived manuscript-aligned continuation after the
192-dimensional subject EEG descriptor.

## Files

- `07_prepare_phenotype_and_construct_graph.py`: median-imputes and standardizes four phenotype variables; builds EEG-only, phenotype-only, and fused graphs with `k=12` and `alpha=0.6`.
- `vgae_model.py`: PyTorch Geometric VGAE architecture and loss logging.
- `08_train_vgae.py`: trains the three graph experiments with `196 -> 64 -> 16` for the fused condition and saves the posterior mean.
- `09_final_k5_validation_archived.py`: archived result-producing fused workflow with its own dense VGAE, UMAP-30, spectral K=5, stability, permutation, and clinical analyses.

## Important provenance notes

1. Graph neighbors are obtained by applying Euclidean kNN to rows of the cosine-similarity matrix. This is **similarity-profile kNN**, not direct top-k selection from each similarity row.
2. Phenotype preparation is part of Stage 07; there is no separate phenotype-only preparation script in the archived workflow.
3. Stage 09 is not a modular continuation of Stage 08. It retrains an archived dense VGAE and is retained because it is the final result-producing workflow.
4. The Stage 07 phenotype column order is `internalizing, attention, externalizing, p_factor`, matching the archived script.
5. Generated `.npy`, `.pt`, metadata, subtype assignments, and participant-level CSV files must remain local and must never be committed.

## Configuration

Paths can be provided with environment variables:

- `EPGSD_SUBJECT_EMBED_FILE`
- `EPGSD_SUBJECT_META_FILE`
- `EPGSD_GRAPH_OUTPUT_ROOT`
- `EPGSD_GRAPH_ROOT`
- `EPGSD_VGAE_OUTPUT_ROOT`
- `EPGSD_FINAL_OUTPUT_ROOT`
