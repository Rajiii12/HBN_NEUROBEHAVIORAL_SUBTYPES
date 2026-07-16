# Sensitivity/ablation provenance audit

## Cross-verification against the latest manuscript

The latest manuscript contains the following linked analyses:

- Table 13: joint alpha and neighborhood selection;
- Table 14: fixed-setting alpha sensitivity;
- Table 15: phenotype-similarity sensitivity;
- Table 16: PCA, graph-neighbor, UMAP-neighbor, and VGAE-latent sensitivity;
- Table 17: CNN-AE latent sensitivity;
- Table 23: two-stream ablation; and
- Table 27: UMAP seed robustness.

## Numerical evidence status

| Table | Status in the collected result audit |
|---:|---|
| 13 | Method script located; exact compact four-row CSV not yet collected |
| 14 | Method script located; exact combined alpha-sensitivity CSV not yet collected |
| 15 | Corrected method script located; exact aggregate CSV not yet present in the public result folder |
| 16 | Verified: all 12 CSV rows match the manuscript after rounding |
| 17 | Verified: all three CSV rows match the manuscript after rounding |
| 23 | Verified: condition summaries and pairwise-seed stability match |
| 27 | Verified: all nine summary rows match after rounding |

The first three scripts are therefore suitable as transparent method code, but
their aggregate result CSVs must be collected and checked before those tables
are described as fully reproduced from the public repository.

## Scientific caveats retained in the release

### Downstream hyperparameter sensitivity

`15_downstream_hyperparameter_sensitivity.py` is the archived implementation
that generated the verified Table 16 CSV. Its sensitivity-specific graph
construction is not byte-equivalent to the final Stage 07 graph implementation:
it shifts EEG cosine similarity, uses an RBF phenotype affinity, and performs a
direct weighted top-k sparsification. The manuscript already treats the values
as within-block controlled reruns rather than exact reproductions of the final
EPG-SD result.

### CNN latent sensitivity

`16_cnn_latent_sensitivity_single_stream.py` loads five band embeddings from one
saved CNN-AE stream for each latent size. It does not retrain and concatenate
both saved streams. Table 17 should therefore be interpreted as a controlled
single-stream latent-size sensitivity, not as a complete two-stream
architecture retraining.

### Alpha sensitivity

`13_alpha_weight_sensitivity_missing_values.py` uses the archived diagnostic
VGAE configuration for the missing alpha values. It does not independently
generate the alpha=0.6 headline row; that row comes from the archived final
pipeline.

### Phenotype similarity

The latest manuscript table is a fresh fixed-seed comparison of all four
similarity functions. Run the script with `--retrain-cosine`; the default mode
reuses the archived final cosine artifacts and corresponds to a different
control-table design.

## Excluded candidates

The uploaded review archive also contained VGAE-component and UMAP-component
ablation candidates. They are not included here because they were outside the
requested sensitivity list and their available aggregate outputs were not
numerically consistent with the current manuscript tables.
