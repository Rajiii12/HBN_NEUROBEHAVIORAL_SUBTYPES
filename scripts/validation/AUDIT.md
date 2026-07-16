# Validation package audit

## Scientific mapping

- `10_k_selection_k2_to_k10.py` corresponds to the archived K=2–10
  candidate-resolution experiment that generated `k_selection_metrics.csv` and
  `eigengap.csv`.
- `11_k456_heldout_age_ehq.py` corresponds to the saved-label analysis used for
  the K=4/5/6 Age and EHQ robustness comparison.
- Final K=5 permutation, resampling, bootstrap, and clinical analyses remain in
  `scripts/core_stage07_to09/09_final_k5_validation_archived.py`.

## Preserved settings

- Candidate K range: 2–10
- Spectral nearest-neighbor setting: 12
- Random seed: 42
- K-selection stability: 50 repetitions of 80% subsampling
- Held-out resolutions: K=4, K=5, K=6
- Held-out variables: Age and EHQ
- FDR family: P-factor, attention, internalizing, externalizing, Age, and EHQ

## Public-release changes

- Removed hard-coded private machine paths.
- Removed editorial and manuscript-drafting comments.
- Required the saved graph adjacency instead of silently creating a fallback
  eigengap graph.
- Changed manifest paths to project-relative paths.
- Added headless Matplotlib configuration.
- Did not include participant labels, metadata, arrays, checkpoints, or
  generated outputs.

## Provenance caution

The K-selection analysis operates on the saved final UMAP representation rather
than retraining the VGAE or UMAP for every K. It is therefore a comparison of
candidate partitions on a common fixed representation.
