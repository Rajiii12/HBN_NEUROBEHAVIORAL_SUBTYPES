# TDBRAIN External Validation — EPG-SD

## Scope

This directory contains recovered Python source code for the
TDBRAIN transdiagnostic external-validation analysis reported
in the EPG-SD IEEE Access manuscript and supplementary material.

The primary analysis uses 85 complete-case participants with
ADHD, major depressive disorder (MDD), or obsessive-compulsive
disorder (OCD), and retains six candidate subtypes (K=6).

This is a complementary external evaluation, not an exact
replication of the five HBN-derived neurobehavioral profiles.

## Input representations

### EEG representation

The analysis uses a frozen 128-dimensional TDBRAIN EEG
subject representation from the preceding Experiment 2,
Stage 7G equal-clean-window sensitivity analysis.

That upstream analysis retained eight clean eyes-open and
eight clean eyes-closed windows per participant.

Its saved EEG feature matrix contains 86 participants.
The Stage 7G EEG-only sensitivity analysis used K=3.

Those K=3 labels are not the Experiment 3 subtype labels.

### Behavioral representation

The primary phenotype view contains 60 baseline NEO-FFI
item responses from session 1 (sessID=1).

No follow-up NEO-FFI responses, unofficial Big-Five
scoring, or imputation are used in the primary fusion cohort.

After complete-case alignment, 85 participants contribute:

- EEG features: 128
- Baseline NEO-FFI variables: 60
- Total subject-level model features: 188

NEO-FFI is a model input and is not treated as an
independent held-out validation measure.

## Primary modeling

The archived Experiment 3 source implements:

- EEG-phenotype similarity fusion with alpha=0.6
- A four-nearest-neighbor subject graph
- VGAE architecture: 188 -> 64 -> 16
- VGAE training: 500 epochs
- UMAP-based structural assessment across candidate K values
- Primary K=6 solution using UMAP seed 42
- Additional initialization, permutation, and resampling analyses
- Subsequent held-out clinical characterization

Cluster resolution was selected from structural criteria
before clinical diagnostic and symptom associations
were examined.

## Source-code organization

The scripts in `scripts/` are organized as follows:

1. Stage 0B: cohort and phenotype-design preparation
2. Stage 0B-R1: baseline-session resolution audit
3. Stage 0B-R2: finalized baseline selection
4. Stage 1A: fused graph construction
5. Stage 2A: VGAE training
6. Stage 3A: UMAP and K-selection assessment
7. Stage 3B: authoritative K=6 partition
8. Stage 3C: 20-seed robustness
9. Stage 3D: fixed-label permutation testing
10. Stage 3E: subject-resampling robustness
11. Stage 4A-R1: held-out clinical preparation
12. Stage 4B: characterization design
13. Stage 4C: held-out association tests
14. Stage 4D: BDI post-hoc design and feasibility
15. Stage 4E: MDD BDI pairwise post-hoc tests

Stage 0B-R1 is an audit stage; Stage 0B-R2 is
the subsequent baseline-selection stage.

## Original code provenance

The 15 Python scripts were extracted from code cells
in `EXTERNAL_VALIDATION2.ipynb`.

Publication filenames were normalized to their actual
stage purposes. The recovered notebook-cell source is preserved separately from these publication copies. Thirteen scripts contain standardized console-output messages only; the scientific program structure was not changed. The table records both original and publication-copy SHA-256 fingerprints.

The source file hashes and original one-based notebook
cell numbers are recorded in the table below.

| Public script | Original notebook cell | Original SHA-256 | Publication SHA-256 |
|---|---:|---|---|
| `TDBRAIN_STAGE0B_COHORT_FUSION_DESIGN.py` | 2 | `bc688c6108b88559f184c55acdf6bf57b57684ad6c48ad515d104015a2c709d2` | `bc688c6108b88559f184c55acdf6bf57b57684ad6c48ad515d104015a2c709d2` |
| `TDBRAIN_STAGE0B_R1_BASELINE_RESOLUTION_AUDIT.py` | 3 | `42875d033b4786c85fa7ea577f55fa1c61c348166d1b43836ab2746a0d9c639f` | `42875d033b4786c85fa7ea577f55fa1c61c348166d1b43836ab2746a0d9c639f` |
| `TDBRAIN_STAGE0B_R2_BASELINE_SELECTION.py` | 4 | `7cf2e903c298765c06e6a9a85238fa50244cddb489732b247546ec76a1acb950` | `5f41f9165ecb462924972e1803dbb51a79322144d85dbe130518503f274dcee9` |
| `TDBRAIN_STAGE1A_FUSED_GRAPH_CONSTRUCTION.py` | 5 | `9cbeeb1748636603071e260bf240c7435aa5d861589f5c2c745191283fd4785a` | `341765fde2338ff1fe7efff92e42553fb516c4d75622a8ba123a15e14e5f5be0` |
| `TDBRAIN_STAGE2A_VGAE_TRAINING.py` | 6 | `ef32f5ff700f15b63ee3c541e7805e77fb256d10834ad434392f5c0131daf78d` | `05350566a2759e02238ea483862cc11c0007bb7da41446a5e1a3e91dd2804d29` |
| `TDBRAIN_STAGE3A_K_SELECTION.py` | 7 | `e46dc862a57722d417cc5404af9361c8d694342e627193eb466a8f089908d4ed` | `a87dbaccffbd2accf6863eb809b0c15e078c878e11faf5fb33f5ada1686731c5` |
| `TDBRAIN_STAGE3B_PRIMARY_K6.py` | 8 | `0d371090c502c3f4fd3c4aeb09d1f24dfe5a1ecdf2c65542a304117c36c88cb0` | `89d580c8f8221c3328bb02dc709308ad8865c3819cbc9fb581c17ee57a69369a` |
| `TDBRAIN_STAGE3C_20SEED_ROBUSTNESS.py` | 9 | `f20165394b2c6388ec28249e4822be8a73c123f33324a5009aabfa2489ca8e62` | `c603a6d1a41e91bbf3c2b15c36a62e7f08968808d1151f6dd5857dec92f51db7` |
| `TDBRAIN_STAGE3D_PERMUTATION_TEST.py` | 10 | `80d92d2d77ebc6af0d190cd19c39b29610d41202bc6988830d3ba70a5f7393c8` | `bdc2e2edd265efa398eaba5fbb4f0893460b20acf244bee1a3f0e9825dff041b` |
| `TDBRAIN_STAGE3E_SUBJECT_RESAMPLING.py` | 11 | `00062a20905c682b561227e2449d36cd918f738f6c6f62b530e66cdb9e548d5a` | `67c0e2c851e8e0de6c68c5937a26769ef4694ff694496c87aff3780c2bfcb430` |
| `TDBRAIN_STAGE4A_R1_HELDOUT_PREPARATION.py` | 13 | `1b27b9e1b15149f8d48dc1eb882c3d856843ae63ab1284d8b9186fbb8a0f2fb8` | `202f0f23b9f84866d02e109da3be08d1b2d51f4ef240e84170a3330de5e4593e` |
| `TDBRAIN_STAGE4B_CHARACTERIZATION_DESIGN.py` | 14 | `bfd08953faf5999567719f1a60e22bb3f1ff6e8fab885532c654937bfdb2593c` | `b37a3f620afda4a3d534b853cbbcf67f289602f381fd91927b01197774ec9286` |
| `TDBRAIN_STAGE4C_HELDOUT_ASSOCIATIONS.py` | 15 | `1e84a30eb2829ae0736246ffea8eccc7bc72696776a7e72e6b86db9db7ac4f9c` | `e0659548c910c76d0cbe45968292fffe0d8e74c938d0dcc6ff12ce95cdff8ef4` |
| `TDBRAIN_STAGE4D_BDI_POSTHOC_FEASIBILITY.py` | 16 | `c49cfef6edd103cee3d22500763e534b5acbb4d23084867fc9f60bffcf25da12` | `7a661594db095270060c1eb9643ee3c8cd43bb4768385a74dbba277ffd6d8f85` |
| `TDBRAIN_STAGE4E_BDI_PAIRWISE_POSTHOC.py` | 26 | `e6372c02f8e0ce7318ed28472802680d121f549a9250ffbaf1b000fea151e93a` | `2a6ee671afbd458eda63874db80e2bb5465f4fab835a824674acf30af1a55859` |

## External dependencies and execution limitations

The code was written for Google Colab and uses paths under
`/content/drive/MyDrive/`.

It expects previously generated experiment directories,
frozen source files, subject-alignment information,
training artifacts, and stage-specific SHA-256 locks.

Major Python packages imported by these scripts include
NumPy, pandas, SciPy, scikit-learn, umap-learn, and PyTorch.
Google Colab integration and Excel-file support are also
used by relevant stages.

Package versions and a completely portable environment
have not been established by this recovered-source package.

The repository does not provide the raw EEG recordings,
individual-level NEO-FFI responses, restricted clinical
metadata, or all intermediate experimental artifacts.
Dataset access is subject to the applicable data terms.

Consequently, these scripts are provided for methodological
transparency and inspection of the archived analysis.
A fresh end-to-end reproduction requires the specified
upstream inputs and a compatible execution environment.

## Upstream Stage 7G source limitation

The original Stage 7G EEG-feature generator was recovered
separately from `external_validation_1.ipynb`.

The saved original notebook cell contains an unconditional
CUDA device-name lookup and records a CPU-related failure.
A separate CPU-compatible review copy removes only this
redundant diagnostic printing statement.

Neither the failed recovered generator nor its unverified
CPU-compatible review copy is included in this directory
as a successfully executed original source file.

The three retained Stage 7G artifacts used for provenance
checking passed SHA-256 verification, including the
86-by-128 EEG feature matrix and equal-window selection.

This artifact verification does not establish that the
currently recovered Stage 7G source cell ran successfully.

## Reported evaluation

The corresponding manuscript and supplementary material
report the primary TDBRAIN result for N=85 and K=6:

- Cluster sizes: 27, 15, 18, 10, 9, 6
- Silhouette: 0.4654
- Davies-Bouldin index: 0.5984
- Calinski-Harabasz index: 163.46

The primary clinical characterization is subsequent to
cluster selection. Diagnosis and symptom variables were
not used to define the primary fused-subject features.

These reported numerical results have not been freshly
recomputed by staging the code in this repository.
