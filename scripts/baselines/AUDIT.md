# Baseline package audit

## Manuscript cross-check

Current manuscript mapping:

- Table 28: native DBSCAN sensitivity
- Table 29: adapted EEG benchmark
- Table 30: tuned conventional clustering
- Table 31: DEC/IDEC saved-latent diagnostics
- Table 32: deep-clustering comparison

Table numbering changed across earlier drafts; mapping is based on current
table titles and numerical rows rather than older filenames alone.

## Result verification

- Table 28: verified against both selected DBSCAN aggregate files.
- Table 29: exact consolidated aggregate file not yet found. Jiman, Pan, Leal,
  and native Du/DBSCAN implementations were traced, but the new native-DBSCAN
  row must be assembled into a fresh checked five-method table.
- Table 30: all nine representative rows verified after rounding.
- Table 31: DEC and IDEC values verified; the manuscript effective-rank column
  corresponds to participation-ratio effective rank.
- Table 32: DEC, IDEC, DeepCluster, VAE-GMM, and GAE-KMeans rows verified. The
  old combined CSV's EPG-SD row differs from the archived final model, so it is
  excluded from the baseline script output.

## Important implementation caveats

1. Native DBSCAN is allowed to choose its native cluster count and leave noise
   unassigned. The archived search found K=2 with partial coverage; no eligible
   non-degenerate K=5 solution was found under the predefined rules.
2. The Pan and Leal adaptations are methodologically aligned reimplementations
   on HBN subject features, not reproductions of scores from their original
   datasets.
3. The archived deep-clustering run reconstructed its 196-dimensional input by
   concatenating the 192-dimensional EEG block with phenotype columns in the
   order p-factor, attention, internalizing, externalizing, then standardizing.
   This order is preserved for result provenance, although it differs from the
   phenotype ordering used in the archived core graph node-feature matrix.
4. DEC/IDEC diagnostics are single-run saved-output checks. They do not prove
   multi-seed stability, assignment confidence, or reconstruction preservation.
5. Clinical variables are evaluated after baseline selection and are not used
   to select tuned conventional-clustering hyperparameters.

## Public-release changes

Private machine paths, internal editorial wording, patched manuscript rows, and
absolute paths in manifests were removed. Numerical algorithms linked to the
verified baseline rows were otherwise preserved. No data or generated outputs
are bundled.
