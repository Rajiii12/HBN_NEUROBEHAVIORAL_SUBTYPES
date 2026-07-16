# EPG-SD selected core scripts: Stage 00 through Stage 06

This folder contains the workflow from input preflight auditing through the 192-dimensional subject-level EEG representation.

## Included stages

- `00_sampling_rate_and_input_shape_audit.py`: supporting preflight utility.
- `01_preprocessing.ipynb`: 500 Hz, 1--40 Hz FIR, 60 Hz notch, average reference, FastICA/ICLabel, channel-wise z-score, 1 s windows with 50% overlap, 4-SD window rejection, and at least 10 accepted windows.
- `02_band_decomposition.py`: delta 1--4, theta 4--8, alpha 8--13, beta 13--30, gamma 30--40 Hz.
- `03_embedding_stream_1.py` and `04_embedding_stream_2.py`: two separately saved, same-architecture frequency-wise Conv1D autoencoder streams; each contributes five 128-dimensional band embeddings.
- `05_ten_view_fusion_pca64.py`: separate view standardization, 1280-dimensional concatenation, PCA to 64.
- `06_subject_aggregation.py`: mean + standard deviation + median, producing 192 EEG features per subject.

## Scientific status

This manuscript-aligned implementation covers Stages 00–06. The gamma upper bound is 40 Hz and preprocessing includes explicit resampling to 500 Hz. Numerical equivalence to archived outputs requires rerunning the pipeline and revalidating the generated results.

The two embedding files implement the same architecture in separate saved streams. They are two saved streams of the same model architecture rather than different model families.

Graph construction, VGAE, UMAP, spectral clustering, validation, ablations, baselines, and figure-generation code are outside this folder.

## Local paths

Set paths with environment variables:

```bash
export EPGSD_RAW_BIDS_DIR=/path/to/ds005516
export EPGSD_PREPROCESSED_DIR=/path/to/preprocessed_HBN_v2
export EPGSD_PARTICIPANTS_FILE=/path/to/participants.xlsx
export EPGSD_MASTER_METADATA=/path/to/master_metadata.csv
```
