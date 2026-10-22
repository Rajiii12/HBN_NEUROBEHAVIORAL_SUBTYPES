#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn


# ======================================================================
# PATHS
# ======================================================================

DRIVE = Path(
    "/content/drive/MyDrive/FULL_23PHD1241_BACKUP_2026_07_04"
)

EXT = (
    DRIVE
    / "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
    / "EXTERNAL_VALIDATION_MIPDB"
)

AUDIT = EXT / "00_AUDIT"

STAGE3 = (
    EXT
    / "04_FEATURES"
    / "MIPDB_HBN_FROZEN_EMBEDDINGS_PRIMARY91"
)

S1 = STAGE3 / "stream1_dynamic"
S2 = STAGE3 / "stream2_cnn"

ORDER_FILE = (
    STAGE3
    / "MIPDB_STAGE3_RECORDING_ORDER.csv"
)

STAGE3_MANIFEST = (
    AUDIT
    / "MIPDB_STAGE3_FROZEN_HBN_EMBEDDING_MANIFEST.csv"
)

TRANSFORM_ROOT = (
    EXT
    / "05_FROZEN_HBN_ARTIFACTS"
    / "hbn_fitted_transforms"
)

SCALER_FILE = (
    TRANSFORM_ROOT
    / "HBN_TEN_VIEW_STANDARD_SCALERS.joblib"
)

PCA_FILE = (
    TRANSFORM_ROOT
    / "HBN_PCA64_HBN_ONLY_FITTED.joblib"
)

CONFIG_FILE = (
    TRANSFORM_ROOT
    / "HBN_TRANSFORM_CONFIG.json"
)

EQUIV_FILE = (
    TRANSFORM_ROOT
    / "HBN_TRANSFORM_EQUIVALENCE_AUDIT.json"
)

HASH_FILE = (
    TRANSFORM_ROOT
    / "HBN_FITTED_TRANSFORMS_SHA256.txt"
)

HBN_AGG_SCRIPT = (
    DRIVE
    / "NEW_TASK_MULTIVIEW_V2"
    / "TECHNICAL_FIX_SPECTRAL_UMAP"
    / "find_exact_recording_metadata_for_branch_ablation.py"
)

OUT = (
    EXT
    / "04_FEATURES"
    / "MIPDB_HBN_FROZEN_TRANSFORM_PRIMARY91"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BANDS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]

EXPECTED_RECORDINGS = 825
EXPECTED_SUBJECTS = 91
VIEW_DIM = 128
PCA_DIM = 64
SUBJECT_DIM = 192


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def parse_hash_manifest(path: Path):

    out = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            None,
            1,
        )

        if len(parts) != 2:
            continue

        digest, name = parts

        out[
            Path(name).name
        ] = digest

    return out


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 4 — FROZEN HBN SCALER + PCA64 + SUBJECT192")
print("NO MIPDB FITTING")
print("NO MODEL TRAINING")
print("=" * 78)


# ======================================================================
# 1. VERIFY FROZEN HBN TRANSFORM LOCK
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY FROZEN HBN TRANSFORMS")
print("=" * 78)

required = [
    SCALER_FILE,
    PCA_FILE,
    CONFIG_FILE,
    EQUIV_FILE,
    HASH_FILE,
]

for p in required:

    if not p.exists():
        raise FileNotFoundError(p)

    print(
        "FOUND =",
        p.name,
    )


expected_hashes = parse_hash_manifest(
    HASH_FILE
)

for p in [
    SCALER_FILE,
    PCA_FILE,
    CONFIG_FILE,
    EQUIV_FILE,
]:

    expected = expected_hashes.get(
        p.name
    )

    if expected is None:
        raise RuntimeError(
            f"No SHA256 entry for {p.name}"
        )

    got = sha256_file(p)

    if got != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {p.name}\n"
            f"EXPECTED={expected}\n"
            f"GOT={got}"
        )

    print(
        p.name,
        "SHA256=PASS",
    )


config = json.loads(
    CONFIG_FILE.read_text(
        encoding="utf-8"
    )
)

equiv = json.loads(
    EQUIV_FILE.read_text(
        encoding="utf-8"
    )
)

if equiv.get("status") != "PASS":
    raise RuntimeError(
        "Frozen HBN transform equivalence audit is not PASS."
    )

if config.get(
    "external_dataset_used_for_fit"
) is not False:
    raise RuntimeError(
        "Frozen transform config does not certify HBN-only fitting."
    )


expected_order = [
    "delta_dynamic",
    "theta_dynamic",
    "alpha_dynamic",
    "beta_dynamic",
    "gamma_dynamic",
    "delta_cnn",
    "theta_cnn",
    "alpha_cnn",
    "beta_cnn",
    "gamma_cnn",
]

if config.get(
    "view_order"
) != expected_order:
    raise RuntimeError(
        "Frozen HBN view order mismatch."
    )


print(
    "HBN_TRANSFORM_EQUIVALENCE = PASS"
)

print(
    "EXTERNAL_DATA_USED_FOR_FIT = NO"
)

print(
    "FROZEN_VIEW_ORDER = PASS"
)


# ======================================================================
# 2. VERIFY STAGE-3 RECORDING ORDER
# ======================================================================

print()
print("=" * 78)
print("2. LOAD AUTHORITATIVE STAGE-3 ORDER")
print("=" * 78)

order = pd.read_csv(
    ORDER_FILE
)

print(
    "ORDER_ROWS =",
    len(order),
)

print(
    "UNIQUE_SUBJECTS =",
    order[
        "subject"
    ].nunique(),
)

if len(order) != EXPECTED_RECORDINGS:

    raise RuntimeError(
        f"Expected {EXPECTED_RECORDINGS} rows, "
        f"found {len(order)}"
    )


if order[
    "subject"
].nunique() != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Unexpected Stage-3 subject count."
    )


if "stage3_row" not in order.columns:

    raise RuntimeError(
        "stage3_row column missing."
    )


expected_rows = np.arange(
    EXPECTED_RECORDINGS
)

if not np.array_equal(
    order["stage3_row"].to_numpy(),
    expected_rows,
):

    raise RuntimeError(
        "Stage-3 row order is not sequential."
    )


# ======================================================================
# 3. VERIFY STAGE-3 EMBEDDING MATRICES
# ======================================================================

print()
print("=" * 78)
print("3. VERIFY TEN STAGE-3 EMBEDDING MATRICES")
print("=" * 78)

stage3_manifest = pd.read_csv(
    STAGE3_MANIFEST
)

if len(stage3_manifest) != 10:

    raise RuntimeError(
        f"Expected 10 Stage-3 output rows, "
        f"found {len(stage3_manifest)}"
    )


arrays = {}

for name in expected_order:

    band, stream = name.rsplit(
        "_",
        1,
    )

    if stream == "dynamic":

        p = (
            S1
            / f"{band}_embedding.npy"
        )

    elif stream == "cnn":

        p = (
            S2
            / f"{band}_cnn_embedding_fixed.npy"
        )

    else:

        raise RuntimeError(
            f"Unknown stream {stream}"
        )


    if not p.exists():

        raise FileNotFoundError(p)


    arr = np.load(
        p,
        allow_pickle=False,
    )


    if arr.shape != (
        EXPECTED_RECORDINGS,
        VIEW_DIM,
    ):

        raise RuntimeError(
            f"{name}: invalid shape {arr.shape}"
        )


    if not np.isfinite(
        arr
    ).all():

        raise RuntimeError(
            f"{name}: non-finite values"
        )


    manifest_row = stage3_manifest[
        stage3_manifest[
            "path"
        ] == str(p)
    ]

    if len(manifest_row) != 1:

        raise RuntimeError(
            f"{name}: Stage-3 manifest row not uniquely found."
        )


    expected_sha = str(
        manifest_row.iloc[0][
            "sha256"
        ]
    )

    got_sha = sha256_file(p)

    if got_sha != expected_sha:

        raise RuntimeError(
            f"{name}: Stage-3 output SHA mismatch."
        )


    arrays[name] = arr.astype(
        np.float32,
        copy=False,
    )


    print(
        f"{name:20s}",
        "SHAPE=",
        arr.shape,
        "SHA256=PASS",
    )


# ======================================================================
# 4. LOAD FROZEN HBN SCALERS + PCA
# ======================================================================

print()
print("=" * 78)
print("4. LOAD FROZEN HBN SCALERS AND PCA64")
print("=" * 78)

scalers = joblib.load(
    SCALER_FILE
)

pca = joblib.load(
    PCA_FILE
)


if set(
    scalers.keys()
) != set(
    expected_order
):

    raise RuntimeError(
        "Frozen scaler key set mismatch."
    )


print(
    "SCALER_COUNT =",
    len(scalers),
)

print(
    "PCA_COMPONENTS =",
    pca.n_components_,
)

print(
    "SCIKIT_LEARN_RUNTIME =",
    sklearn.__version__,
)


# ======================================================================
# 5. APPLY HBN-FITTED SCALERS
# ======================================================================

print()
print("=" * 78)
print("5. APPLY FROZEN HBN SCALERS — TRANSFORM ONLY")
print("=" * 78)

scaled = []

view_stats = []

for name in expected_order:

    x = arrays[name]

    scaler = scalers[name]

    # IMPORTANT:
    # transform() only.
    # No fit(), no fit_transform().
    z = scaler.transform(
        x
    ).astype(
        np.float32,
        copy=False,
    )


    if z.shape != (
        EXPECTED_RECORDINGS,
        VIEW_DIM,
    ):

        raise RuntimeError(
            f"{name}: scaler output shape failure"
        )


    if not np.isfinite(
        z
    ).all():

        raise RuntimeError(
            f"{name}: scaler produced non-finite values"
        )


    scaled.append(z)


    view_stats.append({
        "view": name,
        "recordings":
            int(z.shape[0]),
        "dimensions":
            int(z.shape[1]),
        "mean":
            float(z.mean()),
        "std":
            float(z.std()),
        "min":
            float(z.min()),
        "max":
            float(z.max()),
        "finite":
            True,
    })


    print(
        f"{name:20s}",
        "TRANSFORM=PASS",
        "MEAN=",
        float(z.mean()),
        "STD=",
        float(z.std()),
    )


concat = np.concatenate(
    scaled,
    axis=1,
)


if concat.shape != (
    EXPECTED_RECORDINGS,
    1280,
):

    raise RuntimeError(
        f"Unexpected concatenated shape: {concat.shape}"
    )


print()
print(
    "CONCAT_SHAPE =",
    concat.shape,
)


# ======================================================================
# 6. APPLY FROZEN HBN PCA64
# ======================================================================

print()
print("=" * 78)
print("6. APPLY FROZEN HBN PCA64 — TRANSFORM ONLY")
print("=" * 78)

recording64 = pca.transform(
    concat
).astype(
    np.float32,
    copy=False,
)


if recording64.shape != (
    EXPECTED_RECORDINGS,
    PCA_DIM,
):

    raise RuntimeError(
        f"Unexpected PCA shape {recording64.shape}"
    )


if not np.isfinite(
    recording64
).all():

    raise RuntimeError(
        "PCA64 produced non-finite values."
    )


print(
    "RECORDING_PCA64_SHAPE =",
    recording64.shape,
)

print(
    "RECORDING_PCA64_MEAN =",
    float(
        recording64.mean()
    ),
)

print(
    "RECORDING_PCA64_STD =",
    float(
        recording64.std()
    ),
)

print(
    "RECORDING_PCA64_MIN =",
    float(
        recording64.min()
    ),
)

print(
    "RECORDING_PCA64_MAX =",
    float(
        recording64.max()
    ),
)


# ======================================================================
# 7. SAVE RECORDING-LEVEL PCA64
# ======================================================================

recording_file = (
    OUT
    / "MIPDB_RECORDING_PCA64_FROZEN_HBN.npy"
)

np.save(
    recording_file,
    recording64,
    allow_pickle=False,
)


# ======================================================================
# 8. USE EXACT HBN SUBJECT AGGREGATION FUNCTION
# ======================================================================

print()
print("=" * 78)
print("7. AGGREGATE RECORDINGS -> SUBJECT192")
print("=" * 78)

spec = importlib.util.spec_from_file_location(
    "hbn_aggregation",
    HBN_AGG_SCRIPT,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)


subject_values = mod.normalize_subject_values(
    order[
        "subject"
    ]
)


subject192, subject_ids = mod.aggregate_subjects(
    recording64,
    subject_values,
)


if subject192.shape != (
    EXPECTED_SUBJECTS,
    SUBJECT_DIM,
):

    raise RuntimeError(
        f"Unexpected subject representation "
        f"shape {subject192.shape}"
    )


if not np.isfinite(
    subject192
).all():

    raise RuntimeError(
        "Subject192 contains non-finite values."
    )


print(
    "SUBJECT192_SHAPE =",
    subject192.shape,
)

print(
    "SUBJECT_COUNT =",
    len(subject_ids),
)

print(
    "SUBJECT192_FINITE =",
    bool(
        np.isfinite(
            subject192
        ).all()
    ),
)


# ======================================================================
# 9. SUBJECT METADATA
# ======================================================================

subject_rows = []

normalized_subject_series = pd.Series(
    subject_values
)


for sid in subject_ids:

    mask = (
        normalized_subject_series
        == sid
    ).to_numpy()


    rows = order.loc[
        mask
    ]


    subject_rows.append({
        "subject":
            sid,
        "n_recordings":
            int(mask.sum()),
        "blocks":
            "|".join(
                rows[
                    "block"
                ].astype(
                    str
                ).tolist()
            ),
    })


subject_meta = pd.DataFrame(
    subject_rows
)


if len(subject_meta) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Subject metadata row count failure."
    )


subject_file = (
    OUT
    / "MIPDB_SUBJECT_EEG192_FROZEN_HBN.npy"
)

subject_meta_file = (
    OUT
    / "MIPDB_SUBJECT_METADATA.csv"
)


np.save(
    subject_file,
    subject192.astype(
        np.float32
    ),
    allow_pickle=False,
)

subject_meta.to_csv(
    subject_meta_file,
    index=False,
)


# ======================================================================
# 10. AUDIT STATISTICS
# ======================================================================

print()
print("=" * 78)
print("8. FINAL STAGE-4 AUDIT")
print("=" * 78)

per_dim_std_recording = np.std(
    recording64,
    axis=0,
)

per_dim_std_subject = np.std(
    subject192,
    axis=0,
)


active_recording_dims = int(
    np.sum(
        per_dim_std_recording
        > 1e-8
    )
)

active_subject_dims = int(
    np.sum(
        per_dim_std_subject
        > 1e-8
    )
)


print(
    "RECORDINGS =",
    recording64.shape[0],
)

print(
    "RECORDING_DIMENSIONS =",
    recording64.shape[1],
)

print(
    "ACTIVE_RECORDING_PCA_DIMENSIONS =",
    active_recording_dims,
)

print(
    "SUBJECTS =",
    subject192.shape[0],
)

print(
    "SUBJECT_DIMENSIONS =",
    subject192.shape[1],
)

print(
    "ACTIVE_SUBJECT_DIMENSIONS =",
    active_subject_dims,
)

print(
    "MIN_RECORDINGS_PER_SUBJECT =",
    int(
        subject_meta[
            "n_recordings"
        ].min()
    ),
)

print(
    "MAX_RECORDINGS_PER_SUBJECT =",
    int(
        subject_meta[
            "n_recordings"
        ].max()
    ),
)

print(
    "MEAN_RECORDINGS_PER_SUBJECT =",
    float(
        subject_meta[
            "n_recordings"
        ].mean()
    ),
)


# ======================================================================
# 11. SAVE SUMMARY
# ======================================================================

pd.DataFrame(
    view_stats
).to_csv(
    AUDIT
    / "MIPDB_STAGE4_HBN_SCALER_VIEW_STATS.csv",
    index=False,
)


summary = {
    "recordings":
        int(recording64.shape[0]),

    "subjects":
        int(subject192.shape[0]),

    "input_views":
        10,

    "input_dimension_per_view":
        128,

    "concatenated_dimension":
        1280,

    "pca_dimension":
        64,

    "subject_dimension":
        192,

    "subject_aggregation":
        [
            "mean",
            "std",
            "median",
        ],

    "active_recording_pca_dimensions":
        active_recording_dims,

    "active_subject_dimensions":
        active_subject_dims,

    "recording_pca64_sha256":
        sha256_file(
            recording_file
        ),

    "subject_eeg192_sha256":
        sha256_file(
            subject_file
        ),

    "subject_metadata_sha256":
        sha256_file(
            subject_meta_file
        ),

    "hbn_scaler_file_sha256":
        sha256_file(
            SCALER_FILE
        ),

    "hbn_pca_file_sha256":
        sha256_file(
            PCA_FILE
        ),

    "mipdb_used_for_scaler_fit":
        False,

    "mipdb_used_for_pca_fit":
        False,

    "mipdb_used_for_model_training":
        False,

    "status":
        "PASS",
}


summary_file = (
    AUDIT
    / "MIPDB_STAGE4_FROZEN_HBN_TRANSFORM_SUMMARY.json"
)

summary_file.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ======================================================================
# 12. PERMANENT LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE4_FROZEN_HBN_TRANSFORM_LOCK.txt"
)

lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 4 LOCK
======================================================================

INPUT_RECORDINGS=825
INPUT_SUBJECTS=91

INPUT_VIEWS=10
DIMENSION_PER_VIEW=128
CONCATENATED_DIMENSION=1280

SCALERS=HBN_FITTED_FROZEN
PCA=HBN_FITTED_FROZEN
PCA_COMPONENTS=64

MIPDB_USED_FOR_SCALER_FIT=NO
MIPDB_USED_FOR_PCA_FIT=NO
MIPDB_USED_FOR_MODEL_TRAINING=NO

SUBJECT_AGGREGATION=mean,std,median
SUBJECT_FEATURE_DIMENSION=192

RECORDING_PCA64_FILE=
{recording_file}

SUBJECT_EEG192_FILE=
{subject_file}

SUBJECT_METADATA_FILE=
{subject_meta_file}

RECORDING_PCA64_SHA256=
{summary["recording_pca64_sha256"]}

SUBJECT_EEG192_SHA256=
{summary["subject_eeg192_sha256"]}

SUBJECT_METADATA_SHA256=
{summary["subject_metadata_sha256"]}

ACTIVE_RECORDING_PCA_DIMENSIONS=
{active_recording_dims}

ACTIVE_SUBJECT_DIMENSIONS=
{active_subject_dims}

STAGE4_STATUS=PASS
======================================================================
""",
    encoding="utf-8",
)


print()
print("=" * 78)
print("STAGE 4 COMPLETE")
print("=" * 78)

print(
    "RECORDING_PCA64_SHAPE =",
    recording64.shape,
)

print(
    "SUBJECT_EEG192_SHAPE =",
    subject192.shape,
)

print(
    "MIPDB_USED_FOR_SCALER_FIT=NO"
)

print(
    "MIPDB_USED_FOR_PCA_FIT=NO"
)

print(
    "MIPDB_USED_FOR_MODEL_TRAINING=NO"
)

print(
    "STAGE4_STATUS=PASS"
)

print()
print(
    "OUTPUT_DIR =",
    OUT,
)

print(
    "LOCK_FILE =",
    lock_file,
)

print("=" * 78)
