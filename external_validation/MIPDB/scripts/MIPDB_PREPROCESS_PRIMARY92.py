import os
import re
import csv
import json
import time
import hashlib
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mne

from mne.preprocessing import ICA
from mne_icalabel import label_components


# ======================================================================
# PATHS
# ======================================================================

DRIVE = Path(
    "/content/drive/MyDrive/"
    "FULL_23PHD1241_BACKUP_2026_07_04/"
    "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
)

EXT = DRIVE / "EXTERNAL_VALIDATION_MIPDB"

AUDIT = EXT / "00_AUDIT"

COHORT_FILE = (
    AUDIT
    / "MIPDB_PRIMARY_AGE_MATCHED_SUBJECTS.csv"
)

OUT = (
    EXT
    / "03_PREPROCESSED"
    / "MIPDB_HBN_EXACT_PRIMARY92"
)

META_DIR = OUT / "metadata"

MANIFEST = (
    AUDIT
    / "MIPDB_PREPROCESSING_MANIFEST.csv"
)

CHANNEL_AUDIT = (
    AUDIT
    / "MIPDB_DEAD_BAD_CHANNEL_AUDIT.txt"
)

SUMMARY = (
    AUDIT
    / "MIPDB_PREPROCESSING_SUMMARY.txt"
)

SELECTED_SUBJECTS_TXT = (
    AUDIT
    / "MIPDB_SELECTED_PRIMARY92_SUBJECTS.txt"
)

LOCAL_RAW_ROOT = Path(
    "/content/MIPDB_ROLLING_RAW"
)

LOCAL_OUT_ROOT = Path(
    "/content/MIPDB_ROLLING_OUTPUT"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

META_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOCAL_RAW_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

LOCAL_OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# DATA SOURCE
# ======================================================================

BASE_URL = (
    "https://data.nemar.org/"
    "nm000153/v1.0.0"
)


# ======================================================================
# PRIMARY BLOCKS — LOCKED BEFORE OUTCOME ANALYSIS
# ======================================================================

BLOCKS = {
    "block01": "RestingState",
    "block03": "SymbolSearch",
    "block04": "SurroundSuppression_Block1",
    "block05": "ContrastChange_Block1",
    "block06": "NaturalisticViewing",
    "block07": "ContrastChange_Block2",
    "block08": "NaturalisticViewing",
    "block09": "ContrastChange_Block3",
    "block10": "NaturalisticViewing",
    "block11": "SurroundSuppression_Block2",
}

EXPECTED_SUBJECTS = 92


# ======================================================================
# EXACT HBN PRODUCTION PREPROCESSING CONSTANTS
# ======================================================================

SFREQ_TARGET = 500.0

L_FREQ = 1.0
H_FREQ = 40.0

NOTCH_FREQ = 60.0

WIN_SAMPLES = 500
STRIDE_SAMPLES = 250

AMPLITUDE_THRESH = 4.0

N_ICA_COMPONENTS = 20

DEAD_CHAN_THRESH = 1e-10

ICA_MAX_ITER = 400
ICA_RANDOM_STATE = 42

ICLABEL_THRESHOLD = 0.70

ICLABEL_EXCLUDE_LABELS = [
    "eye blink",
    "muscle artifact",
    "heart beat",
    "line noise",
]


# ======================================================================
# MANIFEST FIELDS
# ======================================================================

MANIFEST_COLUMNS = [
    "subject",
    "block",
    "task_family",
    "status",
    "input_total_channels",
    "bids_eeg_channels",
    "bids_bad_channels_excluded",
    "mne_bad_channels_excluded",
    "missing_bids_eeg_channels",
    "dead_channels_removed",
    "n_channels_after_dead_removal",
    "zero_padded_channels",
    "trimmed_channels",
    "ica_components",
    "ica_excluded_count",
    "ica_excluded_indices",
    "iclabel_fallback",
    "candidate_windows",
    "retained_windows",
    "final_shape",
    "sha256",
    "elapsed_seconds",
    "error",
    "timestamp_utc",
]


# ======================================================================
# HELPERS
# ======================================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    if x.lower() == "nan":
        return ""
    return x.strip()


def normalise_subject(x):
    x = clean_text(x)

    if not x:
        return None

    if x.startswith("sub-"):
        core = x[4:]
    else:
        core = x

    m = re.search(
        r"(A\d+)",
        core,
        flags=re.IGNORECASE
    )

    if not m:
        return None

    return "sub-" + m.group(1).upper()


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def write_text_atomic(
    path,
    text
):
    path = Path(path)

    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8"
    )

    os.replace(
        tmp,
        path
    )


def write_json_atomic(
    path,
    obj
):
    path = Path(path)

    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2
        ),
        encoding="utf-8"
    )

    os.replace(
        tmp,
        path
    )


def format_channels(x):
    if not x:
        return "NONE"

    return ",".join(
        map(
            str,
            x
        )
    )


# ======================================================================
# LOAD AND VERIFY LOCKED 92-SUBJECT COHORT
# ======================================================================

if not COHORT_FILE.exists():

    raise FileNotFoundError(
        f"Locked cohort file missing: "
        f"{COHORT_FILE}"
    )


cohort = pd.read_csv(
    COHORT_FILE
)


best_col = None
best_ids = []

for col in cohort.columns:

    ids = []

    for value in cohort[col]:

        sid = normalise_subject(
            value
        )

        if sid:
            ids.append(
                sid
            )

    ids = list(
        dict.fromkeys(
            ids
        )
    )

    if len(ids) > len(
        best_ids
    ):
        best_col = col
        best_ids = ids


subjects = best_ids


if len(subjects) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "PRIMARY COHORT LOCK FAILURE: "
        f"expected {EXPECTED_SUBJECTS} "
        f"subjects but detected "
        f"{len(subjects)} using column "
        f"{best_col!r}. "
        "STOPPING WITHOUT PREPROCESSING."
    )


selected_lines = [
    "EPG-SD MIPDB PRIMARY COHORT",
    "=" * 70,
    f"Source CSV: {COHORT_FILE}",
    f"Detected subject column: {best_col}",
    f"TOTAL_SELECTED_SUBJECTS={len(subjects)}",
    "",
]

selected_lines.extend(
    subjects
)

write_text_atomic(
    SELECTED_SUBJECTS_TXT,
    "\n".join(
        selected_lines
    )
    + "\n"
)


print(
    "=" * 72
)

print(
    "PRIMARY COHORT LOCK PASS"
)

print(
    "=" * 72
)

print(
    "Subjects =",
    len(subjects)
)

print(
    "Blocks =",
    len(BLOCKS)
)

print(
    "Maximum subject-block pairs =",
    len(subjects)
    * len(BLOCKS)
)

print(
    "Subject column =",
    best_col
)


# ======================================================================
# MANIFEST MANAGEMENT
# ======================================================================

def load_manifest():

    if MANIFEST.exists():

        try:

            df = pd.read_csv(
                MANIFEST,
                dtype=str
            ).fillna("")

        except Exception:

            df = pd.DataFrame(
                columns=MANIFEST_COLUMNS
            )

    else:

        df = pd.DataFrame(
            columns=MANIFEST_COLUMNS
        )

    for col in MANIFEST_COLUMNS:

        if col not in df.columns:
            df[col] = ""

    return df[
        MANIFEST_COLUMNS
    ]


def upsert_manifest(
    record
):

    df = load_manifest()

    subject = str(
        record.get(
            "subject",
            ""
        )
    )

    block = str(
        record.get(
            "block",
            ""
        )
    )

    if len(df):

        mask = ~(
            (
                df["subject"]
                == subject
            )
            &
            (
                df["block"]
                == block
            )
        )

        df = df[
            mask
        ].copy()

    row = {
        col:
        str(
            record.get(
                col,
                ""
            )
        )
        for col
        in MANIFEST_COLUMNS
    }

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [row]
            )
        ],
        ignore_index=True
    )

    df = df.sort_values(
        [
            "subject",
            "block"
        ]
    )

    tmp = Path(
        str(MANIFEST)
        + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False
    )

    os.replace(
        tmp,
        MANIFEST
    )


def refresh_reports():

    df = load_manifest()

    possible = (
        EXPECTED_SUBJECTS
        * len(BLOCKS)
    )

    attempted = len(df)

    passed = int(
        (
            df["status"]
            == "PASS"
        ).sum()
    )

    unavailable = int(
        (
            df["status"]
            == "DOWNLOAD_FAILED_OR_MISSING"
        ).sum()
    )

    failed = int(
        (
            df["status"]
            == "PREPROCESSING_FAILED"
        ).sum()
    )

    dead_mask = (
        (
            df["status"]
            == "PASS"
        )
        &
        (
            ~df[
                "dead_channels_removed"
            ].isin(
                [
                    "",
                    "NONE"
                ]
            )
        )
    )

    bad_mask = (
        (
            df["status"]
            == "PASS"
        )
        &
        (
            ~df[
                "bids_bad_channels_excluded"
            ].isin(
                [
                    "",
                    "NONE"
                ]
            )
        )
    )

    dead_count = int(
        dead_mask.sum()
    )

    bad_count = int(
        bad_mask.sum()
    )

    pass_subjects = int(
        df.loc[
            df["status"]
            == "PASS",
            "subject"
        ].nunique()
    )

    summary = [
        "EPG-SD MIPDB PREPROCESSING SUMMARY",
        "=" * 72,
        f"UPDATED_UTC={utc_now()}",
        "",
        f"PRIMARY_SUBJECTS={EXPECTED_SUBJECTS}",
        f"PRIMARY_BLOCKS={len(BLOCKS)}",
        f"MAXIMUM_SUBJECT_BLOCK_PAIRS={possible}",
        f"PAIRS_ATTEMPTED_OR_CHECKPOINTED={attempted}",
        f"PASS_RECORDINGS={passed}",
        f"SUBJECTS_WITH_AT_LEAST_ONE_PASS_RECORDING={pass_subjects}",
        f"DOWNLOAD_FAILED_OR_MISSING={unavailable}",
        f"PREPROCESSING_FAILED={failed}",
        f"PASS_RECORDINGS_WITH_DEAD_CHANNEL_REMOVAL={dead_count}",
        f"PASS_RECORDINGS_WITH_BIDS_BAD_CHANNEL_EXCLUSION={bad_count}",
        "",
        "PRIMARY_FLAT_CHANNEL_METHOD=DROP_THEN_ZERO_PAD",
        "INTERPOLATION=DISABLED",
        "",
        (
            "RUN_PAIR_SCAN_COMPLETE="
            + (
                "YES"
                if attempted
                == possible
                else "NO"
            )
        ),
    ]

    write_text_atomic(
        SUMMARY,
        "\n".join(
            summary
        )
        + "\n"
    )


    audit_lines = [
        "EPG-SD MIPDB BAD / DEAD CHANNEL AUDIT",
        "=" * 72,
        f"UPDATED_UTC={utc_now()}",
        "",
        (
            "This file identifies exactly which "
            "MIPDB recording had channels excluded "
            "or removed."
        ),
        "",
        (
            "PRIMARY RULE: bad channels excluded; "
            "flat channels SD < 1e-10 dropped; "
            "missing final channels zero-padded "
            "at END; NO interpolation."
        ),
        "",
    ]


    affected = df[
        (
            df["status"]
            == "PASS"
        )
        &
        (
            (
                ~df[
                    "dead_channels_removed"
                ].isin(
                    [
                        "",
                        "NONE"
                    ]
                )
            )
            |
            (
                ~df[
                    "bids_bad_channels_excluded"
                ].isin(
                    [
                        "",
                        "NONE"
                    ]
                )
            )
            |
            (
                ~df[
                    "mne_bad_channels_excluded"
                ].isin(
                    [
                        "",
                        "NONE"
                    ]
                )
            )
        )
    ]


    if len(
        affected
    ) == 0:

        audit_lines.append(
            "NO AFFECTED PASS RECORDINGS "
            "HAVE BEEN OBSERVED YET."
        )

    else:

        for _, row in affected.iterrows():

            audit_lines.extend(
                [
                    "-" * 72,
                    (
                        f"SUBJECT={row['subject']}"
                    ),
                    (
                        f"BLOCK={row['block']}"
                    ),
                    (
                        "BIDS_BAD_CHANNELS_EXCLUDED="
                        f"{row['bids_bad_channels_excluded']}"
                    ),
                    (
                        "MNE_BAD_CHANNELS_EXCLUDED="
                        f"{row['mne_bad_channels_excluded']}"
                    ),
                    (
                        "DEAD_CHANNELS_REMOVED="
                        f"{row['dead_channels_removed']}"
                    ),
                    (
                        "CHANNELS_AFTER_DEAD_REMOVAL="
                        f"{row['n_channels_after_dead_removal']}"
                    ),
                    (
                        "ZERO_CHANNELS_PADDED_AT_END="
                        f"{row['zero_padded_channels']}"
                    ),
                    "INTERPOLATED_CHANNELS=NONE",
                    "STATUS=PASS",
                ]
            )


    write_text_atomic(
        CHANNEL_AUDIT,
        "\n".join(
            audit_lines
        )
        + "\n"
    )


# ======================================================================
# DOWNLOAD ONE FILE
# ======================================================================

def download_file(
    url,
    destination
):

    destination = Path(
        destination
    )

    if (
        destination.exists()
        and destination.stat().st_size
        > 0
    ):
        return True, ""


    partial = Path(
        str(destination)
        + ".part"
    )

    if partial.exists():
        partial.unlink()


    cmd = [
        "curl",
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "-o",
        str(partial),
        url,
    ]


    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if (
        result.returncode
        != 0
        or not partial.exists()
        or partial.stat().st_size
        == 0
    ):

        if partial.exists():
            partial.unlink()

        return (
            False,
            result.stderr[
                -800:
            ]
        )


    os.replace(
        partial,
        destination
    )

    return True, ""


# ======================================================================
# PREPROCESS ONE RECORDING
# ======================================================================

def preprocess_recording(
    subject,
    block,
    task_family,
    raw_dir,
    output_npy,
    output_json
):

    t0 = time.time()

    stem = (
        f"{subject}_task-{block}"
    )

    vhdr = (
        raw_dir
        / f"{stem}_eeg.vhdr"
    )

    channels_tsv = (
        raw_dir
        / f"{stem}_channels.tsv"
    )


    # ------------------------------------------------------------------
    # 5.1 LOAD RAW EEG
    # HBN: read_raw_eeglab()
    # MIPDB technical format adaptation: read_raw_brainvision()
    # ------------------------------------------------------------------

    raw = (
        mne.io
        .read_raw_brainvision(
            vhdr,
            preload=True,
            verbose="ERROR"
        )
    )

    input_total_channels = len(
        raw.ch_names
    )


    # ------------------------------------------------------------------
    # 5.2 KEEP EEG CHANNELS ONLY, EXCLUDE BADS
    # HBN equivalent:
    # raw.pick_types(eeg=True, exclude="bads")
    # ------------------------------------------------------------------

    channels = pd.read_csv(
        channels_tsv,
        sep="\t"
    )

    eeg_rows = channels[
        channels[
            "type"
        ]
        .astype(str)
        .str.upper()
        .eq("EEG")
    ].copy()


    bids_eeg_names = (
        eeg_rows[
            "name"
        ]
        .astype(str)
        .tolist()
    )


    bids_bad = []

    if (
        "status"
        in eeg_rows.columns
    ):

        status = (
            eeg_rows[
                "status"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        bids_bad = (
            eeg_rows.loc[
                status.eq("bad"),
                "name"
            ]
            .astype(str)
            .tolist()
        )


    mne_bad = [
        ch
        for ch
        in raw.info.get(
            "bads",
            []
        )
        if ch
        in bids_eeg_names
    ]


    excluded_initial = set(
        bids_bad
        + mne_bad
    )


    missing_bids_eeg = [
        ch
        for ch
        in bids_eeg_names
        if ch
        not in raw.ch_names
    ]


    eligible = set(
        bids_eeg_names
    ) - excluded_initial


    # Preserve RAW acquisition order,
    # matching HBN pick_types behavior.
    eeg_names = [
        ch
        for ch
        in raw.ch_names
        if ch
        in eligible
    ]


    if len(
        eeg_names
    ) == 0:

        raise ValueError(
            "No usable EEG channels"
        )


    raw.pick(
        eeg_names
    )


    raw.set_channel_types(
        {
            ch: "eeg"
            for ch
            in raw.ch_names
        },
        verbose=False
    )


    # ------------------------------------------------------------------
    # MIPDB technical metadata adaptation only:
    # attach electrode coordinates for ICLabel.
    # NO INTERPOLATION.
    # ------------------------------------------------------------------

    montage = (
        mne.channels
        .make_standard_montage(
            "GSN-HydroCel-129"
        )
    )


    raw.set_montage(
        montage,
        on_missing="raise",
        verbose=False
    )


    # ------------------------------------------------------------------
    # HBN production data were 500 Hz.
    # MIPDB is expected to be 500 Hz.
    # Do not silently change sampling.
    # ------------------------------------------------------------------

    if not np.isclose(
        float(
            raw.info[
                "sfreq"
            ]
        ),
        SFREQ_TARGET
    ):

        raise ValueError(
            "Unexpected sampling frequency: "
            f"{raw.info['sfreq']} Hz; "
            "expected 500 Hz"
        )


    # ------------------------------------------------------------------
    # 5.3 REMOVE DEAD / FLAT CHANNELS — EXACT HBN LOGIC
    # ------------------------------------------------------------------

    data_all = raw.get_data()

    ch_stds = data_all.std(
        axis=1
    )

    dead_idx = np.where(
        ch_stds
        < DEAD_CHAN_THRESH
    )[0]


    dead_names = [
        raw.ch_names[i]
        for i
        in dead_idx
    ]


    if dead_names:

        raw.drop_channels(
            dead_names
        )


    n_channels = len(
        raw.ch_names
    )


    if n_channels < 64:

        raise ValueError(
            "Too few channels after "
            "dead-channel removal: "
            f"{n_channels}"
        )


    # ------------------------------------------------------------------
    # 5.4 AVERAGE REFERENCE — EXACT HBN
    # ------------------------------------------------------------------

    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False
    )


    # ------------------------------------------------------------------
    # 5.5 1-40 Hz FIR HAMMING — EXACT HBN
    # ------------------------------------------------------------------

    raw.filter(
        L_FREQ,
        H_FREQ,
        method="fir",
        fir_window="hamming",
        verbose=False
    )


    # ------------------------------------------------------------------
    # 5.6 60-Hz NOTCH — EXACT HBN
    # ------------------------------------------------------------------

    raw.notch_filter(
        NOTCH_FREQ,
        verbose=False
    )


    # ------------------------------------------------------------------
    # 5.7 FASTICA — EXACT HBN
    # ------------------------------------------------------------------

    n_ica = min(
        N_ICA_COMPONENTS,
        n_channels - 1
    )


    ica = ICA(
        n_components=n_ica,
        method="fastica",
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
        verbose=False
    )


    ica.fit(
        raw,
        verbose=False
    )


    # ------------------------------------------------------------------
    # ICLABEL — EXACT HBN RULE + ORIGINAL FALLBACK BEHAVIOR
    # ------------------------------------------------------------------

    iclabel_fallback = False
    iclabel_error = ""

    bad_ics = []


    try:

        ic_labels = label_components(
            raw,
            ica,
            method="iclabel"
        )


        labels = list(
            ic_labels[
                "labels"
            ]
        )


        probs = np.asarray(
            ic_labels[
                "y_pred_proba"
            ],
            dtype=float
        )


        bad_ics = [
            idx
            for idx, (
                lbl,
                prob
            )
            in enumerate(
                zip(
                    labels,
                    probs
                )
            )
            if (
                lbl
                in ICLABEL_EXCLUDE_LABELS
                and prob
                > ICLABEL_THRESHOLD
            )
        ]


        ica.exclude = (
            bad_ics
        )


    except Exception as e:

        # This preserves the HBN production
        # script's original fallback behavior.
        iclabel_fallback = True
        iclabel_error = str(e)


    ica.apply(
        raw,
        verbose=False
    )


    # ------------------------------------------------------------------
    # 5.8 CLEANED DATA
    # ------------------------------------------------------------------

    data = raw.get_data()

    n_ch, n_times = (
        data.shape
    )


    # ------------------------------------------------------------------
    # 5.9 GLOBAL Z-SCORE PER CHANNEL — EXACT HBN
    # ------------------------------------------------------------------

    ch_mean = data.mean(
        axis=1,
        keepdims=True
    )

    ch_std = (
        data.std(
            axis=1,
            keepdims=True
        )
        + 1e-8
    )

    data = (
        data
        - ch_mean
    ) / ch_std


    # ------------------------------------------------------------------
    # 5.10 SLIDING WINDOW — EXACT HBN PRODUCTION LOOP
    #
    # IMPORTANT:
    # Production GitHub script used:
    # range(0, n_times - WIN_SAMPLES, STRIDE_SAMPLES)
    #
    # We preserve that exact behavior.
    # ------------------------------------------------------------------

    windows = []

    candidate_windows = 0


    for start in range(
        0,
        n_times
        - WIN_SAMPLES,
        STRIDE_SAMPLES
    ):

        candidate_windows += 1


        w = data[
            :,
            start:
            start
            + WIN_SAMPLES
        ]


        # EXACT HBN:
        # reject if any value
        # exceeds 4 z-score units
        if (
            np.abs(
                w
            ).max()
            > AMPLITUDE_THRESH
        ):

            continue


        windows.append(
            w
        )


    if len(
        windows
    ) < 10:

        raise ValueError(
            "Only "
            f"{len(windows)} "
            "clean windows — "
            "recording too short "
            "or too noisy"
        )


    windows = np.array(
        windows,
        dtype=np.float32
    )


    channels_before_pad = (
        windows.shape[1]
    )

    zero_padded = 0
    trimmed = 0


    # ------------------------------------------------------------------
    # 5.11 PAD / TRIM TO EXACTLY 128 CHANNELS — EXACT HBN
    # ------------------------------------------------------------------

    if (
        windows.shape[1]
        > 128
    ):

        trimmed = (
            windows.shape[1]
            - 128
        )

        windows = (
            windows[
                :,
                :128,
                :
            ]
        )


    elif (
        windows.shape[1]
        < 128
    ):

        zero_padded = (
            128
            - windows.shape[1]
        )


        pad = np.zeros(
            (
                windows.shape[0],
                zero_padded,
                WIN_SAMPLES
            ),
            dtype=np.float32
        )


        windows = np.concatenate(
            [
                windows,
                pad
            ],
            axis=1
        )


    if (
        windows.shape[1:]
        != (
            128,
            500
        )
    ):

        raise RuntimeError(
            "Final tensor has "
            "unexpected shape: "
            f"{windows.shape}"
        )


    # ------------------------------------------------------------------
    # SAVE LOCAL FIRST
    # ------------------------------------------------------------------

    local_npy = (
        LOCAL_OUT_ROOT
        / (
            f"{stem}"
            "_preprocessed.npy"
        )
    )


    with open(
        local_npy,
        "wb"
    ) as f:

        np.save(
            f,
            windows
        )


    local_sha = sha256_file(
        local_npy
    )


    # ------------------------------------------------------------------
    # COPY TO DRIVE AND SHA256 VERIFY
    # ------------------------------------------------------------------

    drive_tmp = Path(
        str(
            output_npy
        )
        + ".partial"
    )


    if drive_tmp.exists():
        drive_tmp.unlink()


    shutil.copy2(
        local_npy,
        drive_tmp
    )


    drive_sha = sha256_file(
        drive_tmp
    )


    if (
        local_sha
        != drive_sha
    ):

        drive_tmp.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "SHA256 mismatch "
            "while copying NPY "
            "to Google Drive"
        )


    os.replace(
        drive_tmp,
        output_npy
    )


    local_npy.unlink(
        missing_ok=True
    )


    elapsed = (
        time.time()
        - t0
    )


    metadata = {
        "status":
            "PASS",

        "subject":
            subject,

        "block":
            block,

        "task_family":
            task_family,

        "source_format":
            "BrainVision_BIDS",

        "preprocessing_rule":
            "HBN_PRODUCTION_EXACT",

        "interpolation":
            False,

        "input_total_channels":
            int(
                input_total_channels
            ),

        "bids_eeg_channels":
            int(
                len(
                    bids_eeg_names
                )
            ),

        "bids_bad_channels_excluded":
            bids_bad,

        "mne_bad_channels_excluded":
            mne_bad,

        "missing_bids_eeg_channels":
            missing_bids_eeg,

        "dead_channel_threshold":
            DEAD_CHAN_THRESH,

        "dead_channels":
            dead_names,

        "n_channels_after_dead_removal":
            int(
                n_channels
            ),

        "zero_padded_channels":
            int(
                zero_padded
            ),

        "trimmed_channels":
            int(
                trimmed
            ),

        "sfreq":
            SFREQ_TARGET,

        "reference":
            "average",

        "bandpass_hz":
            [
                L_FREQ,
                H_FREQ
            ],

        "notch_hz":
            NOTCH_FREQ,

        "ica_components":
            int(
                n_ica
            ),

        "ica_method":
            "fastica",

        "ica_max_iter":
            ICA_MAX_ITER,

        "ica_random_state":
            ICA_RANDOM_STATE,

        "ica_excluded":
            [
                int(x)
                for x
                in bad_ics
            ],

        "iclabel_threshold":
            ICLABEL_THRESHOLD,

        "iclabel_fallback":
            bool(
                iclabel_fallback
            ),

        "iclabel_error":
            iclabel_error,

        "win_samples":
            WIN_SAMPLES,

        "stride_samples":
            STRIDE_SAMPLES,

        "candidate_windows":
            int(
                candidate_windows
            ),

        "n_windows":
            int(
                windows.shape[0]
            ),

        "final_shape":
            [
                int(x)
                for x
                in windows.shape
            ],

        "sha256":
            local_sha,

        "elapsed_seconds":
            float(
                elapsed
            ),

        "timestamp_utc":
            utc_now(),
    }


    write_json_atomic(
        output_json,
        metadata
    )


    return {
        "subject":
            subject,

        "block":
            block,

        "task_family":
            task_family,

        "status":
            "PASS",

        "input_total_channels":
            input_total_channels,

        "bids_eeg_channels":
            len(
                bids_eeg_names
            ),

        "bids_bad_channels_excluded":
            format_channels(
                bids_bad
            ),

        "mne_bad_channels_excluded":
            format_channels(
                mne_bad
            ),

        "missing_bids_eeg_channels":
            format_channels(
                missing_bids_eeg
            ),

        "dead_channels_removed":
            format_channels(
                dead_names
            ),

        "n_channels_after_dead_removal":
            n_channels,

        "zero_padded_channels":
            zero_padded,

        "trimmed_channels":
            trimmed,

        "ica_components":
            n_ica,

        "ica_excluded_count":
            len(
                bad_ics
            ),

        "ica_excluded_indices":
            format_channels(
                bad_ics
            ),

        "iclabel_fallback":
            iclabel_fallback,

        "candidate_windows":
            candidate_windows,

        "retained_windows":
            int(
                windows.shape[0]
            ),

        "final_shape":
            str(
                tuple(
                    windows.shape
                )
            ),

        "sha256":
            local_sha,

        "elapsed_seconds":
            f"{elapsed:.2f}",

        "error":
            (
                iclabel_error
                if iclabel_fallback
                else ""
            ),

        "timestamp_utc":
            utc_now(),
    }


# ======================================================================
# PROCESS ALL 92 SUBJECTS
# ======================================================================

print()
print(
    "=" * 72
)

print(
    "STARTING FULL PRIMARY92 PREPROCESSING"
)

print(
    "NO INTERPOLATION"
)

print(
    "=" * 72
)


pair_number = 0

total_pairs = (
    len(subjects)
    * len(BLOCKS)
)


for subject_index, subject in enumerate(
    subjects,
    start=1
):

    print()
    print(
        "=" * 72
    )

    print(
        f"SUBJECT "
        f"{subject_index}/"
        f"{len(subjects)}: "
        f"{subject}"
    )

    print(
        "=" * 72
    )


    for block, task_family in BLOCKS.items():

        pair_number += 1

        stem = (
            f"{subject}"
            f"_task-{block}"
        )


        output_npy = (
            OUT
            / (
                f"{stem}"
                "_preprocessed.npy"
            )
        )


        output_json = (
            META_DIR
            / (
                f"{stem}.json"
            )
        )


        print()
        print(
            f"[{pair_number}/"
            f"{total_pairs}] "
            f"{subject} "
            f"{block} "
            f"({task_family})"
        )


        # --------------------------------------------------------------
        # RESUME: skip verified completed output
        # --------------------------------------------------------------

        if (
            output_npy.exists()
            and output_npy.stat().st_size
            > 0
            and output_json.exists()
        ):

            try:

                saved = json.loads(
                    output_json.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    saved.get(
                        "status"
                    )
                    == "PASS"
                ):

                    print(
                        "  SKIP: "
                        "already completed"
                    )

                    # Ensure manifest contains
                    # the checkpointed result.
                    upsert_manifest(
                        {
                            "subject":
                                subject,

                            "block":
                                block,

                            "task_family":
                                task_family,

                            "status":
                                "PASS",

                            "input_total_channels":
                                saved.get(
                                    "input_total_channels",
                                    ""
                                ),

                            "bids_eeg_channels":
                                saved.get(
                                    "bids_eeg_channels",
                                    ""
                                ),

                            "bids_bad_channels_excluded":
                                format_channels(
                                    saved.get(
                                        "bids_bad_channels_excluded",
                                        []
                                    )
                                ),

                            "mne_bad_channels_excluded":
                                format_channels(
                                    saved.get(
                                        "mne_bad_channels_excluded",
                                        []
                                    )
                                ),

                            "missing_bids_eeg_channels":
                                format_channels(
                                    saved.get(
                                        "missing_bids_eeg_channels",
                                        []
                                    )
                                ),

                            "dead_channels_removed":
                                format_channels(
                                    saved.get(
                                        "dead_channels",
                                        []
                                    )
                                ),

                            "n_channels_after_dead_removal":
                                saved.get(
                                    "n_channels_after_dead_removal",
                                    ""
                                ),

                            "zero_padded_channels":
                                saved.get(
                                    "zero_padded_channels",
                                    ""
                                ),

                            "trimmed_channels":
                                saved.get(
                                    "trimmed_channels",
                                    ""
                                ),

                            "ica_components":
                                saved.get(
                                    "ica_components",
                                    ""
                                ),

                            "ica_excluded_count":
                                len(
                                    saved.get(
                                        "ica_excluded",
                                        []
                                    )
                                ),

                            "ica_excluded_indices":
                                format_channels(
                                    saved.get(
                                        "ica_excluded",
                                        []
                                    )
                                ),

                            "iclabel_fallback":
                                saved.get(
                                    "iclabel_fallback",
                                    ""
                                ),

                            "candidate_windows":
                                saved.get(
                                    "candidate_windows",
                                    ""
                                ),

                            "retained_windows":
                                saved.get(
                                    "n_windows",
                                    ""
                                ),

                            "final_shape":
                                str(
                                    tuple(
                                        saved.get(
                                            "final_shape",
                                            []
                                        )
                                    )
                                ),

                            "sha256":
                                saved.get(
                                    "sha256",
                                    ""
                                ),

                            "elapsed_seconds":
                                saved.get(
                                    "elapsed_seconds",
                                    ""
                                ),

                            "error":
                                saved.get(
                                    "iclabel_error",
                                    ""
                                ),

                            "timestamp_utc":
                                saved.get(
                                    "timestamp_utc",
                                    utc_now()
                                ),
                        }
                    )

                    refresh_reports()

                    continue

            except Exception:
                pass


        # --------------------------------------------------------------
        # ROLLING TEMP DIRECTORY
        # --------------------------------------------------------------

        raw_dir = (
            LOCAL_RAW_ROOT
            / subject
            / "eeg"
        )


        shutil.rmtree(
            raw_dir.parent,
            ignore_errors=True
        )


        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        base = (
            f"{BASE_URL}/"
            f"{subject}/eeg"
        )


        required_files = [
            f"{stem}_eeg.vhdr",
            f"{stem}_eeg.vmrk",
            f"{stem}_channels.tsv",
            f"{stem}_eeg.eeg",
        ]


        download_ok = True
        download_error = ""


        for filename in required_files:

            url = (
                f"{base}/"
                f"{filename}"
            )

            destination = (
                raw_dir
                / filename
            )


            print(
                "  download:",
                filename
            )


            ok, err = download_file(
                url,
                destination
            )


            if not ok:

                download_ok = False

                download_error = (
                    f"{filename}: "
                    f"{err}"
                )

                break


        if not download_ok:

            print(
                "  NOT AVAILABLE "
                "OR DOWNLOAD FAILED"
            )


            upsert_manifest(
                {
                    "subject":
                        subject,

                    "block":
                        block,

                    "task_family":
                        task_family,

                    "status":
                        "DOWNLOAD_FAILED_OR_MISSING",

                    "error":
                        download_error,

                    "timestamp_utc":
                        utc_now(),
                }
            )


            refresh_reports()


            shutil.rmtree(
                raw_dir.parent,
                ignore_errors=True
            )


            continue


        # --------------------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------------------

        try:

            record = (
                preprocess_recording(
                    subject,
                    block,
                    task_family,
                    raw_dir,
                    output_npy,
                    output_json
                )
            )


            upsert_manifest(
                record
            )


            print(
                "  STATUS=PASS"
            )

            print(
                "  DEAD_CHANNELS_REMOVED="
                f"{record['dead_channels_removed']}"
            )

            print(
                "  ZERO_PADDED_CHANNELS="
                f"{record['zero_padded_channels']}"
            )

            print(
                "  RETAINED_WINDOWS="
                f"{record['retained_windows']}"
            )

            print(
                "  FINAL_SHAPE="
                f"{record['final_shape']}"
            )

            print(
                "  SHA256="
                f"{record['sha256']}"
            )


        except Exception as e:

            err = (
                f"{type(e).__name__}: "
                f"{e}"
            )


            print(
                "  STATUS="
                "PREPROCESSING_FAILED"
            )

            print(
                "  ERROR=",
                err
            )


            upsert_manifest(
                {
                    "subject":
                        subject,

                    "block":
                        block,

                    "task_family":
                        task_family,

                    "status":
                        "PREPROCESSING_FAILED",

                    "error":
                        err,

                    "timestamp_utc":
                        utc_now(),
                }
            )


            traceback.print_exc()


        finally:

            refresh_reports()


            # Raw MIPDB is temporary:
            # delete after every recording.
            shutil.rmtree(
                raw_dir.parent,
                ignore_errors=True
            )


# ======================================================================
# FINAL REPORT
# ======================================================================

refresh_reports()

print()
print(
    "=" * 72
)

print(
    "FULL PRIMARY92 PAIR SCAN FINISHED"
)

print(
    "=" * 72
)

print()

print(
    SUMMARY.read_text(
        encoding="utf-8"
    )
)

print(
    "Dead/bad channel audit:"
)

print(
    CHANNEL_AUDIT
)

print(
    "Manifest:"
)

print(
    MANIFEST
)

print(
    "Preprocessed outputs:"
)

print(
    OUT
)

print(
    "FINISHED_UTC=",
    utc_now()
)
