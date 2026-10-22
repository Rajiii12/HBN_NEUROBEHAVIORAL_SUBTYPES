from pathlib import Path
import json
import os
import time
import hashlib
import traceback

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ============================================================
# PATHS
# ============================================================

DRIVE = Path(
    "/content/drive/MyDrive/"
    "FULL_23PHD1241_BACKUP_2026_07_04/"
    "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
)

EXT = DRIVE / "EXTERNAL_VALIDATION_MIPDB"

PRE = (
    EXT
    / "03_PREPROCESSED"
    / "MIPDB_HBN_EXACT_PRIMARY92"
)

PREMETA = PRE / "metadata"

OUT = (
    EXT
    / "04_FEATURES"
    / "MIPDB_BAND_VIEWS_PRIMARY92"
)

OUTMETA = OUT / "metadata"

AUDIT = EXT / "00_AUDIT"

MANIFEST = (
    AUDIT
    / "MIPDB_STAGE2_BAND_MANIFEST.csv"
)

SUMMARY = (
    AUDIT
    / "MIPDB_STAGE2_BAND_SUMMARY.txt"
)

LOCK = (
    AUDIT
    / "MIPDB_STAGE2_HBN_EXACT_BAND_LOCK.txt"
)


# ============================================================
# EXACT VERIFIED HBN SETTINGS
# ============================================================

SAMPLING_RATE = 500

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

FILTER_ORDER = 4
POLL_SECONDS = 30


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for band in BANDS:
    (OUT / band).mkdir(
        parents=True,
        exist_ok=True
    )

OUTMETA.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FREEZE SCIENTIFIC TRANSFORMATION
# ============================================================

LOCK.write_text(
"""EPG-SD MIPDB STAGE 2 — HBN EXACT BAND LOCK

Verified against:
HBN_NEUROBEHAVIORAL_SUBTYPES
scripts/core_stage00_to06/02_band_decomposition.py

Input:
windows x 128 x 500

Recording aggregation:
eeg_mean = eeg.mean(axis=0).astype(np.float32)

Sampling rate:
500 Hz

Bands:
delta 1-4 Hz
theta 4-8 Hz
alpha 8-13 Hz
beta 13-30 Hz
gamma 30-40 Hz

Band filter:
scipy.signal.butter
order = 4
btype = band
scipy.signal.filtfilt
axis = -1
output float32

This stage does NOT modify preprocessing files.
This stage does NOT train CNN-AE.
This stage does NOT perform PCA, graph construction, VGAE,
UMAP, or clustering.

Per-recording saving is an external-validation engineering
wrapper only. Scientific signal transformation is the HBN
Stage-2 transformation.
""",
    encoding="utf-8"
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now():
    from datetime import datetime, timezone
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def atomic_save_npy(path, array):
    path = Path(path)

    temp = Path(
        str(path) + ".partial.npy"
    )

    if temp.exists():
        temp.unlink()

    np.save(
        temp,
        array.astype(np.float32)
    )

    temp_sha = sha256_file(temp)

    os.replace(
        temp,
        path
    )

    final_sha = sha256_file(path)

    if temp_sha != final_sha:
        raise RuntimeError(
            f"SHA256 verification failed: {path}"
        )

    return final_sha


# ============================================================
# EXACT HBN BANDPASS FUNCTION
# ============================================================

def bandpass_filter(
    data,
    lowcut,
    highcut,
    fs=500,
    order=4
):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        data,
        axis=-1
    ).astype(np.float32)


# ============================================================
# OUTPUT HELPERS
# ============================================================

def get_band_paths(stem):

    return {
        band:
        OUT
        / band
        / f"{stem}_{band}.npy"

        for band in BANDS
    }


def recording_complete(stem):

    meta = (
        OUTMETA
        / f"{stem}_bands.json"
    )

    if not meta.exists():
        return False

    try:
        info = json.loads(
            meta.read_text(
                encoding="utf-8"
            )
        )

        if info.get("status") != "PASS":
            return False

    except Exception:
        return False

    for path in get_band_paths(
        stem
    ).values():

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            return False

    return True


# ============================================================
# MANIFEST
# ============================================================

COLUMNS = [
    "subject",
    "block",
    "stem",
    "status",
    "source_windows",
    "source_shape",
    "aggregated_shape",
    "delta_sha256",
    "theta_sha256",
    "alpha_sha256",
    "beta_sha256",
    "gamma_sha256",
    "error",
    "timestamp_utc",
]


def load_manifest():

    if MANIFEST.exists():

        try:
            df = pd.read_csv(
                MANIFEST,
                dtype=str
            ).fillna("")

        except Exception:
            df = pd.DataFrame(
                columns=COLUMNS
            )

    else:
        df = pd.DataFrame(
            columns=COLUMNS
        )

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[COLUMNS]


def update_manifest(record):

    df = load_manifest()

    stem = str(
        record.get(
            "stem",
            ""
        )
    )

    if len(df):
        df = df[
            df["stem"] != stem
        ].copy()

    row = {
        col:
        str(
            record.get(
                col,
                ""
            )
        )
        for col in COLUMNS
    }

    df = pd.concat(
        [
            df,
            pd.DataFrame([row])
        ],
        ignore_index=True
    )

    df = df.sort_values(
        [
            "subject",
            "block",
            "stem"
        ]
    )

    temp = Path(
        str(MANIFEST) + ".tmp"
    )

    df.to_csv(
        temp,
        index=False
    )

    os.replace(
        temp,
        MANIFEST
    )


# ============================================================
# PROCESS ONE PREPROCESSED RECORDING
# ============================================================

def process_recording(meta_path):

    pre_meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    if pre_meta.get("status") != "PASS":
        return False


    stem = meta_path.stem

    source = (
        PRE
        / f"{stem}_preprocessed.npy"
    )


    if not source.exists():
        return False


    if recording_complete(stem):
        return False


    print()
    print("=" * 72)
    print("STAGE2 RECORDING =", stem)
    print("=" * 72)


    # ========================================================
    # LOAD HBN-STYLE PREPROCESSED WINDOWS
    # ========================================================

    eeg = np.load(
        source,
        allow_pickle=False
    )


    if eeg.ndim != 3:
        raise RuntimeError(
            f"Expected 3-D input, received {eeg.shape}"
        )


    if eeg.shape[1:] != (128, 500):
        raise RuntimeError(
            f"Expected (*,128,500), received {eeg.shape}"
        )


    if eeg.shape[0] < 10:
        raise RuntimeError(
            f"Too few accepted windows: {eeg.shape[0]}"
        )


    if not np.isfinite(eeg).all():
        raise RuntimeError(
            "Input contains NaN or Inf"
        )


    source_shape = tuple(
        int(x)
        for x in eeg.shape
    )


    # ========================================================
    # EXACT HBN RECORDING-LEVEL AGGREGATION
    # ========================================================

    eeg_mean = eeg.mean(
        axis=0
    ).astype(np.float32)


    if eeg_mean.shape != (128, 500):
        raise RuntimeError(
            f"Unexpected aggregated shape: {eeg_mean.shape}"
        )


    del eeg


    # ========================================================
    # EXACT FIVE HBN FREQUENCY BANDS
    # ========================================================

    band_paths = get_band_paths(
        stem
    )

    hashes = {}


    for band_name, (
        lowcut,
        highcut
    ) in BANDS.items():

        filtered = bandpass_filter(
            eeg_mean,
            lowcut,
            highcut,
            fs=SAMPLING_RATE,
            order=FILTER_ORDER
        )


        if filtered.shape != (128, 500):
            raise RuntimeError(
                f"{band_name}: wrong shape "
                f"{filtered.shape}"
            )


        if not np.isfinite(
            filtered
        ).all():

            raise RuntimeError(
                f"{band_name}: NaN/Inf detected"
            )


        hashes[band_name] = (
            atomic_save_npy(
                band_paths[band_name],
                filtered
            )
        )


        print(
            f"{band_name:5s} "
            f"shape={filtered.shape} "
            f"SHA256={hashes[band_name]}"
        )


    # ========================================================
    # SAVE BAND METADATA
    # ========================================================

    result = {
        "status":
            "PASS",

        "subject":
            pre_meta.get(
                "subject",
                ""
            ),

        "block":
            pre_meta.get(
                "block",
                ""
            ),

        "stem":
            stem,

        "source_preprocessed_file":
            source.name,

        "source_shape":
            list(source_shape),

        "recording_aggregation":
            "mean_axis_0",

        "aggregated_shape":
            [128, 500],

        "sampling_rate_hz":
            500,

        "filter_type":
            "Butterworth_bandpass",

        "filter_order":
            4,

        "zero_phase_filter":
            "scipy.signal.filtfilt",

        "bands_hz":
            {
                band:
                list(bounds)
                for band, bounds
                in BANDS.items()
            },

        "band_sha256":
            hashes,

        "timestamp_utc":
            utc_now(),
    }


    temp_json = (
        OUTMETA
        / f"{stem}_bands.json.tmp"
    )

    final_json = (
        OUTMETA
        / f"{stem}_bands.json"
    )


    temp_json.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    os.replace(
        temp_json,
        final_json
    )


    update_manifest(
        {
            "subject":
                pre_meta.get(
                    "subject",
                    ""
                ),

            "block":
                pre_meta.get(
                    "block",
                    ""
                ),

            "stem":
                stem,

            "status":
                "PASS",

            "source_windows":
                source_shape[0],

            "source_shape":
                str(source_shape),

            "aggregated_shape":
                "(128, 500)",

            "delta_sha256":
                hashes["delta"],

            "theta_sha256":
                hashes["theta"],

            "alpha_sha256":
                hashes["alpha"],

            "beta_sha256":
                hashes["beta"],

            "gamma_sha256":
                hashes["gamma"],

            "error":
                "",

            "timestamp_utc":
                utc_now(),
        }
    )


    print("STATUS=PASS")
    print(
        "SOURCE_SHAPE=",
        source_shape
    )
    print(
        "AGGREGATED_SHAPE=(128, 500)"
    )
    print(
        "FIVE_BANDS_COMPLETE=YES"
    )

    return True


# ============================================================
# SUMMARY
# ============================================================

def write_summary():

    pre_json_count = len(
        list(
            PREMETA.glob(
                "*.json"
            )
        )
    )

    pre_npy_count = len(
        list(
            PRE.glob(
                "*_preprocessed.npy"
            )
        )
    )

    df = load_manifest()

    if len(df):

        pass_count = int(
            (
                df["status"]
                == "PASS"
            ).sum()
        )

        fail_count = int(
            (
                df["status"]
                == "FAILED"
            ).sum()
        )

    else:

        pass_count = 0
        fail_count = 0


    text = (
        "EPG-SD MIPDB STAGE 2 BAND SUMMARY\n"
        "============================================================\n"
        f"UPDATED_UTC={utc_now()}\n"
        f"PREPROCESSING_JSON_AVAILABLE={pre_json_count}\n"
        f"PREPROCESSING_NPY_AVAILABLE={pre_npy_count}\n"
        f"BAND_PASS_RECORDINGS={pass_count}\n"
        f"BAND_FAILED_RECORDINGS={fail_count}\n"
        "SAMPLING_RATE_HZ=500\n"
        "AGGREGATION=MEAN_AXIS_0\n"
        "DELTA_HZ=1-4\n"
        "THETA_HZ=4-8\n"
        "ALPHA_HZ=8-13\n"
        "BETA_HZ=13-30\n"
        "GAMMA_HZ=30-40\n"
        "FILTER_ORDER=4\n"
        "FILTER=BUTTERWORTH_FILTFILT\n"
    )


    temp = Path(
        str(SUMMARY) + ".tmp"
    )

    temp.write_text(
        text,
        encoding="utf-8"
    )

    os.replace(
        temp,
        SUMMARY
    )


# ============================================================
# CONTINUOUS INCREMENTAL WORKER
# ============================================================

print("=" * 72)
print("MIPDB STAGE 2 HBN-EXACT BAND WORKER STARTED")
print("=" * 72)

print("INPUT =", PRE)
print("OUTPUT =", OUT)


while True:

    processed_now = 0

    meta_files = sorted(
        PREMETA.glob(
            "*.json"
        )
    )


    for meta_path in meta_files:

        stem = meta_path.stem

        if recording_complete(
            stem
        ):
            continue


        try:

            completed = process_recording(
                meta_path
            )

            if completed:
                processed_now += 1


        except Exception as e:

            print()
            print(
                "STAGE2 FAILED =",
                stem
            )

            print(
                type(e).__name__,
                str(e)
            )

            traceback.print_exc()


            try:

                m = json.loads(
                    meta_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:

                m = {}


            update_manifest(
                {
                    "subject":
                        m.get(
                            "subject",
                            ""
                        ),

                    "block":
                        m.get(
                            "block",
                            ""
                        ),

                    "stem":
                        stem,

                    "status":
                        "FAILED",

                    "error":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),

                    "timestamp_utc":
                        utc_now(),
                }
            )


    write_summary()


    df = load_manifest()

    band_pass = (
        int(
            (
                df["status"]
                == "PASS"
            ).sum()
        )
        if len(df)
        else 0
    )

    band_fail = (
        int(
            (
                df["status"]
                == "FAILED"
            ).sum()
        )
        if len(df)
        else 0
    )


    print()
    print("------------------------------------------------------------")
    print(
        "PREPROCESSING_JSON_AVAILABLE=",
        len(meta_files)
    )
    print(
        "BAND_PASS=",
        band_pass
    )
    print(
        "BAND_FAILED=",
        band_fail
    )
    print(
        "NEW_THIS_CYCLE=",
        processed_now
    )
    print(
        "WAIT_SECONDS=",
        POLL_SECONDS
    )
    print("------------------------------------------------------------")


    time.sleep(
        POLL_SECONDS
    )
