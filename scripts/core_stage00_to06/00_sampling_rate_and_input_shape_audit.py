"""Audit preprocessed EEG array shapes before downstream analysis.

The script reports array dimensions and candidate temporal durations for
common sampling rates. Set EPGSD_AUDIT_LIMIT=0 to inspect all files.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
DATASET_PATH = os.environ.get(
    "EPGSD_PREPROCESSED_DIR",
    "./data/preprocessed_HBN_v2",
)
OUTPUT_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/sampling_rate_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FS_OPTIONS = [256, 500]

def infer_duration_seconds(n_times, fs):
    return n_times / fs

def main():
    files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.npy")))
    print("\n===================================")
    print("SAMPLING RATE / SHAPE CHECK")
    print("===================================")
    print("Dataset path:", DATASET_PATH)
    print("Total .npy files:", len(files))

    rows = []
    AUDIT_LIMIT = int(os.environ.get("EPGSD_AUDIT_LIMIT", "50"))
    selected_files = files if AUDIT_LIMIT <= 0 else files[:AUDIT_LIMIT]
    for fp in selected_files:
        try:
            x = np.load(fp, mmap_mode="r")
            shape = tuple(x.shape)
            n_times = shape[-1]
            row = {
                "file": os.path.basename(fp),
                "shape": str(shape),
                "ndim": len(shape),
                "n_times_last_dim": int(n_times),
            }
            for fs in EXPECTED_FS_OPTIONS:
                row[f"duration_if_fs_{fs}"] = infer_duration_seconds(n_times, fs)
            rows.append(row)
        except Exception as e:
            rows.append({"file": os.path.basename(fp), "error": str(e)})

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "preprocessed_shape_sampling_rate_check.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nPreview:")
    print(df.head(10).to_string(index=False))

    print("\nINTERPRETATION GUIDE:")
    print("- Use a consistent sampling rate across preprocessing and spectral analysis.")
    print("- The EPG-SD pipeline expects one-second windows with 500 samples.")

if __name__ == "__main__":
    main()
