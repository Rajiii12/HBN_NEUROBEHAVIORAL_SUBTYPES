"""Construct five recording-level EEG band views at 500 Hz.

Input arrays must have shape (windows, 128, 500). Accepted windows are
averaged within each recording before delta, theta, alpha, beta, and gamma
band filtering, consistent with the recording-level aggregation stage.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

DATASET_PATH = os.environ.get(
    "EPGSD_PREPROCESSED_DIR",
    "./data/preprocessed_HBN_v2",
)
OUTPUT_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/multiview_tensors_FS500")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 500

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

def bandpass_filter(data, lowcut, highcut, fs=500, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)

def main():
    eeg_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.npy")))

    print("\n===================================")
    print("FS=500 MULTIVIEW BAND DECOMPOSITION")
    print("===================================")
    print("Dataset:", DATASET_PATH)
    print("Files found:", len(eeg_files))
    print("Output:", OUTPUT_DIR)

    band_views = {b: [] for b in BANDS}
    metadata = []

    for file_path in tqdm(eeg_files):
        try:
            eeg = np.load(file_path)

            # Expected: epochs x channels x time
            if eeg.ndim != 3:
                raise ValueError(
                    f"Expected windows x channels x time, received {eeg.shape}"
                )
            if eeg.shape[1:] != (128, 500):
                raise ValueError(
                    f"Expected (*, 128, 500), received {eeg.shape}"
                )

            # Recording-level window-mean aggregation shown in the architecture diagram.
            eeg_mean = eeg.mean(axis=0).astype(np.float32)

            for band_name, (lowcut, highcut) in BANDS.items():
                filtered = bandpass_filter(
                    eeg_mean,
                    lowcut,
                    highcut,
                    fs=SAMPLING_RATE,
                    order=4,
                )
                band_views[band_name].append(filtered)

            filename = os.path.basename(file_path)

            # Extract the subject identifier from the expected filename pattern.
            subject = filename.split("_task-")[0] if "_task-" in filename else filename.replace(".npy", "")

            metadata.append({
                "subject_file": filename,
                "subject": subject,
                "sampling_rate": SAMPLING_RATE,
                "original_shape": str(tuple(eeg.shape)),
                "aggregated_shape": str(tuple(eeg_mean.shape)),
            })

        except Exception as e:
            raise RuntimeError(
                f"Band decomposition failed for {file_path}: {e}"
            ) from e

    for band_name in BANDS:
        tensor = np.asarray(band_views[band_name], dtype=np.float32)
        np.save(OUTPUT_DIR / f"{band_name}.npy", tensor)
        print(f"{band_name}: {tensor.shape}")

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(OUTPUT_DIR / "band_decomposition_metadata_FS500.csv", index=False)

    print("\nDONE.")
    print("Saved FS=500 band tensors in:", OUTPUT_DIR)
    print("Metadata:", OUTPUT_DIR / "band_decomposition_metadata_FS500.csv")

if __name__ == "__main__":
    main()
