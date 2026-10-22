#!/usr/bin/env python3

from pathlib import Path
from io import BytesIO
import ast
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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

BAND_ROOT = (
    EXT
    / "04_FEATURES"
    / "MIPDB_BAND_VIEWS_PRIMARY92"
)

FROZEN = (
    EXT
    / "05_FROZEN_HBN_ARTIFACTS"
)

CODE = FROZEN / "production_code"

W1 = FROZEN / "stream1_dynamic"
W2 = FROZEN / "stream2_cnn"

WEIGHT_MANIFEST = (
    FROZEN
    / "HBN_PRODUCTION_ARTIFACT_SHA256.txt"
)

STAGE2_MANIFEST = (
    AUDIT
    / "MIPDB_STAGE2_BAND_MANIFEST.csv"
)

OUT = (
    EXT
    / "04_FEATURES"
    / "MIPDB_HBN_FROZEN_EMBEDDINGS_PRIMARY91"
)

OUT1 = OUT / "stream1_dynamic"
OUT2 = OUT / "stream2_cnn"
RECEIPTS = OUT / "receipts"

for d in [OUT, OUT1, OUT2, RECEIPTS]:
    d.mkdir(parents=True, exist_ok=True)


BANDS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]

EXPECTED_RECORDINGS = 825
EXPECTED_SUBJECTS = 91
EXPECTED_SHAPE = (128, 500)
LATENT_DIM = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray):
    """
    Write through a temporary file in the destination directory,
    then atomically rename.
    """

    tmp = path.with_name(
        path.name + ".partial.npy"
    )

    np.save(
        tmp,
        array,
        allow_pickle=False,
    )

    # np.save does not append another suffix because tmp already ends .npy
    os.replace(
        tmp,
        path,
    )


def load_class_only(script: Path):
    """
    Extract only DynamicEEGAutoEncoder from the frozen production
    script. This deliberately does NOT execute the training script.
    """

    source = script.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(script),
    )

    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DynamicEEGAutoEncoder"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one DynamicEEGAutoEncoder "
            f"in {script}, found {len(matches)}"
        )

    module = ast.Module(
        body=matches,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
    }

    exec(
        compile(
            module,
            filename=str(script),
            mode="exec",
        ),
        namespace,
    )

    return namespace[
        "DynamicEEGAutoEncoder"
    ]


def safe_torch_load(path: Path):
    """
    Load a state_dict without executing arbitrary checkpoint objects.
    """

    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=DEVICE,
        )


def normalize_state_dict(state):
    """
    Handle common checkpoint wrappers while still requiring
    a pure model state_dict.
    """

    if isinstance(state, dict):

        if "state_dict" in state:
            state = state["state_dict"]

        elif "model_state_dict" in state:
            state = state[
                "model_state_dict"
            ]

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint did not contain a state_dict."
        )

    # Handle DataParallel prefix if present.
    if (
        len(state) > 0
        and all(
            str(k).startswith("module.")
            for k in state.keys()
        )
    ):
        state = {
            str(k)[7:]: v
            for k, v in state.items()
        }

    return state


def make_model(
    ModelClass,
    weight_path: Path,
):
    model = ModelClass(
        n_channels=128,
        latent_dim=128,
    )

    state = safe_torch_load(
        weight_path
    )

    state = normalize_state_dict(
        state
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    # Extra safety: frozen inference means no trainable gradients.
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def encode(model, x):
    """
    Prefer the production encode() method.
    If absent, use the latent output returned by forward().
    """

    if hasattr(model, "encode"):
        z = model.encode(x)

    else:
        result = model(x)

        if (
            isinstance(result, (tuple, list))
            and len(result) >= 2
        ):
            z = result[1]

        else:
            raise RuntimeError(
                "Could not identify latent output "
                "from production model."
            )

    if z.ndim != 2:
        raise RuntimeError(
            f"Latent output must be 2-D; "
            f"received {tuple(z.shape)}"
        )

    if z.shape[1] != LATENT_DIM:
        raise RuntimeError(
            f"Expected latent dimension "
            f"{LATENT_DIM}; received {z.shape[1]}"
        )

    return z


def parse_weight_manifest(path: Path):
    expected = {}

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

        digest, rel = parts

        expected[
            rel.strip()
        ] = digest.strip()

    return expected


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 3 — FROZEN HBN CNN-AE INFERENCE")
print("NO TRAINING")
print("NO MIPDB MODEL FITTING")
print("=" * 78)

print()
print("DEVICE =", DEVICE)
print(
    "TORCH_VERSION =",
    torch.__version__,
)


# ======================================================================
# 1. LOAD AUTHORITATIVE STAGE-2 MANIFEST
# ======================================================================

print()
print("=" * 78)
print("1. AUTHORITATIVE STAGE-2 MANIFEST")
print("=" * 78)

manifest = pd.read_csv(
    STAGE2_MANIFEST
)

print(
    "MANIFEST_ROWS =",
    len(manifest),
)

print(
    "STATUS_COUNTS =",
    manifest[
        "status"
    ].value_counts().to_dict(),
)

if len(manifest) != EXPECTED_RECORDINGS:
    raise RuntimeError(
        f"Expected {EXPECTED_RECORDINGS} rows, "
        f"found {len(manifest)}"
    )

if not (
    manifest["status"] == "PASS"
).all():
    raise RuntimeError(
        "Stage-2 manifest contains non-PASS rows."
    )

n_subjects = int(
    manifest[
        "subject"
    ].nunique()
)

print(
    "UNIQUE_SUBJECTS =",
    n_subjects,
)

if n_subjects != EXPECTED_SUBJECTS:
    raise RuntimeError(
        f"Expected {EXPECTED_SUBJECTS} subjects, "
        f"found {n_subjects}"
    )

if manifest[
    ["subject", "block", "stem"]
].duplicated().any():
    raise RuntimeError(
        "Duplicate recording rows detected "
        "in authoritative manifest."
    )


stage2_manifest_sha = sha256_file(
    STAGE2_MANIFEST
)

print(
    "STAGE2_MANIFEST_SHA256 =",
    stage2_manifest_sha,
)


# ======================================================================
# 2. SAVE EXACT STAGE-3 ROW ORDER
# ======================================================================

order = manifest.copy()

order.insert(
    0,
    "stage3_row",
    np.arange(
        len(order),
        dtype=int,
    ),
)

order_file = (
    OUT
    / "MIPDB_STAGE3_RECORDING_ORDER.csv"
)

order.to_csv(
    order_file,
    index=False,
)

print()
print(
    "ROW_ORDER_FILE =",
    order_file,
)


# ======================================================================
# 3. VERIFY FROZEN HBN WEIGHTS
# ======================================================================

print()
print("=" * 78)
print("2. VERIFY FROZEN HBN WEIGHTS")
print("=" * 78)

expected_hashes = parse_weight_manifest(
    WEIGHT_MANIFEST
)

weight_info = {}

for band in BANDS:

    p1 = (
        W1
        / f"{band}_dynamic_norm_autoencoder_best.pt"
    )

    p2 = (
        W2
        / f"{band}_cnn_norm_autoencoder_best.pt"
    )

    if not p1.exists():
        raise FileNotFoundError(p1)

    if not p2.exists():
        raise FileNotFoundError(p2)

    rel1 = (
        f"stream1_dynamic/"
        f"{p1.name}"
    )

    rel2 = (
        f"stream2_cnn/"
        f"{p2.name}"
    )

    got1 = sha256_file(p1)
    got2 = sha256_file(p2)

    exp1 = expected_hashes.get(
        rel1
    )

    exp2 = expected_hashes.get(
        rel2
    )

    if exp1 is None:
        raise RuntimeError(
            f"No frozen hash recorded for {rel1}"
        )

    if exp2 is None:
        raise RuntimeError(
            f"No frozen hash recorded for {rel2}"
        )

    if got1 != exp1:
        raise RuntimeError(
            f"Stream-1 weight SHA mismatch: {p1}"
        )

    if got2 != exp2:
        raise RuntimeError(
            f"Stream-2 weight SHA mismatch: {p2}"
        )

    weight_info[band] = {
        "stream1_path": str(p1),
        "stream1_sha256": got1,
        "stream2_path": str(p2),
        "stream2_sha256": got2,
    }

    print(
        f"{band:6s}",
        "STREAM1=PASS",
        "STREAM2=PASS",
    )


# ======================================================================
# 4. LOAD ONLY MODEL CLASSES — NEVER EXECUTE TRAINING CODE
# ======================================================================

print()
print("=" * 78)
print("3. LOAD FROZEN PRODUCTION MODEL DEFINITIONS")
print("=" * 78)

script1 = (
    CODE
    / "03_autoencoder_embeddings.py"
)

script2 = (
    CODE
    / "04_cnn_autoencoder_embeddings.py"
)

Model1 = load_class_only(
    script1
)

Model2 = load_class_only(
    script2
)

print(
    "STREAM1_MODEL_CLASS =",
    Model1.__name__,
)

print(
    "STREAM2_MODEL_CLASS =",
    Model2.__name__,
)

print(
    "TRAINING_CODE_EXECUTED = NO"
)


# ======================================================================
# 5. FROZEN INFERENCE BAND BY BAND
# ======================================================================

summary_rows = []

for band_index, band in enumerate(
    BANDS,
    start=1,
):

    print()
    print("=" * 78)
    print(
        f"4.{band_index} BAND = {band.upper()}"
    )
    print("=" * 78)

    out1 = (
        OUT1
        / f"{band}_embedding.npy"
    )

    out2 = (
        OUT2
        / f"{band}_cnn_embedding_fixed.npy"
    )

    receipt_path = (
        RECEIPTS
        / f"{band}_PASS.json"
    )

    # --------------------------------------------------------------
    # Resume only when previous receipt matches the same manifest
    # and both outputs are structurally valid.
    # --------------------------------------------------------------

    resume_ok = False

    if (
        out1.exists()
        and out2.exists()
        and receipt_path.exists()
    ):
        try:
            receipt = json.loads(
                receipt_path.read_text(
                    encoding="utf-8"
                )
            )

            a1 = np.load(
                out1,
                mmap_mode="r",
            )

            a2 = np.load(
                out2,
                mmap_mode="r",
            )

            resume_ok = (
                receipt.get(
                    "stage2_manifest_sha256"
                )
                == stage2_manifest_sha

                and tuple(a1.shape)
                == (
                    EXPECTED_RECORDINGS,
                    LATENT_DIM,
                )

                and tuple(a2.shape)
                == (
                    EXPECTED_RECORDINGS,
                    LATENT_DIM,
                )

                and receipt.get(
                    "stream1_weight_sha256"
                )
                == weight_info[
                    band
                ][
                    "stream1_sha256"
                ]

                and receipt.get(
                    "stream2_weight_sha256"
                )
                == weight_info[
                    band
                ][
                    "stream2_sha256"
                ]
            )

        except Exception:
            resume_ok = False

    if resume_ok:

        print(
            "VALID COMPLETED BAND FOUND — SKIPPING"
        )

        emb1 = np.load(out1)
        emb2 = np.load(out2)

    else:

        print(
            "LOADING FROZEN MODELS..."
        )

        model1 = make_model(
            Model1,
            Path(
                weight_info[
                    band
                ][
                    "stream1_path"
                ]
            ),
        )

        model2 = make_model(
            Model2,
            Path(
                weight_info[
                    band
                ][
                    "stream2_path"
                ]
            ),
        )

        embeddings1 = []
        embeddings2 = []

        t0 = time.time()

        for start in range(
            0,
            EXPECTED_RECORDINGS,
            BATCH_SIZE,
        ):

            stop = min(
                start + BATCH_SIZE,
                EXPECTED_RECORDINGS,
            )

            batch_arrays = []

            for row_index in range(
                start,
                stop,
            ):

                row = manifest.iloc[
                    row_index
                ]

                stem = str(
                    row["stem"]
                )

                filename = (
                    f"{stem}_{band}.npy"
                )

                path = (
                    BAND_ROOT
                    / band
                    / filename
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Authoritative Stage-2 "
                        f"file missing:\n{path}"
                    )

                # Read file once:
                # same bytes are used for SHA verification
                # and np.load.
                raw = path.read_bytes()

                got_sha = sha256_bytes(
                    raw
                )

                expected_sha = str(
                    row[
                        f"{band}_sha256"
                    ]
                ).strip()

                if got_sha != expected_sha:
                    raise RuntimeError(
                        f"Stage-2 SHA mismatch\n"
                        f"ROW={row_index}\n"
                        f"FILE={path}\n"
                        f"EXPECTED={expected_sha}\n"
                        f"GOT={got_sha}"
                    )

                arr = np.load(
                    BytesIO(raw),
                    allow_pickle=False,
                )

                if arr.shape != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"Unexpected shape "
                        f"{arr.shape} for {path}; "
                        f"expected "
                        f"{EXPECTED_SHAPE}"
                    )

                if arr.dtype != np.float32:
                    arr = arr.astype(
                        np.float32,
                        copy=False,
                    )

                if not np.isfinite(
                    arr
                ).all():
                    raise RuntimeError(
                        f"Non-finite values in {path}"
                    )

                batch_arrays.append(
                    arr
                )

            x_np = np.stack(
                batch_arrays,
                axis=0,
            )

            x = torch.from_numpy(
                x_np
            ).to(
                DEVICE,
                non_blocking=True,
            )

            with torch.inference_mode():

                z1 = encode(
                    model1,
                    x,
                )

                z2 = encode(
                    model2,
                    x,
                )

            z1 = (
                z1.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            z2 = (
                z2.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            if not np.isfinite(z1).all():
                raise RuntimeError(
                    f"{band}: non-finite "
                    f"Stream-1 embeddings"
                )

            if not np.isfinite(z2).all():
                raise RuntimeError(
                    f"{band}: non-finite "
                    f"Stream-2 embeddings"
                )

            embeddings1.append(z1)
            embeddings2.append(z2)

            print(
                f"{band}: "
                f"{stop}/{EXPECTED_RECORDINGS}"
            )

        emb1 = np.concatenate(
            embeddings1,
            axis=0,
        )

        emb2 = np.concatenate(
            embeddings2,
            axis=0,
        )

        if emb1.shape != (
            EXPECTED_RECORDINGS,
            LATENT_DIM,
        ):
            raise RuntimeError(
                f"{band}: invalid Stream-1 "
                f"shape {emb1.shape}"
            )

        if emb2.shape != (
            EXPECTED_RECORDINGS,
            LATENT_DIM,
        ):
            raise RuntimeError(
                f"{band}: invalid Stream-2 "
                f"shape {emb2.shape}"
            )

        atomic_save_npy(
            out1,
            emb1,
        )

        atomic_save_npy(
            out2,
            emb2,
        )

        receipt = {
            "band": band,
            "recordings": int(
                EXPECTED_RECORDINGS
            ),
            "subjects": int(
                EXPECTED_SUBJECTS
            ),
            "latent_dimension": int(
                LATENT_DIM
            ),
            "input_shape": list(
                EXPECTED_SHAPE
            ),
            "stage2_manifest_sha256":
                stage2_manifest_sha,
            "stream1_weight_sha256":
                weight_info[
                    band
                ][
                    "stream1_sha256"
                ],
            "stream2_weight_sha256":
                weight_info[
                    band
                ][
                    "stream2_sha256"
                ],
            "stream1_output_sha256":
                sha256_file(out1),
            "stream2_output_sha256":
                sha256_file(out2),
            "elapsed_seconds":
                float(
                    time.time() - t0
                ),
            "training_performed": False,
            "mipdb_model_fitting":
                False,
            "status": "PASS",
        }

        receipt_path.write_text(
            json.dumps(
                receipt,
                indent=2,
            ),
            encoding="utf-8",
        )

        del model1
        del model2

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Health statistics
    # --------------------------------------------------------------

    for stream_name, arr in [
        (
            "dynamic",
            emb1,
        ),
        (
            "cnn",
            emb2,
        ),
    ]:

        per_dim_std = np.std(
            arr,
            axis=0,
        )

        active_dims = int(
            np.sum(
                per_dim_std > 1e-8
            )
        )

        summary_rows.append({
            "band": band,
            "stream": stream_name,
            "recordings":
                int(arr.shape[0]),
            "latent_dim":
                int(arr.shape[1]),
            "mean":
                float(arr.mean()),
            "std":
                float(arr.std()),
            "min":
                float(arr.min()),
            "max":
                float(arr.max()),
            "zero_fraction":
                float(
                    (
                        np.abs(arr)
                        < 1e-8
                    ).mean()
                ),
            "active_dimensions":
                active_dims,
            "finite":
                bool(
                    np.isfinite(
                        arr
                    ).all()
                ),
        })

    print(
        "STREAM1_SHAPE =",
        emb1.shape,
    )

    print(
        "STREAM2_SHAPE =",
        emb2.shape,
    )

    print(
        "BAND_STATUS=PASS"
    )


# ======================================================================
# 6. FINAL AUDIT
# ======================================================================

print()
print("=" * 78)
print("5. FINAL STAGE-3 AUDIT")
print("=" * 78)

summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    AUDIT
    / "MIPDB_STAGE3_FROZEN_HBN_EMBEDDING_SUMMARY.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


all_outputs = []

for band in BANDS:

    p1 = (
        OUT1
        / f"{band}_embedding.npy"
    )

    p2 = (
        OUT2
        / f"{band}_cnn_embedding_fixed.npy"
    )

    for stream, p in [
        ("dynamic", p1),
        ("cnn", p2),
    ]:

        a = np.load(
            p,
            mmap_mode="r",
        )

        if tuple(a.shape) != (
            EXPECTED_RECORDINGS,
            LATENT_DIM,
        ):
            raise RuntimeError(
                f"Final output shape "
                f"failure: {p}"
            )

        all_outputs.append({
            "band": band,
            "stream": stream,
            "path": str(p),
            "shape":
                str(tuple(a.shape)),
            "sha256":
                sha256_file(p),
            "status":
                "PASS",
        })


output_manifest = pd.DataFrame(
    all_outputs
)

output_manifest_file = (
    AUDIT
    / "MIPDB_STAGE3_FROZEN_HBN_EMBEDDING_MANIFEST.csv"
)

output_manifest.to_csv(
    output_manifest_file,
    index=False,
)


print(
    "OUTPUT_MATRICES =",
    len(output_manifest),
)

print(
    "EXPECTED_OUTPUT_MATRICES = 10"
)

print(
    "RECORDINGS_PER_VIEW =",
    EXPECTED_RECORDINGS,
)

print(
    "SUBJECTS =",
    EXPECTED_SUBJECTS,
)

print(
    "LATENT_DIMENSION =",
    LATENT_DIM,
)

print(
    "ALL_OUTPUTS_FINITE =",
    bool(
        summary[
            "finite"
        ].all()
    ),
)

print(
    "MIN_ACTIVE_DIMENSIONS =",
    int(
        summary[
            "active_dimensions"
        ].min()
    ),
)


# ======================================================================
# 7. PERMANENT LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE3_FROZEN_HBN_EMBEDDING_LOCK.txt"
)

lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 3 LOCK
======================================================================

INPUT_STAGE2_MANIFEST=
{STAGE2_MANIFEST}

INPUT_STAGE2_MANIFEST_SHA256=
{stage2_manifest_sha}

AUTHORITATIVE_RECORDINGS=825
AUTHORITATIVE_SUBJECTS=91

INPUT_SHAPE_PER_RECORDING_PER_BAND=128x500

BANDS=delta,theta,alpha,beta,gamma

STREAM1=HBN_FROZEN_DYNAMIC_AUTOENCODER
STREAM2=HBN_FROZEN_CNN_AUTOENCODER

LATENT_DIMENSION_PER_VIEW=128
TOTAL_EMBEDDING_VIEWS=10

MODEL_TRAINING_ON_MIPDB=NO
MODEL_PARAMETER_UPDATES_ON_MIPDB=NO
OPTIMIZER_USED=NO
LOSS_FUNCTION_USED=NO

INFERENCE_MODE=torch.inference_mode
MODEL_MODE=eval

OUTPUT_STREAM1=
{OUT1}

OUTPUT_STREAM2=
{OUT2}

OUTPUT_MANIFEST=
{output_manifest_file}

OUTPUT_SUMMARY=
{summary_file}

STAGE3_STATUS=PASS
======================================================================
""",
    encoding="utf-8",
)


print()
print("=" * 78)
print("STAGE 3 COMPLETE")
print("=" * 78)

print(
    "AUTHORITATIVE_RECORDINGS=825"
)

print(
    "AUTHORITATIVE_SUBJECTS=91"
)

print(
    "OUTPUT_VIEWS=10"
)

print(
    "MODEL_TRAINING_ON_MIPDB=NO"
)

print(
    "MODEL_PARAMETER_UPDATES_ON_MIPDB=NO"
)

print(
    "STAGE3_STATUS=PASS"
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

