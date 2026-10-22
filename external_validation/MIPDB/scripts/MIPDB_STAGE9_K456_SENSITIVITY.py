#!/usr/bin/env python3

from pathlib import Path
import ast
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import umap

from scipy.spatial.distance import cdist

from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)


# ======================================================================
# PATHS
# ======================================================================

DRIVE = Path(
    "/content/drive/MyDrive/FULL_23PHD1241_BACKUP_2026_07_04"
)

ROOT = DRIVE / "NEW_TASK_MULTIVIEW_V2"

EXT = (
    DRIVE
    / "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
    / "EXTERNAL_VALIDATION_MIPDB"
)

AUDIT = EXT / "00_AUDIT"

PRIMARY = (
    EXT
    / "08_CLUSTERING"
    / "EEG_ONLY"
    / "PRIMARY_K5"
)

UMAP_FILE = (
    PRIMARY
    / "MIPDB_PRIMARY_UMAP30.npy"
)

PRIMARY_LABELS_FILE = (
    PRIMARY
    / "MIPDB_PRIMARY_K5_LABELS.npy"
)

PRIMARY_METRICS_FILE = (
    PRIMARY
    / "MIPDB_PRIMARY_K5_METRICS.json"
)

STAGE8_LOCK = (
    AUDIT
    / "MIPDB_STAGE8_PRIMARY_K5_STABILITY_LOCK.txt"
)

META_FILE = (
    EXT
    / "06_EXTERNAL_GRAPH"
    / "EEG_ONLY"
    / "subject_metadata.csv"
)

PRODUCTION_SCRIPT = (
    ROOT
    / "hbn-eeg-neurobehavioral-subtypes"
    / "scripts"
    / "09_final_forced_k5_validation_graphK12_alpha06_umap30.py"
)

OUT = (
    EXT
    / "09_RESULTS"
    / "EEG_ONLY"
    / "K456_SENSITIVITY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# FIXED SETTINGS
# ======================================================================

EXPECTED_SUBJECTS = 91

SEED = 42

SPECTRAL_N_NEIGHBORS = 12

K_VALUES = [
    4,
    5,
    6,
]

EXPECTED_PRODUCTION_SHA256 = (
    "b239c2a4a94c28126a64b6c863ae9ab5"
    "c951cf9b76221d21820ced77fd1a8c33"
)


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_exact_functions(path):

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path)
    )


    constants = {}

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign
        ):

            try:
                value = ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            for target in node.targets:

                if (
                    isinstance(
                        target,
                        ast.Name
                    )
                    and target.id.isupper()
                ):
                    constants[
                        target.id
                    ] = value


    wanted = {
        "run_spectral",
        "dunn_index",
    }


    nodes = [

        node

        for node in tree.body

        if (
            isinstance(
                node,
                ast.FunctionDef
            )
            and node.name in wanted
        )
    ]


    found = {
        n.name
        for n in nodes
    }


    if found != wanted:

        raise RuntimeError(
            f"Expected {wanted}, found {found}"
        )


    module = ast.Module(
        body=sorted(
            nodes,
            key=lambda n: n.lineno
        ),
        type_ignores=[]
    )

    ast.fix_missing_locations(
        module
    )


    namespace = {

        **constants,

        "np": np,

        "pd": pd,

        "SpectralClustering":
            SpectralClustering,

        "pairwise_distances":
            pairwise_distances,

        "cdist":
            cdist,
    }


    exec(
        compile(
            module,
            filename=str(path),
            mode="exec"
        ),
        namespace
    )


    return (
        namespace[
            "run_spectral"
        ],

        namespace[
            "dunn_index"
        ],
    )


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 9 — PREDEFINED K=4,5,6 SENSITIVITY")
print("SECONDARY ROBUSTNESS ANALYSIS ONLY")
print("PRIMARY K=5 REMAINS LOCKED")
print("=" * 78)


# ======================================================================
# 1. VERIFY PRIMARY ANALYSIS IS ALREADY LOCKED
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY LOCKED PRIMARY ANALYSIS")
print("=" * 78)


for p in [
    UMAP_FILE,
    PRIMARY_LABELS_FILE,
    PRIMARY_METRICS_FILE,
    STAGE8_LOCK,
    META_FILE,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


primary_metrics = json.loads(
    PRIMARY_METRICS_FILE.read_text(
        encoding="utf-8"
    )
)


if primary_metrics.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Primary K=5 analysis is not PASS."
    )


if primary_metrics.get(
    "spectral_k"
) != 5:

    raise RuntimeError(
        "Primary analysis is not K=5."
    )


X2 = np.load(
    UMAP_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


primary_labels = np.load(
    PRIMARY_LABELS_FILE,
    allow_pickle=False
).astype(
    int,
    copy=False
)


meta = pd.read_csv(
    META_FILE
)


if X2.shape != (
    EXPECTED_SUBJECTS,
    2
):

    raise RuntimeError(
        f"Unexpected UMAP shape {X2.shape}"
    )


if primary_labels.shape != (
    EXPECTED_SUBJECTS,
):

    raise RuntimeError(
        "Primary label shape mismatch."
    )


if len(meta) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Metadata row count mismatch."
    )


print(
    "PRIMARY_K=5"
)

print(
    "PRIMARY_ALREADY_LOCKED=YES"
)

print(
    "PRIMARY_UMAP_SHAPE =",
    X2.shape
)


# ======================================================================
# 2. VERIFY PRODUCTION SCRIPT
# ======================================================================

print()
print("=" * 78)
print("2. LOAD EXACT HBN SPECTRAL FUNCTION")
print("=" * 78)


production_sha = sha256_file(
    PRODUCTION_SCRIPT
)


if production_sha != EXPECTED_PRODUCTION_SHA256:

    raise RuntimeError(
        "Production script SHA256 mismatch."
    )


run_spectral, dunn_index = (
    load_exact_functions(
        PRODUCTION_SCRIPT
    )
)


print(
    "PRODUCTION_SCRIPT_SHA256 =",
    production_sha
)

print(
    "SEED =",
    SEED
)

print(
    "SPECTRAL_N_NEIGHBORS =",
    SPECTRAL_N_NEIGHBORS
)

print(
    "SENSITIVITY_K_VALUES =",
    K_VALUES
)


# ======================================================================
# 3. RUN K=4,5,6 ON THE LOCKED PRIMARY UMAP
# ======================================================================

print()
print("=" * 78)
print("3. K=4,5,6 SENSITIVITY")
print("=" * 78)


summary_rows = []

all_assignments = pd.DataFrame({
    "subject":
        meta[
            "subject"
        ].values
})


for k in K_VALUES:

    print()
    print("-" * 78)
    print(
        "SENSITIVITY_K =",
        k
    )
    print("-" * 78)


    labels = run_spectral(

        X2,

        k=k,

        seed=SEED,

        n_neighbors=
            SPECTRAL_N_NEIGHBORS,
    )


    labels = np.asarray(
        labels,
        dtype=int
    )


    if labels.shape != (
        EXPECTED_SUBJECTS,
    ):

        raise RuntimeError(
            f"K={k}: invalid label shape."
        )


    n_found = len(
        np.unique(
            labels
        )
    )


    if n_found != k:

        raise RuntimeError(
            f"K={k}: found {n_found} clusters."
        )


    sil = float(
        silhouette_score(
            X2,
            labels
        )
    )


    db = float(
        davies_bouldin_score(
            X2,
            labels
        )
    )


    ch = float(
        calinski_harabasz_score(
            X2,
            labels
        )
    )


    dunn = float(
        dunn_index(
            X2,
            labels
        )
    )


    counts = (
        pd.Series(
            labels
        )
        .value_counts()
        .sort_index()
    )


    min_cluster = int(
        counts.min()
    )


    max_cluster = int(
        counts.max()
    )


    balance = float(
        max_cluster
        / min_cluster
    )


    ari_primary = float(
        adjusted_rand_score(
            primary_labels,
            labels
        )
    )


    nmi_primary = float(
        normalized_mutual_info_score(
            primary_labels,
            labels
        )
    )


    summary_rows.append({

        "k":
            k,

        "silhouette":
            sil,

        "davies_bouldin":
            db,

        "calinski_harabasz":
            ch,

        "dunn_index":
            dunn,

        "min_cluster_size":
            min_cluster,

        "max_cluster_size":
            max_cluster,

        "balance_ratio_max_min":
            balance,

        "ARI_vs_primary_K5":
            ari_primary,

        "NMI_vs_primary_K5":
            nmi_primary,

        "cluster_sizes":
            json.dumps(
                {
                    str(
                        int(a) + 1
                    ):
                    int(b)

                    for a, b
                    in counts.items()
                }
            ),
    })


    all_assignments[
        f"K{k}_cluster_zero_based"
    ] = labels


    all_assignments[
        f"K{k}_subtype"
    ] = labels + 1


    print(
        "CLUSTER_SIZES =",
        counts.to_dict()
    )

    print(
        "SILHOUETTE =",
        sil
    )

    print(
        "DAVIES_BOULDIN =",
        db
    )

    print(
        "CALINSKI_HARABASZ =",
        ch
    )

    print(
        "DUNN_INDEX =",
        dunn
    )

    print(
        "MIN_CLUSTER_SIZE =",
        min_cluster
    )

    print(
        "BALANCE_RATIO =",
        balance
    )

    print(
        "ARI_VS_PRIMARY_K5 =",
        ari_primary
    )

    print(
        "NMI_VS_PRIMARY_K5 =",
        nmi_primary
    )


    # --------------------------------------------------------------
    # K=5 must exactly reproduce the locked primary partition,
    # apart from arbitrary label naming.
    # --------------------------------------------------------------

    if k == 5:

        if (
            ari_primary < 0.999999
            or nmi_primary < 0.999999
        ):

            raise RuntimeError(
                "K=5 sensitivity run does not reproduce "
                "the locked primary K=5 partition."
            )

        print(
            "PRIMARY_K5_REPRODUCTION=PASS"
        )


# ======================================================================
# 4. SAVE RESULTS
# ======================================================================

summary = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUT
    / "MIPDB_K456_SENSITIVITY_METRICS.csv"
)


assignment_file = (
    OUT
    / "MIPDB_K456_SENSITIVITY_ASSIGNMENTS.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


all_assignments.to_csv(
    assignment_file,
    index=False
)


# ======================================================================
# 5. REPORT WITHOUT SELECTING A NEW K
# ======================================================================

print()
print("=" * 78)
print("4. SENSITIVITY SUMMARY")
print("=" * 78)

print(
    summary.to_string(
        index=False
    )
)


best_silhouette_k = int(
    summary.loc[
        summary[
            "silhouette"
        ].idxmax(),
        "k"
    ]
)


best_db_k = int(
    summary.loc[
        summary[
            "davies_bouldin"
        ].idxmin(),
        "k"
    ]
)


print()
print(
    "BEST_SILHOUETTE_K_SECONDARY =",
    best_silhouette_k
)

print(
    "BEST_DB_K_SECONDARY =",
    best_db_k
)

print(
    "PRIMARY_K_REMAINS = 5"
)

print(
    "K_RESELECTED_AFTER_SENSITIVITY = NO"
)


# ======================================================================
# 6. SAVE JSON + LOCK
# ======================================================================

json_summary = {

    "analysis":
        "predefined_secondary_K456_sensitivity",

    "primary_k":
        5,

    "sensitivity_k_values":
        K_VALUES,

    "same_locked_primary_umap_used":
        True,

    "seed":
        SEED,

    "spectral_n_neighbors":
        SPECTRAL_N_NEIGHBORS,

    "best_silhouette_k_secondary":
        best_silhouette_k,

    "best_davies_bouldin_k_secondary":
        best_db_k,

    "primary_k_reselected":
        False,

    "hyperparameter_tuning":
        False,

    "results":
        summary_rows,

    "status":
        "PASS",
}


json_file = (
    OUT
    / "MIPDB_K456_SENSITIVITY_SUMMARY.json"
)


json_file.write_text(
    json.dumps(
        json_summary,
        indent=2
    ),
    encoding="utf-8"
)


lock_file = (
    AUDIT
    / "MIPDB_STAGE9_K456_SENSITIVITY_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 9 K SENSITIVITY LOCK
======================================================================

ANALYSIS=SECONDARY_ROBUSTNESS_ONLY

PRIMARY_K=5
PRIMARY_K_ALREADY_LOCKED=YES

SENSITIVITY_K_VALUES=4,5,6

INPUT_UMAP=LOCKED_PRIMARY_UMAP30
SEED=42
SPECTRAL_N_NEIGHBORS=12

BEST_SILHOUETTE_K_SECONDARY={best_silhouette_k}
BEST_DAVIES_BOULDIN_K_SECONDARY={best_db_k}

PRIMARY_K_RESELECTED=NO
HYPERPARAMETER_TUNING_ON_MIPDB=NO

RESULTS_FILE=
{summary_file}

ASSIGNMENTS_FILE=
{assignment_file}

SUMMARY_JSON=
{json_file}

STAGE9_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# 7. SHA256 MANIFEST
# ======================================================================

sha_file = (
    AUDIT
    / "MIPDB_STAGE9_K456_SENSITIVITY_SHA256.txt"
)


with open(
    sha_file,
    "w"
) as f:

    for p in [
        summary_file,
        assignment_file,
        json_file,
        lock_file,
    ]:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 78)
print("STAGE 9 COMPLETE")
print("=" * 78)

for row in summary_rows:

    print(
        f"K={row['k']} "
        f"SIL={row['silhouette']:.6f} "
        f"DB={row['davies_bouldin']:.6f} "
        f"CH={row['calinski_harabasz']:.6f} "
        f"DUNN={row['dunn_index']:.6f} "
        f"MIN={row['min_cluster_size']} "
        f"BAL={row['balance_ratio_max_min']:.3f}"
    )


print()
print(
    "PRIMARY_K=5"
)

print(
    "PRIMARY_K_RESELECTED=NO"
)

print(
    "HYPERPARAMETER_TUNING_ON_MIPDB=NO"
)

print(
    "STAGE9_STATUS=PASS"
)

print()
print(
    "OUTPUT_DIR =",
    OUT
)

print(
    "LOCK_FILE =",
    lock_file
)

print("=" * 78)

