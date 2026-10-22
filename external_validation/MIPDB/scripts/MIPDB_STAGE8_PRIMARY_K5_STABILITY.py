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

from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
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

VGAE_DIR = (
    EXT
    / "07_VGAE"
    / "EEG_ONLY"
)

LATENT_FILE = (
    VGAE_DIR
    / "MIPDB_EEG_ONLY_VGAE_LATENT16_SEED42.npy"
)

PRIMARY_DIR = (
    EXT
    / "08_CLUSTERING"
    / "EEG_ONLY"
    / "PRIMARY_K5"
)

PRIMARY_LABELS_FILE = (
    PRIMARY_DIR
    / "MIPDB_PRIMARY_K5_LABELS.npy"
)

PRIMARY_METRICS_FILE = (
    PRIMARY_DIR
    / "MIPDB_PRIMARY_K5_METRICS.json"
)

PRIMARY_LOCK = (
    AUDIT
    / "MIPDB_STAGE7_PRIMARY_K5_LOCK.txt"
)

STAGE6_SUMMARY = (
    AUDIT
    / "MIPDB_STAGE6_VGAE_FIXED_CONFIG_SUMMARY.json"
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
    / "PRIMARY_K5_STABILITY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# FROZEN SETTINGS
# ======================================================================

EXPECTED_SUBJECTS = 91

FORCED_K = 5
GRAPH_K = 12
UMAP_N = 30

N_SUBSAMPLE = 100
N_BOOTSTRAP = 100

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


def get_literal_constants(tree):

    result = {}

    for node in tree.body:

        if isinstance(node, ast.Assign):

            try:
                value = ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                ):
                    result[target.id] = value

        elif isinstance(
            node,
            ast.AnnAssign
        ):

            if (
                isinstance(
                    node.target,
                    ast.Name
                )
                and node.target.id.isupper()
            ):

                try:
                    result[
                        node.target.id
                    ] = ast.literal_eval(
                        node.value
                    )
                except Exception:
                    pass

    return result


def load_exact_stability_functions(path):

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path)
    )

    constants = get_literal_constants(
        tree
    )

    required_constants = {
        "SEED": 42,
        "FORCED_K": 5,
        "N_SUBSAMPLE": 100,
        "N_BOOTSTRAP": 100,
        "UMAP_MIN_DIST": 0.1,
        "UMAP_METRIC": "euclidean",
    }

    for name, expected in required_constants.items():

        got = constants.get(name)

        if got != expected:

            raise RuntimeError(
                f"Production constant mismatch: "
                f"{name} expected={expected} got={got}"
            )


    wanted = {
        "set_seed",
        "safe_metric",
        "pairwise_jaccard_same_cluster",
        "run_umap_projection",
        "run_spectral",
        "subsample_stability",
        "bootstrap_stability",
        "summarize_values",
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
        node.name
        for node in nodes
    }


    missing = wanted - found

    if missing:

        raise RuntimeError(
            f"Missing production functions: "
            f"{sorted(missing)}"
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

        "os": os,
        "random": random,

        "torch": torch,
        "umap": umap,

        "SpectralClustering":
            SpectralClustering,

        "adjusted_rand_score":
            adjusted_rand_score,

        "normalized_mutual_info_score":
            normalized_mutual_info_score,

        "v_measure_score":
            v_measure_score,

        "fowlkes_mallows_score":
            fowlkes_mallows_score,

        "silhouette_score":
            silhouette_score,
    }


    exec(
        compile(
            module,
            filename=str(path),
            mode="exec"
        ),
        namespace
    )


    return {
        name: namespace[name]
        for name in wanted
    }, constants


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 8 — PRIMARY K=5 STABILITY")
print("EXACT HBN PRODUCTION STABILITY PROTOCOL")
print("PRIMARY K=5 ALREADY LOCKED")
print("=" * 78)


# ======================================================================
# 1. VERIFY PRIMARY LOCK
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY LOCKED PRIMARY K=5 RESULT")
print("=" * 78)


for p in [
    LATENT_FILE,
    PRIMARY_LABELS_FILE,
    PRIMARY_METRICS_FILE,
    PRIMARY_LOCK,
    STAGE6_SUMMARY,
]:

    if not p.exists():

        raise FileNotFoundError(p)


primary_metrics = json.loads(
    PRIMARY_METRICS_FILE.read_text(
        encoding="utf-8"
    )
)


if primary_metrics.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Primary Stage-7 result is not PASS."
    )


if primary_metrics.get(
    "spectral_k"
) != 5:

    raise RuntimeError(
        "Primary Stage-7 K is not 5."
    )


if primary_metrics.get(
    "k4_k6_inspected_before_primary_lock"
) is not False:

    raise RuntimeError(
        "Primary lock does not certify "
        "pre-sensitivity K=5."
    )


Z = np.load(
    LATENT_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


labels = np.load(
    PRIMARY_LABELS_FILE,
    allow_pickle=False
).astype(
    int,
    copy=False
)


if Z.shape != (
    EXPECTED_SUBJECTS,
    16
):

    raise RuntimeError(
        f"Unexpected latent shape {Z.shape}"
    )


if labels.shape != (
    EXPECTED_SUBJECTS,
):

    raise RuntimeError(
        f"Unexpected labels shape {labels.shape}"
    )


if len(
    np.unique(labels)
) != FORCED_K:

    raise RuntimeError(
        "Primary labels do not contain 5 clusters."
    )


print(
    "LATENT_SHAPE =",
    Z.shape
)

print(
    "PRIMARY_LABEL_SHAPE =",
    labels.shape
)

print(
    "PRIMARY_CLUSTERS =",
    len(
        np.unique(labels)
    )
)

print(
    "PRIMARY_CLUSTER_SIZES =",
    pd.Series(
        labels
    ).value_counts().sort_index().to_dict()
)


# ======================================================================
# 2. VERIFY / LOAD EXACT PRODUCTION FUNCTIONS
# ======================================================================

print()
print("=" * 78)
print("2. LOAD EXACT HBN STABILITY FUNCTIONS")
print("=" * 78)


production_sha = sha256_file(
    PRODUCTION_SCRIPT
)


if (
    production_sha
    != EXPECTED_PRODUCTION_SHA256
):

    raise RuntimeError(
        "Production script SHA256 mismatch."
    )


funcs, constants = (
    load_exact_stability_functions(
        PRODUCTION_SCRIPT
    )
)


print(
    "PRODUCTION_SCRIPT_SHA256 =",
    production_sha
)

print(
    "N_SUBSAMPLE =",
    constants["N_SUBSAMPLE"]
)

print(
    "N_BOOTSTRAP =",
    constants["N_BOOTSTRAP"]
)

print(
    "FORCED_K =",
    constants["FORCED_K"]
)

print(
    "UMAP_MIN_DIST =",
    constants["UMAP_MIN_DIST"]
)

print(
    "UMAP_METRIC =",
    constants["UMAP_METRIC"]
)


subsample_stability = funcs[
    "subsample_stability"
]

bootstrap_stability = funcs[
    "bootstrap_stability"
]


# ======================================================================
# 3. RUN EXACT 80% SUBSAMPLE STABILITY
# ======================================================================

print()
print("=" * 78)
print("3. RUN 100 × 80% SUBSAMPLE STABILITY")
print("=" * 78)

print(
    "This may take several minutes on CPU."
)


sub = subsample_stability(

    Z,

    labels,

    umap_n=UMAP_N,

    graph_k=GRAPH_K,
)


if not isinstance(
    sub,
    pd.DataFrame
):

    sub = pd.DataFrame(
        sub
    )


print(
    "SUBSAMPLE_RUNS =",
    len(sub)
)

print(
    "SUBSAMPLE_COLUMNS =",
    list(
        sub.columns
    )
)


if len(sub) != N_SUBSAMPLE:

    raise RuntimeError(
        f"Expected {N_SUBSAMPLE} subsample runs, "
        f"found {len(sub)}"
    )


# ======================================================================
# 4. RUN EXACT BOOTSTRAP STABILITY
# ======================================================================

print()
print("=" * 78)
print("4. RUN 100 × BOOTSTRAP STABILITY")
print("=" * 78)


boot = bootstrap_stability(

    Z,

    labels,

    umap_n=UMAP_N,

    graph_k=GRAPH_K,
)


if not isinstance(
    boot,
    pd.DataFrame
):

    boot = pd.DataFrame(
        boot
    )


print(
    "BOOTSTRAP_RUNS =",
    len(boot)
)

print(
    "BOOTSTRAP_COLUMNS =",
    list(
        boot.columns
    )
)


if len(boot) == 0:

    raise RuntimeError(
        "No valid bootstrap runs were returned."
    )


# ======================================================================
# 5. SAVE RAW RUNS
# ======================================================================

sub_file = (
    OUT
    / "MIPDB_PRIMARY_K5_SUBSAMPLE_100_RUNS.csv"
)


boot_file = (
    OUT
    / "MIPDB_PRIMARY_K5_BOOTSTRAP_100_RUNS.csv"
)


sub.to_csv(
    sub_file,
    index=False
)


boot.to_csv(
    boot_file,
    index=False
)


# ======================================================================
# 6. SUMMARIZE STABILITY
# ======================================================================

print()
print("=" * 78)
print("5. STABILITY SUMMARY")
print("=" * 78)


def metric_summary(df, name):

    x = pd.to_numeric(
        df[name],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    return {
        "mean":
            float(
                np.nanmean(x)
            ),

        "std":
            float(
                np.nanstd(x)
            ),

        "median":
            float(
                np.nanmedian(x)
            ),

        "min":
            float(
                np.nanmin(x)
            ),

        "max":
            float(
                np.nanmax(x)
            ),

        "n_valid":
            int(
                np.sum(
                    np.isfinite(x)
                )
            ),
    }


sub_metrics = [
    "ARI",
    "NMI",
    "V_measure",
    "Fowlkes_Mallows",
    "pairwise_jaccard",
    "silhouette",
]


boot_metrics = [
    "ARI",
    "NMI",
    "pairwise_jaccard",
    "silhouette",
]


sub_summary = {

    m: metric_summary(
        sub,
        m
    )

    for m in sub_metrics

    if m in sub.columns
}


boot_summary = {

    m: metric_summary(
        boot,
        m
    )

    for m in boot_metrics

    if m in boot.columns
}


for metric in sub_metrics:

    if metric in sub_summary:

        s = sub_summary[
            metric
        ]

        print(
            "SUBSAMPLE",
            metric,
            "MEAN=",
            s["mean"],
            "STD=",
            s["std"],
            "MEDIAN=",
            s["median"],
        )


print()


for metric in boot_metrics:

    if metric in boot_summary:

        s = boot_summary[
            metric
        ]

        print(
            "BOOTSTRAP",
            metric,
            "MEAN=",
            s["mean"],
            "STD=",
            s["std"],
            "MEDIAN=",
            s["median"],
        )


# ======================================================================
# 7. SAVE SUMMARY
# ======================================================================

summary = {

    "validation_branch":
        "MIPDB_EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION",

    "analysis_role":
        "PRIMARY_K5_STABILITY",

    "subjects":
        EXPECTED_SUBJECTS,

    "reference_k":
        FORCED_K,

    "graph_k":
        GRAPH_K,

    "umap_n_neighbors":
        UMAP_N,

    "subsample_fraction":
        0.8,

    "subsample_requested_runs":
        N_SUBSAMPLE,

    "subsample_completed_runs":
        int(
            len(sub)
        ),

    "bootstrap_requested_runs":
        N_BOOTSTRAP,

    "bootstrap_completed_runs":
        int(
            len(boot)
        ),

    "subsample":
        sub_summary,

    "bootstrap":
        boot_summary,

    "production_script":
        str(
            PRODUCTION_SCRIPT
        ),

    "production_script_sha256":
        production_sha,

    "exact_hbn_subsample_function_used":
        True,

    "exact_hbn_bootstrap_function_used":
        True,

    "primary_k_changed":
        False,

    "hyperparameter_tuning_on_mipdb":
        False,

    "status":
        "PASS",
}


summary_file = (
    OUT
    / "MIPDB_PRIMARY_K5_STABILITY_SUMMARY.json"
)


summary_file.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ======================================================================
# 8. PERMANENT LOCK
# ======================================================================

sub_ari = sub_summary[
    "ARI"
]["mean"]

sub_nmi = sub_summary[
    "NMI"
]["mean"]

boot_ari = boot_summary[
    "ARI"
]["mean"]

boot_nmi = boot_summary[
    "NMI"
]["mean"]


lock_file = (
    AUDIT
    / "MIPDB_STAGE8_PRIMARY_K5_STABILITY_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 8 STABILITY LOCK
======================================================================

VALIDATION_BRANCH=EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION

REFERENCE_ANALYSIS=LOCKED_PRIMARY_K5

SUBJECTS=91
PRIMARY_K=5
GRAPH_K=12
UMAP_N_NEIGHBORS=30

SUBSAMPLE_RUNS_REQUESTED=100
SUBSAMPLE_RUNS_COMPLETED={len(sub)}
SUBSAMPLE_FRACTION=0.8

SUBSAMPLE_ARI_MEAN={sub_ari}
SUBSAMPLE_NMI_MEAN={sub_nmi}

BOOTSTRAP_RUNS_REQUESTED=100
BOOTSTRAP_RUNS_COMPLETED={len(boot)}

BOOTSTRAP_ARI_MEAN={boot_ari}
BOOTSTRAP_NMI_MEAN={boot_nmi}

EXACT_HBN_SUBSAMPLE_FUNCTION_USED=YES
EXACT_HBN_BOOTSTRAP_FUNCTION_USED=YES

PRIMARY_K_CHANGED=NO
HYPERPARAMETER_TUNING_ON_MIPDB=NO

SUBSAMPLE_FILE=
{sub_file}

BOOTSTRAP_FILE=
{boot_file}

SUMMARY_FILE=
{summary_file}

STAGE8_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# 9. SHA256 MANIFEST
# ======================================================================

sha_file = (
    AUDIT
    / "MIPDB_STAGE8_PRIMARY_K5_STABILITY_SHA256.txt"
)


with open(
    sha_file,
    "w"
) as f:

    for p in [
        sub_file,
        boot_file,
        summary_file,
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
print("STAGE 8 COMPLETE")
print("=" * 78)

print(
    "PRIMARY_K=5"
)

print(
    "SUBSAMPLE_RUNS=",
    len(sub)
)

print(
    "SUBSAMPLE_ARI_MEAN=",
    sub_ari
)

print(
    "SUBSAMPLE_NMI_MEAN=",
    sub_nmi
)

print(
    "BOOTSTRAP_RUNS=",
    len(boot)
)

print(
    "BOOTSTRAP_ARI_MEAN=",
    boot_ari
)

print(
    "BOOTSTRAP_NMI_MEAN=",
    boot_nmi
)

print(
    "PRIMARY_K_CHANGED=NO"
)

print(
    "HYPERPARAMETER_TUNING_ON_MIPDB=NO"
)

print(
    "STAGE8_STATUS=PASS"
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

