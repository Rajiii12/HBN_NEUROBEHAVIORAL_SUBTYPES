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
import sklearn
import umap

from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)

from scipy.spatial.distance import cdist


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

STAGE6_SUMMARY = (
    AUDIT
    / "MIPDB_STAGE6_VGAE_FIXED_CONFIG_SUMMARY.json"
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
    / "08_CLUSTERING"
    / "EEG_ONLY"
    / "PRIMARY_K5"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# FROZEN PRIMARY CONFIGURATION
# ======================================================================

EXPECTED_SUBJECTS = 91

SEED = 42

GRAPH_K = 12

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

FORCED_K = 5
SPECTRAL_N_NEIGHBORS = 12


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
                    result[
                        target.id
                    ] = value


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


def load_exact_clustering_functions(
    path
):

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


    # --------------------------------------------------------------
    # Verify the key frozen production constants
    # --------------------------------------------------------------

    if constants.get("SEED") != SEED:

        raise RuntimeError(
            "Production SEED is not 42."
        )


    if constants.get(
        "UMAP_N_GRID"
    ) != [30]:

        raise RuntimeError(
            "Production UMAP_N_GRID is not [30]."
        )


    if constants.get(
        "UMAP_MIN_DIST"
    ) != UMAP_MIN_DIST:

        raise RuntimeError(
            "Production UMAP_MIN_DIST mismatch."
        )


    if constants.get(
        "UMAP_METRIC"
    ) != UMAP_METRIC:

        raise RuntimeError(
            "Production UMAP_METRIC mismatch."
        )


    # FORCED_K should exist in the final production script.
    if (
        "FORCED_K" in constants
        and constants["FORCED_K"] != FORCED_K
    ):

        raise RuntimeError(
            "Production FORCED_K is not 5."
        )


    wanted = {
        "set_seed",
        "run_umap_projection",
        "run_spectral",
        "dunn_index",
    }


    function_nodes = [

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
        for node in function_nodes
    }


    required = {
        "run_umap_projection",
        "run_spectral",
    }


    if not required.issubset(
        found
    ):

        raise RuntimeError(
            "Required production clustering "
            f"functions missing: {found}"
        )


    module = ast.Module(
        body=sorted(
            function_nodes,
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

        "random": random,

        "os": os,

        "torch": torch,

        "umap": umap,

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
            "run_umap_projection"
        ],

        namespace[
            "run_spectral"
        ],

        namespace.get(
            "set_seed"
        ),

        namespace.get(
            "dunn_index"
        ),

        constants,
    )


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 7 — PRIMARY EXTERNAL K=5 CLUSTERING")
print("VGAE16 -> UMAP30 -> SPECTRAL K=5")
print("NO K SELECTION ON MIPDB")
print("NO HYPERPARAMETER TUNING")
print("=" * 78)


# ======================================================================
# 1. VERIFY STAGE 6
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY LOCKED STAGE-6 VGAE")
print("=" * 78)


if not LATENT_FILE.exists():

    raise FileNotFoundError(
        LATENT_FILE
    )


if not STAGE6_SUMMARY.exists():

    raise FileNotFoundError(
        STAGE6_SUMMARY
    )


if not META_FILE.exists():

    raise FileNotFoundError(
        META_FILE
    )


stage6 = json.loads(
    STAGE6_SUMMARY.read_text(
        encoding="utf-8"
    )
)


if stage6.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Stage-6 VGAE is not PASS."
    )


if stage6.get(
    "latent_dimension"
) != 16:

    raise RuntimeError(
        "Stage-6 latent dimension is not 16."
    )


Z = np.load(
    LATENT_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


meta = pd.read_csv(
    META_FILE
)


if Z.shape != (
    EXPECTED_SUBJECTS,
    16
):

    raise RuntimeError(
        f"Unexpected latent shape {Z.shape}"
    )


if len(meta) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Metadata row count mismatch."
    )


if not np.isfinite(
    Z
).all():

    raise RuntimeError(
        "Latent representation contains "
        "non-finite values."
    )


expected_latent_sha = stage6.get(
    "latent_sha256"
)


if (
    expected_latent_sha is not None
    and sha256_file(
        LATENT_FILE
    ) != expected_latent_sha
):

    raise RuntimeError(
        "Stage-6 latent SHA256 mismatch."
    )


print(
    "LATENT_SHAPE =",
    Z.shape
)

print(
    "LATENT_FINITE =",
    True
)

print(
    "ACTIVE_LATENT_DIMENSIONS =",
    int(
        np.sum(
            np.std(
                Z,
                axis=0
            ) > 1e-8
        )
    )
)


# ======================================================================
# 2. VERIFY FROZEN PRODUCTION SCRIPT
# ======================================================================

print()
print("=" * 78)
print("2. LOAD EXACT HBN UMAP / SPECTRAL FUNCTIONS")
print("=" * 78)


production_sha = sha256_file(
    PRODUCTION_SCRIPT
)


print(
    "PRODUCTION_SCRIPT_SHA256 =",
    production_sha
)


if (
    production_sha
    != EXPECTED_PRODUCTION_SHA256
):

    raise RuntimeError(
        "Frozen HBN production script "
        "SHA256 changed."
    )


(
    run_umap_projection,
    run_spectral,
    set_seed,
    production_dunn_index,
    constants,

) = load_exact_clustering_functions(
    PRODUCTION_SCRIPT
)


print(
    "SEED =",
    SEED
)

print(
    "UMAP_N_NEIGHBORS =",
    UMAP_N_NEIGHBORS
)

print(
    "UMAP_MIN_DIST =",
    UMAP_MIN_DIST
)

print(
    "UMAP_METRIC =",
    UMAP_METRIC
)

print(
    "PRIMARY_K =",
    FORCED_K
)

print(
    "SPECTRAL_N_NEIGHBORS =",
    SPECTRAL_N_NEIGHBORS
)


# ======================================================================
# 3. FIX RANDOM STATE
# ======================================================================

if set_seed is not None:

    set_seed(
        SEED
    )

else:

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )


# ======================================================================
# 4. EXACT PRODUCTION UMAP
# ======================================================================

print()
print("=" * 78)
print("3. PRIMARY UMAP PROJECTION")
print("=" * 78)


X2 = run_umap_projection(

    Z,

    seed=SEED,

    n_neighbors=
        UMAP_N_NEIGHBORS,
)


X2 = np.asarray(
    X2,
    dtype=np.float32
)


if X2.shape != (
    EXPECTED_SUBJECTS,
    2
):

    raise RuntimeError(
        f"Unexpected UMAP shape {X2.shape}"
    )


if not np.isfinite(
    X2
).all():

    raise RuntimeError(
        "UMAP contains non-finite values."
    )


print(
    "UMAP_SHAPE =",
    X2.shape
)

print(
    "UMAP1_RANGE =",
    float(
        X2[:, 0].min()
    ),
    "to",
    float(
        X2[:, 0].max()
    )
)

print(
    "UMAP2_RANGE =",
    float(
        X2[:, 1].min()
    ),
    "to",
    float(
        X2[:, 1].max()
    )
)


# ======================================================================
# 5. PRIMARY FROZEN K=5 SPECTRAL CLUSTERING
# ======================================================================

print()
print("=" * 78)
print("4. PRIMARY SPECTRAL CLUSTERING — K=5")
print("=" * 78)


labels = run_spectral(

    X2,

    k=FORCED_K,

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
        f"Unexpected labels shape "
        f"{labels.shape}"
    )


unique_labels = np.unique(
    labels
)


if len(
    unique_labels
) != FORCED_K:

    raise RuntimeError(
        f"Expected {FORCED_K} clusters, "
        f"found {len(unique_labels)}"
    )


print(
    "CLUSTERS_FOUND =",
    len(unique_labels)
)

print(
    "RAW_CLUSTER_LABELS =",
    unique_labels.tolist()
)


# ======================================================================
# 6. PRIMARY QUALITY METRICS
# ======================================================================

print()
print("=" * 78)
print("5. PRIMARY EXTERNAL CLUSTER QUALITY")
print("=" * 78)


silhouette = float(
    silhouette_score(
        X2,
        labels
    )
)


davies_bouldin = float(
    davies_bouldin_score(
        X2,
        labels
    )
)


calinski_harabasz = float(
    calinski_harabasz_score(
        X2,
        labels
    )
)


if production_dunn_index is not None:

    dunn = float(
        production_dunn_index(
            X2,
            labels
        )
    )

else:

    # Fallback used only if the production function
    # cannot be extracted.
    unique = np.unique(
        labels
    )

    max_intra = 0.0

    min_inter = np.inf

    for i, ci in enumerate(
        unique
    ):

        a = X2[
            labels == ci
        ]

        if len(a) > 1:

            intra = pairwise_distances(
                a
            )

            max_intra = max(
                max_intra,
                float(
                    np.max(
                        intra
                    )
                )
            )


        for cj in unique[
            i + 1:
        ]:

            b = X2[
                labels == cj
            ]

            inter = pairwise_distances(
                a,
                b
            )

            min_inter = min(
                min_inter,
                float(
                    np.min(
                        inter
                    )
                )
            )


    dunn = (
        float(
            min_inter
            / max_intra
        )
        if max_intra > 0
        else float("nan")
    )


counts = pd.Series(
    labels
).value_counts().sort_index()


min_cluster = int(
    counts.min()
)

max_cluster = int(
    counts.max()
)

balance_ratio = float(
    max_cluster
    / min_cluster
)


print(
    "SILHOUETTE =",
    silhouette
)

print(
    "DAVIES_BOULDIN =",
    davies_bouldin
)

print(
    "CALINSKI_HARABASZ =",
    calinski_harabasz
)

print(
    "DUNN_INDEX =",
    dunn
)

print()
print(
    "CLUSTER_SIZES =",
    counts.to_dict()
)

print(
    "MIN_CLUSTER_SIZE =",
    min_cluster
)

print(
    "MAX_CLUSTER_SIZE =",
    max_cluster
)

print(
    "BALANCE_RATIO_MAX_MIN =",
    balance_ratio
)


# ======================================================================
# 7. SAVE PRIMARY OUTPUTS
# ======================================================================

print()
print("=" * 78)
print("6. SAVE PRIMARY K=5 RESULTS")
print("=" * 78)


umap_npy = (
    OUT
    / "MIPDB_PRIMARY_UMAP30.npy"
)


labels_npy = (
    OUT
    / "MIPDB_PRIMARY_K5_LABELS.npy"
)


assign_file = (
    OUT
    / "MIPDB_PRIMARY_K5_SUBTYPE_ASSIGNMENTS.csv"
)


metrics_file = (
    OUT
    / "MIPDB_PRIMARY_K5_METRICS.json"
)


cluster_sizes_file = (
    OUT
    / "MIPDB_PRIMARY_K5_CLUSTER_SIZES.csv"
)


np.save(
    umap_npy,
    X2.astype(
        np.float32
    ),
    allow_pickle=False
)


np.save(
    labels_npy,
    labels.astype(
        np.int64
    ),
    allow_pickle=False
)


assign = pd.DataFrame({

    "subject":
        meta[
            "subject"
        ].values,

    # Preserve sklearn's raw 0-based label.
    "cluster_zero_based":
        labels,

    # Human-readable subtype numbering.
    "subtype":
        labels + 1,

    "UMAP1":
        X2[:, 0],

    "UMAP2":
        X2[:, 1],
})


assign.to_csv(
    assign_file,
    index=False
)


cluster_sizes = pd.DataFrame({

    "cluster_zero_based":
        counts.index.astype(
            int
        ),

    "subtype":
        counts.index.astype(
            int
        ) + 1,

    "n_subjects":
        counts.values.astype(
            int
        ),
})


cluster_sizes.to_csv(
    cluster_sizes_file,
    index=False
)


# ======================================================================
# 8. METRICS JSON
# ======================================================================

metrics = {

    "validation_branch":
        "MIPDB_EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION",

    "analysis_role":
        "PRIMARY_EXTERNAL_K5",

    "subjects":
        EXPECTED_SUBJECTS,

    "vgae_latent_dimension":
        16,

    "graph_k":
        GRAPH_K,

    "seed":
        SEED,

    "umap_n_neighbors":
        UMAP_N_NEIGHBORS,

    "umap_min_dist":
        UMAP_MIN_DIST,

    "umap_metric":
        UMAP_METRIC,

    "spectral_k":
        FORCED_K,

    "spectral_affinity":
        "nearest_neighbors",

    "spectral_n_neighbors":
        SPECTRAL_N_NEIGHBORS,

    "spectral_assign_labels":
        "kmeans",

    "silhouette":
        silhouette,

    "davies_bouldin":
        davies_bouldin,

    "calinski_harabasz":
        calinski_harabasz,

    "dunn_index":
        dunn,

    "cluster_sizes":
        {
            str(
                int(k) + 1
            ):
            int(v)

            for k, v
            in counts.items()
        },

    "minimum_cluster_size":
        min_cluster,

    "maximum_cluster_size":
        max_cluster,

    "balance_ratio_max_min":
        balance_ratio,

    "number_clusters_found":
        int(
            len(
                unique_labels
            )
        ),

    "production_script":
        str(
            PRODUCTION_SCRIPT
        ),

    "production_script_sha256":
        production_sha,

    "exact_production_umap_function_used":
        True,

    "exact_production_spectral_function_used":
        True,

    "mipdb_k_selection_performed":
        False,

    "mipdb_hyperparameter_tuning":
        False,

    "k4_k6_inspected_before_primary_lock":
        False,

    "status":
        "PASS",
}


metrics_file.write_text(
    json.dumps(
        metrics,
        indent=2
    ),
    encoding="utf-8"
)


# ======================================================================
# 9. PRIMARY LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE7_PRIMARY_K5_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 7 PRIMARY K=5 LOCK
======================================================================

VALIDATION_BRANCH=EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION

PRIMARY_ANALYSIS=YES
PRIMARY_K=5

SUBJECTS=91

INPUT=VGAE_LATENT_16D

GRAPH_K=12

SEED=42

UMAP_N_NEIGHBORS=30
UMAP_MIN_DIST=0.1
UMAP_METRIC=euclidean

SPECTRAL_K=5
SPECTRAL_AFFINITY=nearest_neighbors
SPECTRAL_N_NEIGHBORS=12
SPECTRAL_ASSIGN_LABELS=kmeans

SILHOUETTE={silhouette}
DAVIES_BOULDIN={davies_bouldin}
CALINSKI_HARABASZ={calinski_harabasz}
DUNN_INDEX={dunn}

MIN_CLUSTER_SIZE={min_cluster}
MAX_CLUSTER_SIZE={max_cluster}
BALANCE_RATIO_MAX_MIN={balance_ratio}

CLUSTER_SIZES={counts.to_dict()}

K_SELECTED_USING_MIPDB=NO
HYPERPARAMETER_TUNING_ON_MIPDB=NO

K4_K6_INSPECTED_BEFORE_PRIMARY_LOCK=NO

ASSIGNMENT_FILE=
{assign_file}

METRICS_FILE=
{metrics_file}

PRODUCTION_SCRIPT=
{PRODUCTION_SCRIPT}

PRODUCTION_SCRIPT_SHA256=
{production_sha}

STAGE7_PRIMARY_K5_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# 10. SHA256 MANIFEST
# ======================================================================

sha_file = (
    AUDIT
    / "MIPDB_STAGE7_PRIMARY_K5_SHA256.txt"
)


outputs = [

    umap_npy,
    labels_npy,
    assign_file,
    cluster_sizes_file,
    metrics_file,
    lock_file,
]


with open(
    sha_file,
    "w"
) as f:

    for p in outputs:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 78)
print("STAGE 7 PRIMARY K=5 COMPLETE")
print("=" * 78)

print(
    "SUBJECTS=91"
)

print(
    "VGAE_LATENT_SHAPE=",
    Z.shape
)

print(
    "UMAP_SHAPE=",
    X2.shape
)

print(
    "PRIMARY_K=5"
)

print(
    "CLUSTERS_FOUND=",
    len(
        unique_labels
    )
)

print(
    "CLUSTER_SIZES=",
    counts.to_dict()
)

print()
print(
    "SILHOUETTE=",
    silhouette
)

print(
    "DAVIES_BOULDIN=",
    davies_bouldin
)

print(
    "CALINSKI_HARABASZ=",
    calinski_harabasz
)

print(
    "DUNN_INDEX=",
    dunn
)

print(
    "MIN_CLUSTER_SIZE=",
    min_cluster
)

print(
    "BALANCE_RATIO_MAX_MIN=",
    balance_ratio
)

print()
print(
    "K_SELECTED_USING_MIPDB=NO"
)

print(
    "HYPERPARAMETER_TUNING_ON_MIPDB=NO"
)

print(
    "K4_K6_INSPECTED_BEFORE_PRIMARY_LOCK=NO"
)

print(
    "STAGE7_PRIMARY_K5_STATUS=PASS"
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

