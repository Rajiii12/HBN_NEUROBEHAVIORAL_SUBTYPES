#!/usr/bin/env python3

from pathlib import Path
import ast
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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

STAGE4 = (
    EXT
    / "04_FEATURES"
    / "MIPDB_HBN_FROZEN_TRANSFORM_PRIMARY91"
)

EEG_FILE = (
    STAGE4
    / "MIPDB_SUBJECT_EEG192_FROZEN_HBN.npy"
)

META_FILE = (
    STAGE4
    / "MIPDB_SUBJECT_METADATA.csv"
)

STAGE4_SUMMARY = (
    AUDIT
    / "MIPDB_STAGE4_FROZEN_HBN_TRANSFORM_SUMMARY.json"
)

PRODUCTION_GRAPH_SCRIPT = (
    ROOT
    / "hbn-eeg-neurobehavioral-subtypes"
    / "scripts"
    / "07_graph_construction_final_graphK12_alpha06.py"
)

OUT = (
    EXT
    / "06_EXTERNAL_GRAPH"
    / "EEG_ONLY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

K_G = 12
EXPECTED_SUBJECTS = 91
EXPECTED_DIM = 192


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


def load_exact_graph_function(script_path):

    """
    Extract ONLY build_knn_adjacency() from the frozen
    production graph script.

    The remainder of the original script is NOT executed.
    """

    source = script_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(script_path)
    )

    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_knn_adjacency"
    ]

    if len(funcs) != 1:

        raise RuntimeError(
            "Could not uniquely locate "
            "build_knn_adjacency()"
        )

    module = ast.Module(
        body=funcs,
        type_ignores=[]
    )

    ast.fix_missing_locations(module)

    namespace = {
        "np": np,
        "kneighbors_graph": kneighbors_graph,
    }

    exec(
        compile(
            module,
            filename=str(script_path),
            mode="exec"
        ),
        namespace
    )

    return namespace[
        "build_knn_adjacency"
    ]


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 5 — EEG-ONLY SUBJECT GRAPH")
print("FROZEN GRAPH K = 12")
print("NO PHENOTYPE FUSION")
print("NO HYPERPARAMETER TUNING")
print("=" * 78)


# ======================================================================
# 1. VERIFY STAGE-4 INPUT
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY STAGE-4 SUBJECT REPRESENTATION")
print("=" * 78)

if not EEG_FILE.exists():
    raise FileNotFoundError(EEG_FILE)

if not META_FILE.exists():
    raise FileNotFoundError(META_FILE)

if not STAGE4_SUMMARY.exists():
    raise FileNotFoundError(STAGE4_SUMMARY)


summary4 = json.loads(
    STAGE4_SUMMARY.read_text(
        encoding="utf-8"
    )
)


if summary4.get("status") != "PASS":

    raise RuntimeError(
        "Stage-4 summary is not PASS."
    )


X_raw = np.load(
    EEG_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


meta = pd.read_csv(
    META_FILE
)


print(
    "RAW_EEG_SHAPE =",
    X_raw.shape
)

print(
    "METADATA_ROWS =",
    len(meta)
)

print(
    "RAW_EEG_FINITE =",
    bool(
        np.isfinite(X_raw).all()
    )
)


if X_raw.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_DIM
):

    raise RuntimeError(
        f"Expected {(EXPECTED_SUBJECTS, EXPECTED_DIM)}, "
        f"found {X_raw.shape}"
    )


if len(meta) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Metadata row count mismatch."
    )


if meta["subject"].duplicated().any():

    raise RuntimeError(
        "Duplicate subject IDs found."
    )


expected_sha = summary4.get(
    "subject_eeg192_sha256"
)

got_sha = sha256_file(
    EEG_FILE
)


if expected_sha != got_sha:

    raise RuntimeError(
        "Stage-4 subject EEG192 SHA256 mismatch."
    )


print(
    "STAGE4_INPUT_SHA256=PASS"
)


# ======================================================================
# 2. LOAD EXACT PRODUCTION GRAPH FUNCTION
# ======================================================================

print()
print("=" * 78)
print("2. LOAD EXACT HBN GRAPH-CONSTRUCTION FUNCTION")
print("=" * 78)

if not PRODUCTION_GRAPH_SCRIPT.exists():

    raise FileNotFoundError(
        PRODUCTION_GRAPH_SCRIPT
    )


build_knn_adjacency = (
    load_exact_graph_function(
        PRODUCTION_GRAPH_SCRIPT
    )
)


graph_script_sha = sha256_file(
    PRODUCTION_GRAPH_SCRIPT
)


print(
    "PRODUCTION_SCRIPT =",
    PRODUCTION_GRAPH_SCRIPT
)

print(
    "PRODUCTION_SCRIPT_SHA256 =",
    graph_script_sha
)

print(
    "FUNCTION = build_knn_adjacency"
)

print(
    "FUNCTION_SOURCE=HBN_FINAL_PRODUCTION"
)


# ======================================================================
# 3. APPLY EXACT GRAPH-LEVEL STANDARDIZATION
# ======================================================================

print()
print("=" * 78)
print("3. STANDARDIZE MIPDB SUBJECT FEATURES")
print("=" * 78)

print(
    "NOTE: this StandardScaler is part of the "
    "fixed graph-construction rule."
)

print(
    "It is NOT CNN/PCA refitting and "
    "does NOT use labels or outcomes."
)


graph_scaler = StandardScaler()

X_eeg = graph_scaler.fit_transform(
    X_raw
).astype(
    np.float32
)


if X_eeg.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_DIM
):

    raise RuntimeError(
        "Standardized EEG shape failure."
    )


if not np.isfinite(
    X_eeg
).all():

    raise RuntimeError(
        "Non-finite standardized EEG."
    )


print(
    "STANDARDIZED_SHAPE =",
    X_eeg.shape
)

print(
    "GLOBAL_MEAN =",
    float(X_eeg.mean())
)

print(
    "GLOBAL_STD =",
    float(X_eeg.std())
)

print(
    "ACTIVE_FEATURES =",
    int(
        np.sum(
            np.std(
                X_eeg,
                axis=0
            ) > 1e-8
        )
    ),
    "/",
    EXPECTED_DIM
)


# ======================================================================
# 4. EXACT EEG COSINE SIMILARITY
# ======================================================================

print()
print("=" * 78)
print("4. COMPUTE EEG COSINE SIMILARITY")
print("=" * 78)


eeg_similarity = cosine_similarity(
    X_eeg
).astype(
    np.float32
)


if eeg_similarity.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_SUBJECTS
):

    raise RuntimeError(
        "Similarity shape failure."
    )


if not np.isfinite(
    eeg_similarity
).all():

    raise RuntimeError(
        "Similarity contains non-finite values."
    )


sym_error = float(
    np.max(
        np.abs(
            eeg_similarity
            - eeg_similarity.T
        )
    )
)


print(
    "SIMILARITY_SHAPE =",
    eeg_similarity.shape
)

print(
    "SIMILARITY_MIN =",
    float(
        eeg_similarity.min()
    )
)

print(
    "SIMILARITY_MAX =",
    float(
        eeg_similarity.max()
    )
)

print(
    "SIMILARITY_MEAN =",
    float(
        eeg_similarity.mean()
    )
)

print(
    "SIMILARITY_SYMMETRY_MAX_ERROR =",
    sym_error
)


# ======================================================================
# 5. EXACT PRODUCTION KNN GRAPH — K_G=12
# ======================================================================

print()
print("=" * 78)
print("5. BUILD EEG-ONLY KNN GRAPH")
print("=" * 78)

print(
    "K_G =",
    K_G
)


adjacency = build_knn_adjacency(
    eeg_similarity,
    k=K_G
)


if hasattr(
    adjacency,
    "toarray"
):

    adjacency = adjacency.toarray()


adjacency = np.asarray(
    adjacency,
    dtype=np.float32
)


if adjacency.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_SUBJECTS
):

    raise RuntimeError(
        f"Adjacency shape failure: "
        f"{adjacency.shape}"
    )


if not np.isfinite(
    adjacency
).all():

    raise RuntimeError(
        "Adjacency contains non-finite values."
    )


adj_sym_error = float(
    np.max(
        np.abs(
            adjacency
            - adjacency.T
        )
    )
)


diag_max = float(
    np.max(
        np.abs(
            np.diag(adjacency)
        )
    )
)


unique_adj = np.unique(
    adjacency
)


print(
    "ADJACENCY_SHAPE =",
    adjacency.shape
)

print(
    "ADJACENCY_UNIQUE_VALUES =",
    unique_adj.tolist()
)

print(
    "ADJACENCY_SYMMETRY_MAX_ERROR =",
    adj_sym_error
)

print(
    "ADJACENCY_DIAGONAL_MAX =",
    diag_max
)


# ======================================================================
# 6. GRAPH DIAGNOSTICS
# ======================================================================

print()
print("=" * 78)
print("6. GRAPH DIAGNOSTICS")
print("=" * 78)


degree = adjacency.sum(
    axis=1
)


adjacency_entries = int(
    adjacency.sum()
)


if adj_sym_error <= 1e-8:

    undirected_edges = int(
        adjacency_entries // 2
    )

else:

    undirected_edges = None


n_components, component_labels = (
    connected_components(
        csr_matrix(adjacency),
        directed=False,
        return_labels=True
    )
)


component_sizes = np.bincount(
    component_labels
)


print(
    "ADJACENCY_ENTRIES =",
    adjacency_entries
)

print(
    "UNDIRECTED_EDGES =",
    undirected_edges
)

print(
    "MIN_DEGREE =",
    float(degree.min())
)

print(
    "MEAN_DEGREE =",
    float(degree.mean())
)

print(
    "MAX_DEGREE =",
    float(degree.max())
)

print(
    "CONNECTED_COMPONENTS =",
    int(n_components)
)

print(
    "COMPONENT_SIZES =",
    component_sizes.tolist()
)

print(
    "ISOLATED_NODES =",
    int(
        np.sum(
            degree == 0
        )
    )
)


# ======================================================================
# 7. SAVE GRAPH ARTIFACTS
# ======================================================================

print()
print("=" * 78)
print("7. SAVE STAGE-5 GRAPH ARTIFACTS")
print("=" * 78)


node_file = (
    OUT
    / "node_features.npy"
)

sim_file = (
    OUT
    / "eeg_similarity.npy"
)

adj_file = (
    OUT
    / "adjacency.npy"
)

meta_out = (
    OUT
    / "subject_metadata.csv"
)

scaler_file = (
    OUT
    / "MIPDB_GRAPH_LEVEL_STANDARD_SCALER.joblib"
)


np.save(
    node_file,
    X_eeg.astype(
        np.float32
    ),
    allow_pickle=False
)

np.save(
    sim_file,
    eeg_similarity.astype(
        np.float32
    ),
    allow_pickle=False
)

np.save(
    adj_file,
    adjacency.astype(
        np.float32
    ),
    allow_pickle=False
)

meta.to_csv(
    meta_out,
    index=False
)

joblib.dump(
    graph_scaler,
    scaler_file
)


# ======================================================================
# 8. SAVE GRAPH SUMMARY
# ======================================================================

graph_summary = {

    "experiment":
        "MIPDB_EEG_ONLY_EXTERNAL_VALIDATION",

    "subjects":
        EXPECTED_SUBJECTS,

    "node_feature_dimension":
        EXPECTED_DIM,

    "graph_k":
        K_G,

    "phenotype_view_used":
        False,

    "alpha_used":
        False,

    "similarity":
        "cosine_similarity",

    "knn_input":
        "similarity_rows",

    "knn_mode":
        "connectivity",

    "graph_function_source":
        str(
            PRODUCTION_GRAPH_SCRIPT
        ),

    "graph_function_source_sha256":
        graph_script_sha,

    "adjacency_entries":
        adjacency_entries,

    "undirected_edges":
        undirected_edges,

    "minimum_degree":
        float(
            degree.min()
        ),

    "mean_degree":
        float(
            degree.mean()
        ),

    "maximum_degree":
        float(
            degree.max()
        ),

    "connected_components":
        int(
            n_components
        ),

    "component_sizes":
        component_sizes.tolist(),

    "isolated_nodes":
        int(
            np.sum(
                degree == 0
            )
        ),

    "node_features_sha256":
        sha256_file(
            node_file
        ),

    "eeg_similarity_sha256":
        sha256_file(
            sim_file
        ),

    "adjacency_sha256":
        sha256_file(
            adj_file
        ),

    "metadata_sha256":
        sha256_file(
            meta_out
        ),

    "graph_scaler_sha256":
        sha256_file(
            scaler_file
        ),

    "hyperparameter_selection_on_mipdb":
        False,

    "status":
        "PASS",
}


summary_file = (
    AUDIT
    / "MIPDB_STAGE5_EEG_ONLY_GRAPH_SUMMARY.json"
)


summary_file.write_text(
    json.dumps(
        graph_summary,
        indent=2
    ),
    encoding="utf-8"
)


# ======================================================================
# 9. PERMANENT LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE5_EEG_ONLY_GRAPH_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 5 GRAPH LOCK
======================================================================

VALIDATION_BRANCH=EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION

SUBJECTS=91
NODE_FEATURE_DIMENSION=192

GRAPH_STANDARDIZATION=StandardScaler
GRAPH_STANDARDIZATION_SCOPE=MIPDB_EXTERNAL_COHORT
GRAPH_STANDARDIZATION_IS_FIXED_PIPELINE_STEP=YES

SIMILARITY=cosine_similarity
GRAPH_K=12

KNN_INPUT=similarity_rows
KNN_MODE=connectivity
INCLUDE_SELF=NO

PHENOTYPE_VIEW_USED=NO
ALPHA_USED=NO

GRAPH_HYPERPARAMETER_TUNING_ON_MIPDB=NO
GRAPH_K_RESELECTED_ON_MIPDB=NO

ADJACENCY_ENTRIES={adjacency_entries}
UNDIRECTED_EDGES={undirected_edges}

MIN_DEGREE={float(degree.min())}
MEAN_DEGREE={float(degree.mean())}
MAX_DEGREE={float(degree.max())}

CONNECTED_COMPONENTS={int(n_components)}
ISOLATED_NODES={int(np.sum(degree == 0))}

NODE_FEATURES_FILE=
{node_file}

EEG_SIMILARITY_FILE=
{sim_file}

ADJACENCY_FILE=
{adj_file}

SUBJECT_METADATA_FILE=
{meta_out}

PRODUCTION_GRAPH_SCRIPT=
{PRODUCTION_GRAPH_SCRIPT}

PRODUCTION_GRAPH_SCRIPT_SHA256=
{graph_script_sha}

STAGE5_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# 10. SHA256 MANIFEST
# ======================================================================

files = [
    node_file,
    sim_file,
    adj_file,
    meta_out,
    scaler_file,
    summary_file,
    lock_file,
]


hash_manifest = (
    AUDIT
    / "MIPDB_STAGE5_EEG_ONLY_GRAPH_SHA256.txt"
)


with open(
    hash_manifest,
    "w"
) as f:

    for p in files:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 78)
print("STAGE 5 COMPLETE")
print("=" * 78)

print(
    "SUBJECTS=91"
)

print(
    "NODE_FEATURES_SHAPE=",
    X_eeg.shape
)

print(
    "EEG_SIMILARITY_SHAPE=",
    eeg_similarity.shape
)

print(
    "ADJACENCY_SHAPE=",
    adjacency.shape
)

print(
    "GRAPH_K=12"
)

print(
    "PHENOTYPE_VIEW_USED=NO"
)

print(
    "ALPHA_USED=NO"
)

print(
    "CONNECTED_COMPONENTS=",
    int(n_components)
)

print(
    "ISOLATED_NODES=",
    int(
        np.sum(
            degree == 0
        )
    )
)

print(
    "HYPERPARAMETER_TUNING_ON_MIPDB=NO"
)

print(
    "STAGE5_STATUS=PASS"
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
