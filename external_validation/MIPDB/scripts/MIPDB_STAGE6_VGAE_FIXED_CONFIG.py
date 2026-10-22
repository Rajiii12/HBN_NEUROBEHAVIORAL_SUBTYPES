#!/usr/bin/env python3

from pathlib import Path
import ast
import hashlib
import json
import os
import random
import sys

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

ROOT = (
    DRIVE
    / "NEW_TASK_MULTIVIEW_V2"
)

EXT = (
    DRIVE
    / "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
    / "EXTERNAL_VALIDATION_MIPDB"
)

AUDIT = EXT / "00_AUDIT"

GRAPH = (
    EXT
    / "06_EXTERNAL_GRAPH"
    / "EEG_ONLY"
)

NODE_FILE = (
    GRAPH
    / "node_features.npy"
)

ADJ_FILE = (
    GRAPH
    / "adjacency.npy"
)

META_FILE = (
    GRAPH
    / "subject_metadata.csv"
)

STAGE5_SUMMARY = (
    AUDIT
    / "MIPDB_STAGE5_EEG_ONLY_GRAPH_SUMMARY.json"
)

FINAL_HBN_SCRIPT = (
    ROOT
    / "hbn-eeg-neurobehavioral-subtypes"
    / "scripts"
    / "09_final_forced_k5_validation_graphK12_alpha06_umap30.py"
)

OUT = (
    EXT
    / "07_VGAE"
    / "EEG_ONLY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# FIXED CONFIGURATION
# ======================================================================

EXPECTED_SUBJECTS = 91
EXPECTED_INPUT_DIM = 192

EXPECTED_SEED = 42
EXPECTED_EPOCHS = 500
EXPECTED_LR = 0.001
EXPECTED_HIDDEN = 64
EXPECTED_LATENT = 16


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


def literal_constant(tree, name):

    for node in tree.body:

        if isinstance(
            node,
            (ast.Assign, ast.AnnAssign)
        ):

            targets = []

            if isinstance(
                node,
                ast.Assign
            ):

                targets = node.targets

                value = node.value

            else:

                targets = [
                    node.target
                ]

                value = node.value


            for target in targets:

                if (
                    isinstance(
                        target,
                        ast.Name
                    )
                    and target.id == name
                ):

                    try:

                        return ast.literal_eval(
                            value
                        )

                    except Exception:

                        return None

    return None


def load_exact_vgae_components(
    script_path
):

    """
    Load exact VGAE model/training definitions from the final
    HBN production script without executing its main workflow.
    """

    source = script_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            script_path
        )
    )


    # --------------------------------------------------------------
    # Confirm frozen constants directly from production source
    # --------------------------------------------------------------

    constants = {

        "SEED":
            literal_constant(
                tree,
                "SEED"
            ),

        "VGAE_EPOCHS":
            literal_constant(
                tree,
                "VGAE_EPOCHS"
            ),

        "LR":
            literal_constant(
                tree,
                "LR"
            ),

        "HIDDEN_DIM":
            literal_constant(
                tree,
                "HIDDEN_DIM"
            ),

        "LATENT_DIM":
            literal_constant(
                tree,
                "LATENT_DIM"
            ),
    }


    expected = {

        "SEED":
            EXPECTED_SEED,

        "VGAE_EPOCHS":
            EXPECTED_EPOCHS,

        "LR":
            EXPECTED_LR,

        "HIDDEN_DIM":
            EXPECTED_HIDDEN,

        "LATENT_DIM":
            EXPECTED_LATENT,
    }


    print(
        "PRODUCTION_CONSTANTS =",
        constants
    )


    if constants != expected:

        raise RuntimeError(
            "Frozen VGAE constants differ "
            "from expected production values."
        )


    # --------------------------------------------------------------
    # Extract model classes from the VGAE section.
    #
    # All class definitions appearing before train_vgae_dense()
    # are retained. This preserves the exact DenseGCN / VGAE
    # architecture from the production script.
    # --------------------------------------------------------------

    train_node = None

    for node in tree.body:

        if (
            isinstance(
                node,
                ast.FunctionDef
            )
            and node.name
            == "train_vgae_dense"
        ):

            train_node = node

            break


    if train_node is None:

        raise RuntimeError(
            "train_vgae_dense() not found."
        )


    class_nodes = [

        node

        for node in tree.body

        if (
            isinstance(
                node,
                ast.ClassDef
            )

            and node.lineno
            < train_node.lineno
        )
    ]


    required_function_names = {

        "set_seed",
        "normalize_adjacency",
        "train_vgae_dense",
    }


    function_nodes = [

        node

        for node in tree.body

        if (
            isinstance(
                node,
                ast.FunctionDef
            )

            and node.name
            in required_function_names
        )
    ]


    found_functions = {
        node.name
        for node in function_nodes
    }


    if (
        found_functions
        != required_function_names
    ):

        raise RuntimeError(
            "Required VGAE functions missing. "
            f"Found={found_functions}"
        )


    selected_nodes = (
        class_nodes
        + function_nodes
    )


    selected_nodes = sorted(
        selected_nodes,
        key=lambda n: n.lineno
    )


    module = ast.Module(
        body=selected_nodes,
        type_ignores=[]
    )

    ast.fix_missing_locations(
        module
    )


    # --------------------------------------------------------------
    # Import every literal uppercase constant from the frozen
    # production script (e.g. DROPOUT) without executing that script.
    # The five primary VGAE constants are still checked separately.
    # --------------------------------------------------------------

    production_literals = {}

    for node in tree.body:

        if isinstance(node, ast.Assign):

            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                ):
                    production_literals[target.id] = value

        elif isinstance(node, ast.AnnAssign):

            if (
                isinstance(node.target, ast.Name)
                and node.target.id.isupper()
            ):

                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue

                production_literals[node.target.id] = value


    if "DROPOUT" not in production_literals:

        raise RuntimeError(
            "DROPOUT could not be recovered from "
            "the frozen HBN production script."
        )


    print(
        "PRODUCTION_DROPOUT =",
        production_literals["DROPOUT"]
    )


    namespace = {

        **production_literals,

        "np": np,
        "pd": pd,

        "torch": torch,
        "nn": nn,
        "F": F,

        "random": random,
        "os": os,

        "Path": Path,

        "DEVICE": DEVICE,

        "SEED":
            EXPECTED_SEED,

        "VGAE_EPOCHS":
            EXPECTED_EPOCHS,

        "LR":
            EXPECTED_LR,

        "HIDDEN_DIM":
            EXPECTED_HIDDEN,

        "LATENT_DIM":
            EXPECTED_LATENT,
    }


    exec(
        compile(
            module,
            filename=str(
                script_path
            ),
            mode="exec"
        ),
        namespace
    )


    print(
        "EXTRACTED_CLASSES =",
        [
            n.name
            for n in class_nodes
        ]
    )

    print(
        "EXTRACTED_FUNCTIONS =",
        sorted(
            found_functions
        )
    )


    return (
        namespace[
            "set_seed"
        ],

        namespace[
            "normalize_adjacency"
        ],

        namespace[
            "train_vgae_dense"
        ],

        constants,
    )


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 6 — FIXED-CONFIG VGAE")
print("EEG-ONLY EXTERNAL VALIDATION")
print("NO HYPERPARAMETER TUNING")
print("=" * 78)

print()
print(
    "DEVICE =",
    DEVICE
)

print(
    "TORCH_VERSION =",
    torch.__version__
)


# ======================================================================
# 1. VERIFY STAGE-5 LOCKED GRAPH
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY LOCKED STAGE-5 GRAPH")
print("=" * 78)


for p in [

    NODE_FILE,
    ADJ_FILE,
    META_FILE,
    STAGE5_SUMMARY,

]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


stage5 = json.loads(
    STAGE5_SUMMARY.read_text(
        encoding="utf-8"
    )
)


if stage5.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Stage-5 graph is not PASS."
    )


if stage5.get(
    "graph_k"
) != 12:

    raise RuntimeError(
        "Stage-5 graph K is not 12."
    )


if stage5.get(
    "phenotype_view_used"
) is not False:

    raise RuntimeError(
        "Stage-5 graph is not EEG-only."
    )


X = np.load(
    NODE_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


A = np.load(
    ADJ_FILE,
    allow_pickle=False
).astype(
    np.float32,
    copy=False
)


meta = pd.read_csv(
    META_FILE
)


if X.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_INPUT_DIM
):

    raise RuntimeError(
        f"Unexpected node feature shape "
        f"{X.shape}"
    )


if A.shape != (
    EXPECTED_SUBJECTS,
    EXPECTED_SUBJECTS
):

    raise RuntimeError(
        f"Unexpected adjacency shape "
        f"{A.shape}"
    )


if len(meta) != EXPECTED_SUBJECTS:

    raise RuntimeError(
        "Metadata row count mismatch."
    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(
        "Node features contain "
        "non-finite values."
    )


if not np.isfinite(
    A
).all():

    raise RuntimeError(
        "Adjacency contains "
        "non-finite values."
    )


if (
    sha256_file(
        NODE_FILE
    )
    != stage5[
        "node_features_sha256"
    ]
):

    raise RuntimeError(
        "Node-feature SHA256 mismatch."
    )


if (
    sha256_file(
        ADJ_FILE
    )
    != stage5[
        "adjacency_sha256"
    ]
):

    raise RuntimeError(
        "Adjacency SHA256 mismatch."
    )


print(
    "NODE_FEATURES_SHAPE =",
    X.shape
)

print(
    "ADJACENCY_SHAPE =",
    A.shape
)

print(
    "STAGE5_HASH_VERIFICATION=PASS"
)


# ======================================================================
# 2. LOAD EXACT FINAL HBN VGAE IMPLEMENTATION
# ======================================================================

print()
print("=" * 78)
print("2. LOAD EXACT HBN FINAL VGAE IMPLEMENTATION")
print("=" * 78)


if not FINAL_HBN_SCRIPT.exists():

    raise FileNotFoundError(
        FINAL_HBN_SCRIPT
    )


production_sha = sha256_file(
    FINAL_HBN_SCRIPT
)


print(
    "PRODUCTION_SCRIPT =",
    FINAL_HBN_SCRIPT
)

print(
    "PRODUCTION_SCRIPT_SHA256 =",
    production_sha
)


(
    set_seed,
    normalize_adjacency,
    train_vgae_dense,
    production_constants,

) = load_exact_vgae_components(
    FINAL_HBN_SCRIPT
)


print(
    "SEED =",
    production_constants[
        "SEED"
    ]
)

print(
    "EPOCHS =",
    production_constants[
        "VGAE_EPOCHS"
    ]
)

print(
    "LR =",
    production_constants[
        "LR"
    ]
)

print(
    "HIDDEN_DIM =",
    production_constants[
        "HIDDEN_DIM"
    ]
)

print(
    "LATENT_DIM =",
    production_constants[
        "LATENT_DIM"
    ]
)


# ======================================================================
# 3. VERIFY EXACT ADJACENCY NORMALIZATION BEFORE TRAINING
# ======================================================================

print()
print("=" * 78)
print("3. VERIFY PRODUCTION ADJACENCY NORMALIZATION")
print("=" * 78)


A_norm = normalize_adjacency(
    A.copy()
)


A_norm = np.asarray(
    A_norm,
    dtype=np.float32
)


if A_norm.shape != A.shape:

    raise RuntimeError(
        "Normalized adjacency "
        "shape mismatch."
    )


if not np.isfinite(
    A_norm
).all():

    raise RuntimeError(
        "Normalized adjacency "
        "contains non-finite values."
    )


print(
    "NORMALIZED_ADJACENCY_SHAPE =",
    A_norm.shape
)

print(
    "NORMALIZED_ADJACENCY_MIN =",
    float(
        A_norm.min()
    )
)

print(
    "NORMALIZED_ADJACENCY_MAX =",
    float(
        A_norm.max()
    )
)

print(
    "NORMALIZED_ADJACENCY_FINITE =",
    True
)


# ======================================================================
# 4. TRAIN PRIMARY EXTERNAL VGAE
# ======================================================================

print()
print("=" * 78)
print("4. TRAIN PRIMARY VGAE — SEED 42")
print("=" * 78)

print(
    "IMPORTANT:"
)

print(
    "This is unsupervised fitting on "
    "the independent MIPDB graph."
)

print(
    "Architecture and all hyperparameters "
    "were fixed from HBN."
)

print(
    "No MIPDB labels or clustering outcomes "
    "are used for model selection."
)


set_seed(
    EXPECTED_SEED
)


result = train_vgae_dense(
    X,
    A,
    OUT
)


# ======================================================================
# 5. IDENTIFY / VERIFY 16-D LATENT OUTPUT
# ======================================================================

print()
print("=" * 78)
print("5. VERIFY VGAE LATENT REPRESENTATION")
print("=" * 78)


latent = None
production_latent_file = None


if result is not None:

    candidate = np.asarray(
        result
    )

    if candidate.shape == (
        EXPECTED_SUBJECTS,
        EXPECTED_LATENT
    ):

        latent = candidate.astype(
            np.float32,
            copy=False
        )


if latent is None:

    for p in sorted(
        OUT.glob("*.npy")
    ):

        try:

            arr = np.load(
                p,
                allow_pickle=False
            )

        except Exception:

            continue


        if arr.shape == (
            EXPECTED_SUBJECTS,
            EXPECTED_LATENT
        ):

            latent = arr.astype(
                np.float32,
                copy=False
            )

            production_latent_file = p

            break


if latent is None:

    raise RuntimeError(
        "Could not identify the "
        "91x16 production VGAE latent output."
    )


if not np.isfinite(
    latent
).all():

    raise RuntimeError(
        "VGAE latent representation "
        "contains non-finite values."
    )


canonical_latent = (
    OUT
    / "MIPDB_EEG_ONLY_VGAE_LATENT16_SEED42.npy"
)


np.save(
    canonical_latent,
    latent.astype(
        np.float32
    ),
    allow_pickle=False
)


latent_csv = (
    OUT
    / "MIPDB_EEG_ONLY_VGAE_LATENT16_SEED42.csv"
)


latent_df = pd.DataFrame(
    latent,
    columns=[
        f"vgae_{i+1:02d}"
        for i in range(
            EXPECTED_LATENT
        )
    ]
)


latent_df.insert(
    0,
    "subject",
    meta[
        "subject"
    ].values
)


latent_df.to_csv(
    latent_csv,
    index=False
)


active_dims = int(
    np.sum(
        np.std(
            latent,
            axis=0
        )
        > 1e-8
    )
)


print(
    "LATENT_SHAPE =",
    latent.shape
)

print(
    "LATENT_FINITE =",
    True
)

print(
    "ACTIVE_LATENT_DIMENSIONS =",
    active_dims,
    "/",
    EXPECTED_LATENT
)

print(
    "LATENT_MEAN =",
    float(
        latent.mean()
    )
)

print(
    "LATENT_STD =",
    float(
        latent.std()
    )
)

print(
    "LATENT_MIN =",
    float(
        latent.min()
    )
)

print(
    "LATENT_MAX =",
    float(
        latent.max()
    )
)


# ======================================================================
# 6. VERIFY RECONSTRUCTED GRAPH IF PRODUCTION FUNCTION SAVED IT
# ======================================================================

print()
print("=" * 78)
print("6. RECONSTRUCTED-GRAPH CHECK")
print("=" * 78)


recon_file = (
    OUT
    / "reconstructed_graph.npy"
)


recon_summary = None


if recon_file.exists():

    R = np.load(
        recon_file,
        allow_pickle=False
    )


    if R.shape != (
        EXPECTED_SUBJECTS,
        EXPECTED_SUBJECTS
    ):

        raise RuntimeError(
            "Unexpected reconstructed "
            f"graph shape {R.shape}"
        )


    if not np.isfinite(
        R
    ).all():

        raise RuntimeError(
            "Reconstructed graph contains "
            "non-finite values."
        )


    recon_summary = {

        "shape":
            list(R.shape),

        "minimum":
            float(R.min()),

        "maximum":
            float(R.max()),

        "mean":
            float(R.mean()),

        "sha256":
            sha256_file(
                recon_file
            ),
    }


    print(
        "RECONSTRUCTED_GRAPH_SHAPE =",
        R.shape
    )

    print(
        "RECONSTRUCTED_GRAPH_MIN =",
        float(
            R.min()
        )
    )

    print(
        "RECONSTRUCTED_GRAPH_MAX =",
        float(
            R.max()
        )
    )

else:

    print(
        "RECONSTRUCTED_GRAPH_FILE_NOT_FOUND"
    )


# ======================================================================
# 7. TRAINING LOG CHECK
# ======================================================================

print()
print("=" * 78)
print("7. TRAINING LOG CHECK")
print("=" * 78)


csv_candidates = sorted(
    OUT.glob("*.csv")
)


for p in csv_candidates:

    print(
        "CSV_OUTPUT =",
        p.name
    )


training_log = None


for p in csv_candidates:

    try:

        df = pd.read_csv(
            p
        )

    except Exception:

        continue


    if (
        "epoch"
        in df.columns

        and "total_loss"
        in df.columns
    ):

        training_log = df

        training_log_file = p

        break


if training_log is not None:

    print(
        "TRAINING_LOG_FILE =",
        training_log_file
    )

    print(
        "TRAINING_LOG_ROWS =",
        len(training_log)
    )

    print(
        "FIRST_TOTAL_LOSS =",
        float(
            training_log[
                "total_loss"
            ].iloc[0]
        )
    )

    print(
        "FINAL_TOTAL_LOSS =",
        float(
            training_log[
                "total_loss"
            ].iloc[-1]
        )
    )

else:

    print(
        "TRAINING_LOG_WITH_EPOCH_COLUMN_NOT_FOUND"
    )


# ======================================================================
# 8. FINAL AUDIT SUMMARY
# ======================================================================

summary = {

    "validation_branch":
        "MIPDB_EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION",

    "subjects":
        EXPECTED_SUBJECTS,

    "input_dimension":
        EXPECTED_INPUT_DIM,

    "graph_k":
        12,

    "seed":
        EXPECTED_SEED,

    "epochs":
        EXPECTED_EPOCHS,

    "learning_rate":
        EXPECTED_LR,

    "hidden_dimension":
        EXPECTED_HIDDEN,

    "latent_dimension":
        EXPECTED_LATENT,

    "optimizer":
        "Adam",

    "production_script":
        str(
            FINAL_HBN_SCRIPT
        ),

    "production_script_sha256":
        production_sha,

    "production_function":
        "train_vgae_dense",

    "exact_production_model_definitions_used":
        True,

    "exact_production_adjacency_normalization_used":
        True,

    "exact_production_loss_used":
        True,

    "latent_shape":
        list(
            latent.shape
        ),

    "active_latent_dimensions":
        active_dims,

    "latent_sha256":
        sha256_file(
            canonical_latent
        ),

    "latent_csv_sha256":
        sha256_file(
            latent_csv
        ),

    "reconstructed_graph":
        recon_summary,

    "phenotype_used":
        False,

    "mipdb_labels_used_for_training":
        False,

    "mipdb_hyperparameter_tuning":
        False,

    "vgae_fitted_on_external_graph":
        True,

    "status":
        "PASS",
}


summary_file = (
    AUDIT
    / "MIPDB_STAGE6_VGAE_FIXED_CONFIG_SUMMARY.json"
)


summary_file.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ======================================================================
# 9. PERMANENT LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE6_VGAE_FIXED_CONFIG_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 6 VGAE LOCK
======================================================================

VALIDATION_BRANCH=EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION

SUBJECTS=91
INPUT_FEATURE_DIMENSION=192
GRAPH_K=12

VGAE_SOURCE=HBN_FINAL_PRODUCTION_IMPLEMENTATION
VGAE_SOURCE_FILE=
{FINAL_HBN_SCRIPT}

VGAE_SOURCE_SHA256=
{production_sha}

SEED=42
EPOCHS=500
LEARNING_RATE=0.001
HIDDEN_DIMENSION=64
LATENT_DIMENSION=16
OPTIMIZER=Adam

LOSS=EXACT_HBN_PRODUCTION_LOSS
ADJACENCY_NORMALIZATION=EXACT_HBN_PRODUCTION_FUNCTION

VGAE_FITTED_ON_MIPDB_GRAPH=YES

MIPDB_LABELS_USED_FOR_TRAINING=NO
MIPDB_PHENOTYPE_USED=NO
MIPDB_HYPERPARAMETER_TUNING=NO

LATENT_FILE=
{canonical_latent}

LATENT_SHA256=
{sha256_file(canonical_latent)}

LATENT_SHAPE=91x16
ACTIVE_LATENT_DIMENSIONS={active_dims}

STAGE6_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# 10. SHA256 MANIFEST
# ======================================================================

hash_files = [

    canonical_latent,
    latent_csv,
    summary_file,
    lock_file,

]


if recon_file.exists():

    hash_files.append(
        recon_file
    )


sha_manifest = (
    AUDIT
    / "MIPDB_STAGE6_VGAE_SHA256.txt"
)


with open(
    sha_manifest,
    "w"
) as f:

    for p in hash_files:

        f.write(
            f"{sha256_file(p)}  {p}\n"
        )


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 78)
print("STAGE 6 COMPLETE")
print("=" * 78)

print(
    "SUBJECTS=91"
)

print(
    "INPUT_DIMENSION=192"
)

print(
    "GRAPH_K=12"
)

print(
    "VGAE_SEED=42"
)

print(
    "VGAE_EPOCHS=500"
)

print(
    "VGAE_LR=0.001"
)

print(
    "VGAE_HIDDEN_DIM=64"
)

print(
    "VGAE_LATENT_DIM=16"
)

print(
    "LATENT_SHAPE=",
    latent.shape
)

print(
    "ACTIVE_LATENT_DIMENSIONS=",
    active_dims
)

print(
    "MIPDB_LABELS_USED=NO"
)

print(
    "MIPDB_HYPERPARAMETER_TUNING=NO"
)

print(
    "STAGE6_STATUS=PASS"
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

