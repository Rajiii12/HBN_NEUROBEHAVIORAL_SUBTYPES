# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 3B
# AUTHORITATIVE PRIMARY PARTITION FREEZE
#
# FROZEN FROM STAGE3A:
#   selected K = 6
#   primary UMAP seed = 42
#
# IMPORTANT
# ---------
# Stage3A selected K before any clinical opening.
#
# This stage:
#   - verifies Stage3A outputs
#   - independently reconstructs the seed42 pipeline
#   - verifies the saved K=6 partition
#   - freezes the authoritative numeric cluster labels
#   - audits the relationship between UMAP connected
#     components and the K=6 partition
#
# NO:
#   diagnosis analysis
#   ADHD/YBOCS/BDI analysis
#   phenotype interpretation
#   cluster naming
#   K change
# ============================================================

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from umap import UMAP

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from scipy.sparse import (
    load_npz
)

from scipy.sparse.csgraph import (
    connected_components
)


# ============================================================
# 1. ROOTS
# ============================================================

DRIVE = Path(
    "/content/drive/MyDrive"
)

EXP3 = (
    DRIVE
    / "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION"
)

AUDIT = (
    EXP3
    / "00_METADATA_AUDIT"
)

LOCK = (
    EXP3
    / "01_LOCK"
)

VGAE_DIR = (
    EXP3
    / "04_VGAE"
)

STAGE3_DIR = (
    EXP3
    / "05_UMAP_K_SELECTION"
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 3B — AUTHORITATIVE PRIMARY PARTITION FREEZE")
print("=" * 120)


# ============================================================
# 2. HELPERS
# ============================================================

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

            h.update(
                chunk
            )

    return h.hexdigest()


def build_umap_affinity(Y):

    A = kneighbors_graph(

        Y,

        n_neighbors=4,

        mode="connectivity",

        metric="euclidean",

        include_self=False,

        n_jobs=1
    ).tocsr()


    A = A.maximum(
        A.T
    ).tocsr()


    A.setdiag(
        0
    )

    A.eliminate_zeros()


    if A.nnz > 0:

        A.data[:] = 1.0


    return A


# ============================================================
# 3. FROZEN INPUT FILES
# ============================================================

STAGE3A_DESIGN = (
    LOCK
    / "STAGE3A_K_SELECTION_DESIGN_LOCK.json"
)

STAGE3A_RESULT = (
    LOCK
    / "STAGE3A_K_SELECTION_RESULT_LOCK.json"
)

PER_SEED_RESULTS = (
    AUDIT
    / "STAGE3A_PER_SEED_K_RESULTS.csv"
)

AFFINITY_AUDIT = (
    AUDIT
    / "STAGE3A_UMAP_4NN_AFFINITY_AUDIT.csv"
)

PAIRWISE_STABILITY = (
    AUDIT
    / "STAGE3A_PAIRWISE_SEED_STABILITY.csv"
)

K_SUMMARY = (
    AUDIT
    / "STAGE3A_K_SELECTION_SUMMARY.csv"
)

STAGE3A_REPORT = (
    AUDIT
    / "STAGE3A_K_SELECTION_REPORT.txt"
)


MU_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.npy"
)

SUBJECT_ORDER = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)


SAVED_UMAP = (
    STAGE3_DIR
    / "STAGE3A_UMAP_SEED42.npy"
)

SAVED_AFFINITY = (
    STAGE3_DIR
    / "STAGE3A_UMAP_4NN_AFFINITY_SEED42.npz"
)

SAVED_LABELS = (
    STAGE3_DIR
    / "STAGE3A_LABELS_SEED42_K6.csv"
)


# ============================================================
# 4. VERIFY FROZEN STAGE3A HASHES
# ============================================================

EXPECTED_HASHES = {

    STAGE3A_DESIGN:
        "132232810b5c9a9104b1a4e390713f37ab94168105442358d5872eef70f1fc87",

    PER_SEED_RESULTS:
        "28a3e3c19137dd1d66c3aac7e5ac69a8d94e862a1107c4c08444188495f7468e",

    AFFINITY_AUDIT:
        "d34f98ca5906ef24704fe929f0baf804f2f17db73a3163f315b2412cd8dc31b0",

    PAIRWISE_STABILITY:
        "9d8cb009dec51a65aa74466ed14ece5a6bd2eccf155e2f9ae3934dac99f79a4f",

    K_SUMMARY:
        "1c99a44001dba0015e9851a86fff390381a423ed9f39cd1990a4818774033746",

    STAGE3A_RESULT:
        "c48881e873b052c6d4bb328dc31c4105028f1d37d252d5e1588ed423f74ffd18",

    STAGE3A_REPORT:
        "3bed048a97e88aa17dc8ee836f2fe0e21d039034dca1bd2f039ac9e85f83d9a1",

    MU_FILE:
        "89ef44d4aa3f718e59dc391f43df51baca891cda1178c05c23346f0c46963415",

    SUBJECT_ORDER:
        "8310ccc52f0d0698ed171ded6feb6f14cf4738787b21b93dbedcc0a1d1c99c6e"
}


print(
    "\n" + "=" * 120
)

print(
    "FROZEN INPUT HASH VERIFICATION"
)

print(
    "=" * 120
)


for path, expected in EXPECTED_HASHES.items():

    if not path.exists():

        raise FileNotFoundError(
            str(
                path
            )
        )


    actual = sha256_file(
        path
    )


    print(
        "PASS"
        if actual == expected
        else
        "FAIL",

        path.name
    )


    if actual != expected:

        print(
            "Expected:",
            expected
        )

        print(
            "Actual:",
            actual
        )


        raise RuntimeError(
            "Frozen Stage3A input changed. STOP."
        )


for path in [
    SAVED_UMAP,
    SAVED_AFFINITY,
    SAVED_LABELS
]:

    if not path.exists():

        raise FileNotFoundError(
            str(
                path
            )
        )


print(
    "\nALL FROZEN STAGE3A INPUT HASHES PASS."
)


# ============================================================
# 5. VERIFY SELECTED K
# ============================================================

with open(
    STAGE3A_RESULT,
    "r",
    encoding="utf-8"
) as f:

    result_lock = json.load(
        f
    )


assert result_lock[
    "status"
] == "K_SELECTED_BEFORE_AUTHORITATIVE_PARTITION_FREEZE"


assert result_lock[
    "N"
] == 85


assert result_lock[
    "selected_K"
] == 6


assert result_lock[
    "primary_seed_for_next_partition_freeze"
] == 42


assert result_lock[
    "clinical_variables_used"
] is False


assert result_lock[
    "diagnosis_used"
] is False


assert result_lock[
    "Experiment2_labels_used"
] is False


SELECTED_K = 6

PRIMARY_SEED = 42


print(
    "\nFrozen K-selection result:"
)

print(
    "Selected K:",
    SELECTED_K
)

print(
    "Primary UMAP seed:",
    PRIMARY_SEED
)


# ============================================================
# 6. LOAD FROZEN VGAE + SUBJECT ORDER
# ============================================================

mu = np.load(
    MU_FILE
).astype(
    np.float64
)


assert mu.shape == (
    85,
    16
)


assert np.isfinite(
    mu
).all()


subjects = pd.read_csv(
    SUBJECT_ORDER
)


assert len(
    subjects
) == 85


assert subjects[
    "subject"
].nunique() == 85


assert np.array_equal(

    subjects[
        "experiment3_row"
    ].to_numpy(
        dtype=int
    ),

    np.arange(
        85
    )
)


# ============================================================
# 7. LOAD SAVED STAGE3A PRIMARY-SEED OUTPUTS
# ============================================================

Y_saved = np.load(
    SAVED_UMAP
).astype(
    np.float64
)


A_saved = load_npz(
    SAVED_AFFINITY
).tocsr()


labels_saved_table = pd.read_csv(
    SAVED_LABELS
)


assert Y_saved.shape == (
    85,
    2
)


assert A_saved.shape == (
    85,
    85
)


assert len(
    labels_saved_table
) == 85


assert "candidate_cluster" in (
    labels_saved_table.columns
)


# ------------------------------------------------------------
# Verify subject order in candidate-label file.
# ------------------------------------------------------------

assert np.array_equal(

    labels_saved_table[
        "experiment3_row"
    ].to_numpy(
        dtype=int
    ),

    subjects[
        "experiment3_row"
    ].to_numpy(
        dtype=int
    )
)


assert (

    labels_saved_table[
        "subject"
    ].astype(str).tolist()

    ==

    subjects[
        "subject"
    ].astype(str).tolist()
)


labels_saved = labels_saved_table[
    "candidate_cluster"
].to_numpy(
    dtype=int
)


assert len(
    np.unique(
        labels_saved
    )
) == 6


# ============================================================
# 8. INDEPENDENT SEED42 RECONSTRUCTION
#
# This confirms that the saved Stage3A primary partition
# follows exactly from the frozen VGAE embedding/settings.
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "PRIMARY SEED42 RECONSTRUCTION CHECK"
)

print(
    "=" * 120
)


umap_model = UMAP(

    n_neighbors=30,

    n_components=2,

    min_dist=0.1,

    metric="euclidean",

    init="spectral",

    random_state=42,

    transform_seed=42,

    n_jobs=1
)


Y_recomputed = umap_model.fit_transform(
    mu
).astype(
    np.float64
)


assert Y_recomputed.shape == (
    85,
    2
)


umap_max_abs_difference = float(

    np.max(

        np.abs(
            Y_saved
            -
            Y_recomputed
        )
    )
)


print(
    "Saved/recomputed UMAP max abs difference:",
    f"{umap_max_abs_difference:.12g}"
)


if not np.allclose(

    Y_saved,
    Y_recomputed,

    rtol=1e-5,
    atol=1e-5

):

    raise RuntimeError(
        "Seed42 UMAP could not be reproduced. STOP."
    )


A_recomputed = build_umap_affinity(
    Y_recomputed
)


affinity_difference_nnz = int(

    (
        A_saved
        !=
        A_recomputed
    ).nnz
)


print(
    "Saved/recomputed affinity differing entries:",
    affinity_difference_nnz
)


if affinity_difference_nnz != 0:

    raise RuntimeError(
        "Seed42 4NN affinity could not be reproduced. STOP."
    )


spectral_model = SpectralClustering(

    n_clusters=6,

    affinity="precomputed",

    assign_labels="kmeans",

    random_state=42,

    n_init=20
)


with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore"
    )


    labels_recomputed = spectral_model.fit_predict(
        A_recomputed
    ).astype(
        int
    )


reconstruction_ARI = float(

    adjusted_rand_score(
        labels_saved,
        labels_recomputed
    )
)


reconstruction_NMI = float(

    normalized_mutual_info_score(
        labels_saved,
        labels_recomputed
    )
)


print(
    "Saved/recomputed partition ARI:",
    f"{reconstruction_ARI:.10f}"
)

print(
    "Saved/recomputed partition NMI:",
    f"{reconstruction_NMI:.10f}"
)


if not np.isclose(
    reconstruction_ARI,
    1.0
):

    raise RuntimeError(
        "Seed42 K=6 partition could not be reproduced. STOP."
    )


if not np.isclose(
    reconstruction_NMI,
    1.0
):

    raise RuntimeError(
        "Seed42 K=6 partition could not be reproduced. STOP."
    )


print(
    "PRIMARY SEED42 RECONSTRUCTION: PASS"
)


# ============================================================
# 9. PRIMARY PARTITION METRICS
# ============================================================

sizes = np.bincount(

    labels_saved,

    minlength=6
)


assert sizes.tolist() == [
    27,
    15,
    18,
    10,
    9,
    6
]


min_size = int(
    sizes.min()
)

max_size = int(
    sizes.max()
)

balance_ratio = float(
    max_size
    /
    min_size
)


silhouette = float(

    silhouette_score(
        Y_saved,
        labels_saved,
        metric="euclidean"
    )
)


db = float(

    davies_bouldin_score(
        Y_saved,
        labels_saved
    )
)


ch = float(

    calinski_harabasz_score(
        Y_saved,
        labels_saved
    )
)


print(
    "\n" + "=" * 120
)

print(
    "PRIMARY K=6 PARTITION"
)

print(
    "=" * 120
)


print(
    "Cluster sizes:",
    sizes.tolist()
)

print(
    "Minimum cluster size:",
    min_size
)

print(
    "Maximum cluster size:",
    max_size
)

print(
    "Balance ratio:",
    f"{balance_ratio:.6f}"
)

print(
    "Silhouette:",
    f"{silhouette:.10f}"
)

print(
    "Davies-Bouldin:",
    f"{db:.10f}"
)

print(
    "Calinski-Harabasz:",
    f"{ch:.10f}"
)


# ============================================================
# 10. PRIMARY 4NN AFFINITY DIAGNOSTICS
# ============================================================

component_count, component_labels = (
    connected_components(

        A_saved,

        directed=False,

        return_labels=True
    )
)


degree = np.asarray(

    A_saved.sum(
        axis=1
    )

).ravel()


isolates = int(

    np.sum(
        degree == 0
    )
)


edges = int(
    A_saved.nnz
    //
    2
)


component_sizes = (

    pd.Series(
        component_labels
    )
    .value_counts()
    .sort_index()
)


print(
    "\nPrimary UMAP 4NN affinity:"
)

print(
    "Connected components:",
    component_count
)

print(
    "Component sizes:",
    component_sizes.to_dict()
)

print(
    "Isolated nodes:",
    isolates
)

print(
    "Undirected edges:",
    edges
)


assert component_count == 2

assert isolates == 0

assert edges == 206


# ============================================================
# 11. COMPONENT × CLUSTER AUDIT
#
# This is purely structural.
# No phenotype/diagnosis data are opened.
# ============================================================

component_cluster = pd.crosstab(

    pd.Series(
        component_labels,
        name="UMAP_4NN_component"
    ),

    pd.Series(
        labels_saved,
        name="EEG_PHENO_cluster"
    )
)


print(
    "\n" + "=" * 120
)

print(
    "UMAP COMPONENT × K=6 CLUSTER AUDIT"
)

print(
    "=" * 120
)


print(
    component_cluster.to_string()
)


COMPONENT_CROSSTAB_FILE = (
    AUDIT
    / "STAGE3B_UMAP_COMPONENT_BY_CLUSTER_COUNTS.csv"
)


component_cluster.to_csv(
    COMPONENT_CROSSTAB_FILE
)


COMPONENT_ASSIGNMENT_FILE = (
    AUDIT
    / "STAGE3B_UMAP_COMPONENT_ASSIGNMENTS.csv"
)


component_assignment = subjects[
    [
        "experiment3_row",
        "subject",
        "subject_norm"
    ]
].copy()


component_assignment[
    "UMAP_4NN_component"
] = component_labels


component_assignment[
    "EEG_PHENO_cluster"
] = labels_saved


component_assignment.to_csv(
    COMPONENT_ASSIGNMENT_FILE,
    index=False
)


# ============================================================
# 12. FREEZE AUTHORITATIVE PARTITION
#
# Numeric cluster IDs only.
# NO subtype names yet.
# ============================================================

PARTITION_FILE = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PRIMARY_PARTITION.csv"
)


partition = subjects[
    [
        "experiment3_row",
        "subject",
        "subject_norm"
    ]
].copy()


partition[
    "EEG_PHENO_cluster"
] = labels_saved


partition[
    "K"
] = 6


partition[
    "UMAP_seed"
] = 42


partition.to_csv(
    PARTITION_FILE,
    index=False
)


# ============================================================
# 13. SUMMARY
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE3B_PRIMARY_PARTITION_SUMMARY.json"
)


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3B_AUTHORITATIVE_PRIMARY_PARTITION",

    "status":
        "PRIMARY_PARTITION_FROZEN",

    "N":
        85,

    "K":
        6,

    "UMAP_seed":
        42,

    "cluster_sizes":
        {

            str(i):
                int(
                    sizes[
                        i
                    ]
                )

            for i in range(
                6
            )
        },

    "minimum_cluster_size":
        min_size,

    "maximum_cluster_size":
        max_size,

    "balance_ratio":
        balance_ratio,

    "internal_geometry":
        {

            "silhouette":
                silhouette,

            "Davies_Bouldin":
                db,

            "Calinski_Harabasz":
                ch
        },

    "UMAP_4NN_affinity":
        {

            "components":
                int(
                    component_count
                ),

            "component_sizes":
                {

                    str(k):
                        int(v)

                    for k, v in

                    component_sizes.to_dict().items()
                },

            "isolates":
                isolates,

            "undirected_edges":
                edges
        },

    "structural_caveat":
        (
            "The primary seed42 exact-4NN UMAP affinity contains "
            "two connected components but no isolated subjects. "
            "Disconnected affinity was not a disqualifying criterion "
            "in the prospectively frozen Stage3A design and therefore "
            "does not alter the selected K or primary seed."
        ),

    "K_selection_interpretation":
        (
            "K=6 was the only candidate resolution satisfying "
            "the prospectively frozen minimum-cluster-size and "
            "balance guards across all five Stage3A UMAP seeds. "
            "It is retained as the selected resolution and is not "
            "interpreted as a uniquely true biological subtype count."
        ),

    "seed42_reconstruction":
        {

            "UMAP_max_abs_difference":
                umap_max_abs_difference,

            "affinity_differing_entries":
                affinity_difference_nnz,

            "partition_ARI":
                reconstruction_ARI,

            "partition_NMI":
                reconstruction_NMI
        },

    "diagnosis_opened":
        False,

    "clinical_scores_opened":
        False,

    "cluster_names_assigned":
        False,

    "K_change_allowed_after_this_stage":
        False
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 14. AUTHORITATIVE PARTITION LOCK
# ============================================================

PARTITION_LOCK_FILE = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PARTITION_LOCK.json"
)


partition_lock = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3B_AUTHORITATIVE_PARTITION_LOCK",

    "status":
        "FROZEN",

    "N":
        85,

    "K":
        6,

    "primary_UMAP_seed":
        42,

    "numeric_cluster_labels_authoritative":
        True,

    "cluster_names_assigned":
        False,

    "primary_partition_may_be_replaced_by_later_clinical_results":
        False,

    "K_may_be_changed_by_later_clinical_results":
        False,

    "diagnosis_used":
        False,

    "ADHD_YBOCS_BDI_used":
        False,

    "Experiment2_labels_used":
        False,

    "primary_affinity_components":
        int(
            component_count
        ),

    "primary_affinity_isolates":
        isolates,

    "primary_affinity_edges":
        edges,

    "cluster_sizes":
        sizes.tolist(),

    "input_hashes":
        {

            "Stage3A_design":
                sha256_file(
                    STAGE3A_DESIGN
                ),

            "Stage3A_result":
                sha256_file(
                    STAGE3A_RESULT
                ),

            "posterior_mu":
                sha256_file(
                    MU_FILE
                ),

            "Stage3A_seed42_UMAP":
                sha256_file(
                    SAVED_UMAP
                ),

            "Stage3A_seed42_affinity":
                sha256_file(
                    SAVED_AFFINITY
                ),

            "Stage3A_seed42_K6_labels":
                sha256_file(
                    SAVED_LABELS
                )
        },

    "output_hashes":
        {

            "authoritative_partition":
                sha256_file(
                    PARTITION_FILE
                ),

            "component_cluster_crosstab":
                sha256_file(
                    COMPONENT_CROSSTAB_FILE
                ),

            "component_assignments":
                sha256_file(
                    COMPONENT_ASSIGNMENT_FILE
                ),

            "summary":
                sha256_file(
                    SUMMARY_FILE
                )
        }
}


if PARTITION_LOCK_FILE.exists():

    with open(
        PARTITION_LOCK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        existing_lock = json.load(
            f
        )


    if existing_lock != partition_lock:

        raise RuntimeError(
            "Existing Stage3B partition lock differs. STOP."
        )


    print(
        "\nExisting Stage3B authoritative partition lock verified."
    )


else:

    with open(
        PARTITION_LOCK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            partition_lock,
            f,
            indent=2
        )


    print(
        "\nStage3B authoritative partition lock created."
    )


# ============================================================
# 15. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE3B_AUTHORITATIVE_PARTITION_REPORT.txt"
)


report = f"""
TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
STAGE 3B — AUTHORITATIVE PRIMARY PARTITION

PRIMARY COHORT
--------------
N:
85

SELECTED RESOLUTION
-------------------
K:
6

Primary UMAP seed:
42

Cluster sizes:
{sizes.tolist()}

Minimum cluster size:
{min_size}

Maximum cluster size:
{max_size}

Balance ratio:
{balance_ratio:.10f}


INTERNAL GEOMETRY
-----------------
Silhouette:
{silhouette:.10f}

Davies-Bouldin:
{db:.10f}

Calinski-Harabasz:
{ch:.10f}


PRIMARY UMAP 4NN AFFINITY
-------------------------
Connected components:
{component_count}

Component sizes:
{component_sizes.to_dict()}

Isolated subjects:
{isolates}

Undirected edges:
{edges}


IMPORTANT STRUCTURAL CAVEAT
---------------------------
The primary seed42 exact-4NN UMAP affinity contains
two connected components but zero isolated subjects.

Disconnected affinity was not a disqualifying condition
in the prospectively frozen Stage3A selection design.

The K=6 solution therefore remains authoritative.

The disconnected support must be reported transparently
and tracked in subsequent robustness analyses.


K-SELECTION INTERPRETATION
--------------------------
K=6 was the only candidate K that satisfied the frozen
minimum-cluster-size and balance guards across every one
of the five Stage3A UMAP seeds.

It was not selected because it maximized every individual
internal metric.

K=6 should therefore be interpreted as the retained
data-supported resolution under the prespecified selection
procedure, not as a uniquely established biological truth.


REPRODUCTION CHECK
------------------
UMAP max absolute difference:
{umap_max_abs_difference:.12g}

Affinity differing entries:
{affinity_difference_nnz}

Partition ARI:
{reconstruction_ARI:.10f}

Partition NMI:
{reconstruction_NMI:.10f}


DESIGN PROTECTION
-----------------
Diagnosis used:
NO

ADHD/YBOCS/BDI used:
NO

Experiment2 cluster labels used:
NO

Clinical names assigned:
NO

K may be changed after clinical characterization:
NO


FINAL STATUS
------------
STAGE 3B PASS —
AUTHORITATIVE EEG+PHENOTYPE K=6 PARTITION FROZEN.
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report
    )


# ============================================================
# 16. FINAL HASHES
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 3B OUTPUT HASHES"
)

print(
    "=" * 120
)


print(
    "Authoritative partition:",
    sha256_file(
        PARTITION_FILE
    )
)

print(
    "Component × cluster counts:",
    sha256_file(
        COMPONENT_CROSSTAB_FILE
    )
)

print(
    "Component assignments:",
    sha256_file(
        COMPONENT_ASSIGNMENT_FILE
    )
)

print(
    "Summary:",
    sha256_file(
        SUMMARY_FILE
    )
)

print(
    "Report:",
    sha256_file(
        REPORT_FILE
    )
)

print(
    "Partition lock:",
    sha256_file(
        PARTITION_LOCK_FILE
    )
)


# ============================================================
# 17. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 3B PASS — "
    "AUTHORITATIVE K=6 PARTITION FROZEN"
)

print(
    "=" * 120
)


print(
    "\nAUTHORITATIVE PRIMARY PARTITION:"
)

print(
    "N = 85"
)

print(
    "K = 6"
)

print(
    "Cluster sizes =",
    sizes.tolist()
)

print(
    "Silhouette =",
    f"{silhouette:.6f}"
)

print(
    "DB =",
    f"{db:.6f}"
)

print(
    "CH =",
    f"{ch:.6f}"
)


print(
    "\nPrimary UMAP affinity components =",
    component_count
)

print(
    "Primary UMAP affinity isolates =",
    isolates
)


print(
    "\nK IS NOW PERMANENTLY FROZEN FOR THE PRIMARY ANALYSIS."
)

print(
    "DO NOT CHANGE TO K=3, K=4, OR K=5 "
    "BASED ON LATER CLINICAL RESULTS."
)

print(
    "NO CLINICAL NAMES HAVE BEEN ASSIGNED."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 3C — 20-SEED K=6 ROBUSTNESS AUDIT, "
    "INCLUDING THE FREQUENCY OF DISCONNECTED "
    "UMAP 4NN SUPPORTS."
)