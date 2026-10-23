# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 1A — STANDARDIZATION + FUSED GRAPH CONSTRUCTION
#
# PRIMARY COHORT:
#   N = 85
#
# EEG VIEW:
#   Stage7G equal-window subject representation
#   128 dimensions
#
# PHENOTYPE VIEW:
#   baseline sessID=1 NEO-FFI item responses
#   60 dimensions
#
# FUSION:
#   alpha = 0.6
#
# IMPORTANT IMPLEMENTATION DETAIL:
#   We preserve the archived EPG-SD graph implementation:
#
#       1. standardize each view
#       2. calculate cosine-similarity matrices
#       3. fuse similarities
#       4. treat each similarity row as the kNN feature profile
#       5. sklearn kneighbors_graph(..., mode="connectivity")
#       6. symmetrize with maximum(A, A.T)
#
# Thus the final graph is BINARY.
#
# GRAPH K:
#   candidate k = 4, 6, 8, 10, 12
#   choose smallest k producing:
#       - one connected component
#       - zero isolated subjects
#
# THIS STAGE DOES NOT:
#   run VGAE
#   run UMAP
#   select K clusters
#   open diagnosis for model construction
#   use ADHD/YBOCS/BDI
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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

INPUTS = (
    EXP3
    / "02_INPUTS"
)

GRAPH_DIR = (
    EXP3
    / "03_GRAPH"
)


GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 1A — STANDARDIZATION + FUSED SUBJECT GRAPH")
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


def build_similarity_profile_knn(
    similarity_matrix,
    k
):

    # --------------------------------------------------------
    # IMPORTANT:
    # This preserves the archived EPG-SD implementation.
    #
    # kneighbors_graph treats every similarity ROW as a
    # feature vector and uses Euclidean distance between
    # similarity profiles.
    #
    # mode="connectivity" creates a binary adjacency.
    # --------------------------------------------------------

    adjacency = kneighbors_graph(

        similarity_matrix,

        n_neighbors=k,

        mode="connectivity",

        include_self=False
    )


    adjacency = adjacency.toarray().astype(
        np.float32
    )


    # Undirected graph.
    adjacency = np.maximum(
        adjacency,
        adjacency.T
    )


    np.fill_diagonal(
        adjacency,
        0.0
    )


    return adjacency


def graph_diagnostics(
    adjacency
):

    A = np.asarray(
        adjacency,
        dtype=np.float64
    )


    degree = A.sum(
        axis=1
    )


    isolates = int(
        np.sum(
            degree == 0
        )
    )


    n_components, component_labels = (
        connected_components(

            csr_matrix(
                A
            ),

            directed=False,

            return_labels=True
        )
    )


    component_sizes = (

        pd.Series(
            component_labels
        )

        .value_counts()

        .sort_values(
            ascending=False
        )

        .tolist()
    )


    undirected_edges = int(

        np.count_nonzero(
            np.triu(
                A,
                k=1
            )
        )
    )


    density = float(

        (
            2.0
            *
            undirected_edges
        )

        /
        (
            A.shape[0]
            *
            (
                A.shape[0]
                -
                1
            )
        )
    )


    return {

        "components":
            int(
                n_components
            ),

        "component_sizes":
            component_sizes,

        "isolates":
            isolates,

        "undirected_edges":
            undirected_edges,

        "density":
            density,

        "degree_min":
            float(
                degree.min()
            ),

        "degree_mean":
            float(
                degree.mean()
            ),

        "degree_max":
            float(
                degree.max()
            )
    }


# ============================================================
# 3. FROZEN STAGE0B-R2 INPUTS
# ============================================================

DESIGN_LOCK = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_DESIGN_LOCK.json"
)

SUBJECT_ORDER = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)

EEG_RAW = (
    INPUTS
    / "PRIMARY_EQUAL8_EEG128_RAW.npy"
)

PHENO_RAW = (
    INPUTS
    / "PRIMARY_BASELINE_NEOFFI60_RAW.npy"
)

PHENO_RAW_CSV = (
    INPUTS
    / "PRIMARY_BASELINE_NEOFFI60_RAW.csv"
)

COHORT_AUDIT = (
    AUDIT
    / "STAGE0B_R2_PRIMARY_FUSION_COHORT_AUDIT.csv"
)

STAGE0B_SUMMARY = (
    AUDIT
    / "STAGE0B_R2_BASELINE_PHENOTYPE_COHORT_SUMMARY.json"
)


EXPECTED_HASHES = {

    DESIGN_LOCK:
        "0c7b3db00154d450c26bbab92e2d289ac72ae4d09866767237f9e8733a0025a0",

    SUBJECT_ORDER:
        "8310ccc52f0d0698ed171ded6feb6f14cf4738787b21b93dbedcc0a1d1c99c6e",

    EEG_RAW:
        "3497f171c233331740f3c375bf414707d17718f89510d9f39941836e3bca24b8",

    PHENO_RAW:
        "87de4a2278c7877e0122b2e6b7921526cd9ce8d6d74f2343aecc3abe6c8be13a",

    PHENO_RAW_CSV:
        "0eacbeab89f74fb4bef8a435199a7585921c1df20e1dc980ddd4a8e0fba54224",

    COHORT_AUDIT:
        "81e0d6fcd75e6ac810a6a344f74f8f92a048b89d2faca9069b87e9456bce4d65",

    STAGE0B_SUMMARY:
        "030341f63d084d7a087d4f6b02a01e6f7ae6d909eb5d01c645e59dfa626cb334"
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
            "Actual  :",
            actual
        )


        raise RuntimeError(
            "Frozen Stage0B-R2 input changed. STOP."
        )


print(
    "\nALL STAGE0B-R2 INPUT HASHES PASS."
)


# ============================================================
# 4. VERIFY FROZEN DESIGN
# ============================================================

with open(
    DESIGN_LOCK,
    "r",
    encoding="utf-8"
) as f:

    design = json.load(
        f
    )


assert design[
    "status"
] == "FROZEN_BEFORE_GRAPH_CONSTRUCTION"


assert design[
    "phenotype_complete_case_rule"
][
    "primary_fusion_N"
] == 85


assert design[
    "primary_EEG_view"
][
    "dimension"
] == 128


assert design[
    "primary_phenotype_view"
][
    "dimension"
] == 60


assert design[
    "fusion"
][
    "alpha"
] == 0.6


assert design[
    "fusion"
][
    "TDBRAIN_alpha_retuning_permitted"
] is False


assert design[
    "graph_neighborhood_candidates"
] == [
    4,
    6,
    8,
    10,
    12
]


ALPHA = 0.6

K_CANDIDATES = [
    4,
    6,
    8,
    10,
    12
]


print(
    "\nFrozen design:"
)

print(
    "N = 85"
)

print(
    "EEG dimensions = 128"
)

print(
    "Phenotype dimensions = 60"
)

print(
    "alpha = 0.6"
)

print(
    "graph k candidates =",
    K_CANDIDATES
)


# ============================================================
# 5. LOAD PRIMARY SUBJECT ORDER
# ============================================================

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


print(
    "\nPrimary subject-order integrity: PASS"
)


# ============================================================
# 6. LOAD TWO FROZEN MODEL VIEWS
# ============================================================

X_eeg_raw = np.load(
    EEG_RAW
).astype(
    np.float64
)


X_phe_raw = np.load(
    PHENO_RAW
).astype(
    np.float64
)


assert X_eeg_raw.shape == (
    85,
    128
)


assert X_phe_raw.shape == (
    85,
    60
)


assert np.isfinite(
    X_eeg_raw
).all()


assert np.isfinite(
    X_phe_raw
).all()


if np.any(

    np.std(
        X_eeg_raw,
        axis=0,
        ddof=0
    )

    <= 0
):

    raise RuntimeError(
        "Zero-variance EEG feature detected."
    )


if np.any(

    np.std(
        X_phe_raw,
        axis=0,
        ddof=0
    )

    <= 0
):

    raise RuntimeError(
        "Zero-variance NEO-FFI feature detected."
    )


print(
    "\nRaw EEG:",
    X_eeg_raw.shape
)

print(
    "Raw phenotype:",
    X_phe_raw.shape
)


# ============================================================
# 7. STANDARDIZE EACH VIEW SEPARATELY
# ============================================================

eeg_scaler = StandardScaler()

phe_scaler = StandardScaler()


X_eeg = eeg_scaler.fit_transform(
    X_eeg_raw
)


X_phe = phe_scaler.fit_transform(
    X_phe_raw
)


assert X_eeg.shape == (
    85,
    128
)


assert X_phe.shape == (
    85,
    60
)


assert np.isfinite(
    X_eeg
).all()


assert np.isfinite(
    X_phe
).all()


# StandardScaler population SD.
assert np.max(

    np.abs(
        np.mean(
            X_eeg,
            axis=0
        )
    )

) < 1e-10


assert np.max(

    np.abs(
        np.mean(
            X_phe,
            axis=0
        )
    )

) < 1e-10


assert np.max(

    np.abs(

        np.std(
            X_eeg,
            axis=0,
            ddof=0
        )

        -
        1.0
    )

) < 1e-8


assert np.max(

    np.abs(

        np.std(
            X_phe,
            axis=0,
            ddof=0
        )

        -
        1.0
    )

) < 1e-8


print(
    "\nStandardization: PASS"
)

print(
    "EEG feature-wise mean ≈ 0, SD ≈ 1"
)

print(
    "NEO feature-wise mean ≈ 0, SD ≈ 1"
)


# ============================================================
# 8. SAVE STANDARDIZED VIEWS + SCALER PARAMETERS
# ============================================================

EEG_Z_FILE = (
    GRAPH_DIR
    / "STAGE1A_EEG128_STANDARDIZED.npy"
)

PHE_Z_FILE = (
    GRAPH_DIR
    / "STAGE1A_NEOFFI60_STANDARDIZED.npy"
)

SCALER_FILE = (
    GRAPH_DIR
    / "STAGE1A_STANDARDIZATION_PARAMETERS.npz"
)


np.save(
    EEG_Z_FILE,
    X_eeg.astype(
        np.float32
    )
)


np.save(
    PHE_Z_FILE,
    X_phe.astype(
        np.float32
    )
)


np.savez(

    SCALER_FILE,

    EEG_mean=eeg_scaler.mean_.astype(
        np.float64
    ),

    EEG_scale=eeg_scaler.scale_.astype(
        np.float64
    ),

    PHENOTYPE_mean=phe_scaler.mean_.astype(
        np.float64
    ),

    PHENOTYPE_scale=phe_scaler.scale_.astype(
        np.float64
    )
)


# ============================================================
# 9. NODE FEATURES
#
# EPG-SD-style concatenation:
#
#   x_n = [ EEG128 || NEO60 ]
#
# total = 188 dimensions
# ============================================================

X_node = np.concatenate(

    [
        X_eeg,
        X_phe
    ],

    axis=1
)


assert X_node.shape == (
    85,
    188
)


assert np.isfinite(
    X_node
).all()


NODE_FEATURE_FILE = (
    GRAPH_DIR
    / "STAGE1A_NODE_FEATURES188.npy"
)


np.save(
    NODE_FEATURE_FILE,
    X_node.astype(
        np.float32
    )
)


# ============================================================
# 10. VIEW-SPECIFIC COSINE SIMILARITIES
# ============================================================

A_eeg = cosine_similarity(
    X_eeg
).astype(
    np.float64
)


A_phe = cosine_similarity(
    X_phe
).astype(
    np.float64
)


assert A_eeg.shape == (
    85,
    85
)


assert A_phe.shape == (
    85,
    85
)


assert np.isfinite(
    A_eeg
).all()


assert np.isfinite(
    A_phe
).all()


assert np.allclose(
    A_eeg,
    A_eeg.T,
    atol=1e-12
)


assert np.allclose(
    A_phe,
    A_phe.T,
    atol=1e-12
)


assert np.allclose(

    np.diag(
        A_eeg
    ),

    1.0,

    atol=1e-10
)


assert np.allclose(

    np.diag(
        A_phe
    ),

    1.0,

    atol=1e-10
)


EEG_SIM_FILE = (
    GRAPH_DIR
    / "STAGE1A_EEG_COSINE_SIMILARITY.npy"
)

PHE_SIM_FILE = (
    GRAPH_DIR
    / "STAGE1A_NEOFFI_COSINE_SIMILARITY.npy"
)


np.save(
    EEG_SIM_FILE,
    A_eeg.astype(
        np.float32
    )
)


np.save(
    PHE_SIM_FILE,
    A_phe.astype(
        np.float32
    )
)


print(
    "\nEEG cosine similarity range:",
    f"{A_eeg.min():.6f}",
    "to",
    f"{A_eeg.max():.6f}"
)


print(
    "Phenotype cosine similarity range:",
    f"{A_phe.min():.6f}",
    "to",
    f"{A_phe.max():.6f}"
)


# ============================================================
# 11. PRESPECIFIED alpha=0.6 FUSION
# ============================================================

A_fused = (

    ALPHA
    *
    A_eeg

    +

    (
        1.0
        -
        ALPHA
    )
    *
    A_phe
)


assert A_fused.shape == (
    85,
    85
)


assert np.isfinite(
    A_fused
).all()


assert np.allclose(
    A_fused,
    A_fused.T,
    atol=1e-12
)


FUSED_SIM_FILE = (
    GRAPH_DIR
    / "STAGE1A_FUSED_SIMILARITY_ALPHA06.npy"
)


np.save(
    FUSED_SIM_FILE,
    A_fused.astype(
        np.float32
    )
)


print(
    "\nFused similarity:"
)

print(
    "alpha:",
    ALPHA
)

print(
    "EEG weight:",
    ALPHA
)

print(
    "Phenotype weight:",
    1.0 - ALPHA
)

print(
    "range:",
    f"{A_fused.min():.6f}",
    "to",
    f"{A_fused.max():.6f}"
)


# ============================================================
# 12. GRAPH-K CONNECTIVITY AUDIT
#
# NO clustering criterion is used here.
#
# We preserve similarity-profile kNN from archived EPG-SD.
# ============================================================

candidate_rows = []

candidate_graphs = {}


for k in K_CANDIDATES:

    adjacency = (
        build_similarity_profile_knn(
            A_fused,
            k
        )
    )


    diagnostics = graph_diagnostics(
        adjacency
    )


    candidate_graphs[
        k
    ] = adjacency


    candidate_rows.append({

        "k":
            k,

        "components":
            diagnostics[
                "components"
            ],

        "isolates":
            diagnostics[
                "isolates"
            ],

        "undirected_edges":
            diagnostics[
                "undirected_edges"
            ],

        "density":
            diagnostics[
                "density"
            ],

        "degree_min":
            diagnostics[
                "degree_min"
            ],

        "degree_mean":
            diagnostics[
                "degree_mean"
            ],

        "degree_max":
            diagnostics[
                "degree_max"
            ],

        "component_sizes":
            "|".join(
                map(
                    str,
                    diagnostics[
                        "component_sizes"
                    ]
                )
            ),

        "eligible_connected_graph":
            bool(

                diagnostics[
                    "components"
                ]
                ==
                1

                and

                diagnostics[
                    "isolates"
                ]
                ==
                0
            )
    })


graph_audit = pd.DataFrame(
    candidate_rows
)


GRAPH_AUDIT_FILE = (
    AUDIT
    / "STAGE1A_GRAPH_K_CONNECTIVITY_AUDIT.csv"
)


graph_audit.to_csv(
    GRAPH_AUDIT_FILE,
    index=False
)


print(
    "\n" + "=" * 120
)

print(
    "GRAPH-K CONNECTIVITY AUDIT"
)

print(
    "=" * 120
)


print(

    graph_audit.to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 13. SELECT SMALLEST CONNECTED K
#
# This rule was frozen BEFORE seeing graph results.
# ============================================================

eligible = graph_audit[

    graph_audit[
        "eligible_connected_graph"
    ]

].copy()


if len(
    eligible
) == 0:

    raise RuntimeError(

        "None of the frozen k candidates "
        "[4,6,8,10,12] produced a connected graph. "
        "STOP — do not extend k automatically."
    )


SELECTED_K = int(

    eligible[
        "k"
    ].min()
)


A_selected = candidate_graphs[
    SELECTED_K
]


selected_diag = graph_diagnostics(
    A_selected
)


assert selected_diag[
    "components"
] == 1


assert selected_diag[
    "isolates"
] == 0


print(
    "\nSelected graph k:",
    SELECTED_K
)

print(
    "Selection rule: smallest connected candidate"
)

print(
    "Components:",
    selected_diag[
        "components"
    ]
)

print(
    "Isolates:",
    selected_diag[
        "isolates"
    ]
)

print(
    "Undirected edges:",
    selected_diag[
        "undirected_edges"
    ]
)

print(
    "Degree range:",
    f"{selected_diag['degree_min']:.0f}",
    "to",
    f"{selected_diag['degree_max']:.0f}"
)

print(
    "Mean degree:",
    f"{selected_diag['degree_mean']:.4f}"
)


# ============================================================
# 14. SAVE SELECTED GRAPH
# ============================================================

ADJ_FILE = (
    GRAPH_DIR
    / f"STAGE1A_FUSED_BINARY_ADJ_K{SELECTED_K}.npy"
)


np.save(
    ADJ_FILE,
    A_selected.astype(
        np.float32
    )
)


# Also save sparse edge list for inspection.

edge_rows = []


for i in range(
    85
):

    for j in range(
        i + 1,
        85
    ):

        if A_selected[
            i,
            j
        ] > 0:

            edge_rows.append({

                "row_i":
                    i,

                "row_j":
                    j,

                "subject_i":
                    subjects.loc[
                        i,
                        "subject"
                    ],

                "subject_j":
                    subjects.loc[
                        j,
                        "subject"
                    ],

                "EEG_cosine":
                    float(
                        A_eeg[
                            i,
                            j
                        ]
                    ),

                "NEOFFI_cosine":
                    float(
                        A_phe[
                            i,
                            j
                        ]
                    ),

                "fused_similarity_alpha06":
                    float(
                        A_fused[
                            i,
                            j
                        ]
                    )
            })


edges = pd.DataFrame(
    edge_rows
)


EDGE_FILE = (
    GRAPH_DIR
    / f"STAGE1A_FUSED_GRAPH_K{SELECTED_K}_EDGES.csv"
)


edges.to_csv(
    EDGE_FILE,
    index=False
)


assert len(
    edges
) == selected_diag[
    "undirected_edges"
]


# ============================================================
# 15. SAVE MODALITY-SPECIFIC GRAPHS AT SAME K
#
# These are saved for future fair ablation comparison.
# They are NOT used to select k.
# ============================================================

A_eeg_graph = build_similarity_profile_knn(
    A_eeg,
    SELECTED_K
)

A_phe_graph = build_similarity_profile_knn(
    A_phe,
    SELECTED_K
)


EEG_GRAPH_FILE = (
    GRAPH_DIR
    / f"STAGE1A_EEG_ONLY_BINARY_ADJ_K{SELECTED_K}.npy"
)

PHE_GRAPH_FILE = (
    GRAPH_DIR
    / f"STAGE1A_NEOFFI_ONLY_BINARY_ADJ_K{SELECTED_K}.npy"
)


np.save(
    EEG_GRAPH_FILE,
    A_eeg_graph.astype(
        np.float32
    )
)


np.save(
    PHE_GRAPH_FILE,
    A_phe_graph.astype(
        np.float32
    )
)


eeg_graph_diag = graph_diagnostics(
    A_eeg_graph
)

phe_graph_diag = graph_diagnostics(
    A_phe_graph
)


print(
    "\nSame-k modality graph diagnostics:"
)

print(
    "EEG-only:",
    eeg_graph_diag
)

print(
    "NEO-only:",
    phe_graph_diag
)


# ============================================================
# 16. GRAPH SUMMARY
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE1A_FUSED_GRAPH_SUMMARY.json"
)


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE1A_STANDARDIZATION_AND_FUSED_GRAPH",

    "N":
        85,

    "EEG_dimension":
        128,

    "phenotype_dimension":
        60,

    "node_feature_dimension":
        188,

    "EEG_standardized":
        True,

    "phenotype_standardized":
        True,

    "EEG_similarity":
        "cosine",

    "phenotype_similarity":
        "cosine",

    "fusion_alpha":
        0.6,

    "fusion_formula":
        "0.6*EEG_cosine + 0.4*NEOFFI_cosine",

    "graph_implementation":
        "archived_EPGSD_similarity_profile_kNN_binary_connectivity",

    "graph_k_candidates":
        K_CANDIDATES,

    "graph_k_selection_rule":
        "smallest_connected_candidate_zero_isolates",

    "selected_k":
        SELECTED_K,

    "selected_graph":
        selected_diag,

    "EEG_only_same_k_graph":
        eeg_graph_diag,

    "phenotype_only_same_k_graph":
        phe_graph_diag,

    "diagnosis_used":
        False,

    "clinical_scores_used":
        False,

    "Experiment2_clusters_used":
        False,

    "VGAE_run":
        False,

    "UMAP_run":
        False,

    "clustering_run":
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
# 17. STAGE1A GRAPH LOCK
# ============================================================

STAGE1A_LOCK = (
    LOCK
    / "STAGE1A_FUSED_GRAPH_LOCK.json"
)


lock_content = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE1A_FUSED_GRAPH_LOCK",

    "status":
        "FROZEN_BEFORE_VGAE",

    "N":
        85,

    "subject_order_sha256":
        sha256_file(
            SUBJECT_ORDER
        ),

    "view_dimensions":
        {

            "EEG":
                128,

            "NEOFFI":
                60,

            "node_features":
                188
        },

    "standardization":
        "separate_featurewise_StandardScaler_for_each_view",

    "similarities":
        {

            "EEG":
                "cosine",

            "NEOFFI":
                "cosine"
        },

    "fusion":
        {

            "alpha":
                0.6,

            "EEG_weight":
                0.6,

            "NEOFFI_weight":
                0.4
        },

    "graph_construction":
        {

            "implementation":
                "kneighbors_graph_on_fused_similarity_rows",

            "neighbor_distance":
                "euclidean_distance_between_similarity_profiles",

            "mode":
                "connectivity",

            "weighted":
                False,

            "symmetrization":
                "elementwise_max_with_transpose",

            "candidate_k":
                K_CANDIDATES,

            "selection_rule":
                "smallest_connected_candidate_zero_isolates",

            "selected_k":
                SELECTED_K,

            "components":
                selected_diag[
                    "components"
                ],

            "isolates":
                selected_diag[
                    "isolates"
                ],

            "undirected_edges":
                selected_diag[
                    "undirected_edges"
                ]
        },

    "clinical_information_used_for_graph_selection":
        False,

    "K_clusters_selected":
        False,

    "input_hashes":
        {

            "Stage0B_R2_design_lock":
                sha256_file(
                    DESIGN_LOCK
                ),

            "subject_order":
                sha256_file(
                    SUBJECT_ORDER
                ),

            "EEG_raw":
                sha256_file(
                    EEG_RAW
                ),

            "phenotype_raw":
                sha256_file(
                    PHENO_RAW
                )
        },

    "output_hashes":
        {

            "EEG_standardized":
                sha256_file(
                    EEG_Z_FILE
                ),

            "phenotype_standardized":
                sha256_file(
                    PHE_Z_FILE
                ),

            "standardization_parameters":
                sha256_file(
                    SCALER_FILE
                ),

            "node_features188":
                sha256_file(
                    NODE_FEATURE_FILE
                ),

            "EEG_similarity":
                sha256_file(
                    EEG_SIM_FILE
                ),

            "phenotype_similarity":
                sha256_file(
                    PHE_SIM_FILE
                ),

            "fused_similarity":
                sha256_file(
                    FUSED_SIM_FILE
                ),

            "selected_adjacency":
                sha256_file(
                    ADJ_FILE
                ),

            "selected_edges":
                sha256_file(
                    EDGE_FILE
                ),

            "EEG_only_graph":
                sha256_file(
                    EEG_GRAPH_FILE
                ),

            "phenotype_only_graph":
                sha256_file(
                    PHE_GRAPH_FILE
                ),

            "graph_k_audit":
                sha256_file(
                    GRAPH_AUDIT_FILE
                ),

            "summary":
                sha256_file(
                    SUMMARY_FILE
                )
        }
}


if STAGE1A_LOCK.exists():

    with open(
        STAGE1A_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing = json.load(
            f
        )


    if existing != lock_content:

        raise RuntimeError(
            "Existing Stage1A graph lock differs. STOP."
        )


    print(
        "\nExisting Stage1A graph lock verified."
    )


else:

    with open(
        STAGE1A_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            lock_content,
            f,
            indent=2
        )


    print(
        "\nStage1A fused graph lock created."
    )


# ============================================================
# 18. FINAL OUTPUT INTEGRITY
# ============================================================

A_check = np.load(
    ADJ_FILE
)


X_check = np.load(
    NODE_FEATURE_FILE
)


assert A_check.shape == (
    85,
    85
)


assert X_check.shape == (
    85,
    188
)


assert np.allclose(
    A_check,
    A_check.T
)


assert np.all(
    np.diag(
        A_check
    )
    ==
    0
)


assert set(
    np.unique(
        A_check
    ).tolist()
).issubset(
    {
        0.0,
        1.0
    }
)


check_diag = graph_diagnostics(
    A_check
)


assert check_diag[
    "components"
] == 1


assert check_diag[
    "isolates"
] == 0


# ============================================================
# 19. HASHES
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 1A OUTPUT HASHES"
)

print(
    "=" * 120
)


print(
    "EEG standardized:",
    sha256_file(
        EEG_Z_FILE
    )
)

print(
    "NEO standardized:",
    sha256_file(
        PHE_Z_FILE
    )
)

print(
    "Node features188:",
    sha256_file(
        NODE_FEATURE_FILE
    )
)

print(
    "EEG similarity:",
    sha256_file(
        EEG_SIM_FILE
    )
)

print(
    "NEO similarity:",
    sha256_file(
        PHE_SIM_FILE
    )
)

print(
    "Fused similarity:",
    sha256_file(
        FUSED_SIM_FILE
    )
)

print(
    "Selected adjacency:",
    sha256_file(
        ADJ_FILE
    )
)

print(
    "Graph audit:",
    sha256_file(
        GRAPH_AUDIT_FILE
    )
)

print(
    "Summary:",
    sha256_file(
        SUMMARY_FILE
    )
)

print(
    "Stage1A lock:",
    sha256_file(
        STAGE1A_LOCK
    )
)


# ============================================================
# 20. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 1A PASS — "
    "EEG + NEO-FFI FUSED GRAPH FROZEN"
)

print(
    "=" * 120
)


print(
    "\nPrimary fusion cohort N:",
    85
)

print(
    "Node feature matrix:",
    X_node.shape
)

print(
    "Fusion alpha:",
    0.6
)

print(
    "Selected graph k:",
    SELECTED_K
)

print(
    "Graph components:",
    selected_diag[
        "components"
    ]
)

print(
    "Graph isolates:",
    selected_diag[
        "isolates"
    ]
)

print(
    "Undirected edges:",
    selected_diag[
        "undirected_edges"
    ]
)


print(
    "\nDiagnosis used for graph construction: NO"
)

print(
    "Clinical symptom scores used: NO"
)

print(
    "Experiment2 labels used: NO"
)


print(
    "\nNO VGAE HAS BEEN TRAINED."
)

print(
    "NO UMAP HAS BEEN RUN."
)

print(
    "NO CLUSTER NUMBER HAS BEEN SELECTED."
)

print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 2A — TRAIN THE FROZEN 188 -> 64 -> 16 VGAE "
    "ON THE FUSED GRAPH."
)