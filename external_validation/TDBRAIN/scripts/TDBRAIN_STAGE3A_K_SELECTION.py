# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 3A
# MULTI-SEED UMAP STRUCTURAL AUDIT + K SELECTION
#
# INPUT:
#   frozen VGAE posterior mean
#   N = 85
#   dimension = 16
#
# UMAP:
#   n_neighbors = 30
#   n_components = 2
#   min_dist = 0.1
#   metric = euclidean
#
# UMAP SEEDS:
#   42, 52, 62, 72, 82
#
# SPECTRAL AFFINITY:
#   exact 4NN on each UMAP embedding
#   Euclidean
#   connectivity graph
#   symmetric binary adjacency
#
# CANDIDATE K:
#   2, 3, 4, 5, 6
#
# IMPORTANT:
#   K-selection design is frozen BEFORE UMAP/clustering.
#
# K SELECTION USES ONLY:
#   silhouette
#   Davies-Bouldin
#   Calinski-Harabasz
#   minimum cluster size
#   balance ratio
#   cross-seed ARI
#   cross-seed NMI
#
# NO:
#   diagnosis
#   ADHD scores
#   YBOCS
#   BDI
#   oddball behavior
#   Experiment2 labels
#
# THIS STAGE SELECTS K.
# IT DOES NOT YET FREEZE THE AUTHORITATIVE PARTITION.
# ============================================================

from pathlib import Path
import hashlib
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from scipy.sparse.csgraph import connected_components


try:
    from umap import UMAP
except Exception as e:
    raise ImportError(
        "umap-learn is not available in this runtime."
    ) from e


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


STAGE3_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 3A — MULTI-SEED UMAP STRUCTURAL AUDIT + K SELECTION")
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


def affinity_diagnostics(A):

    components, labels = connected_components(

        A,

        directed=False,

        return_labels=True
    )


    degrees = np.asarray(
        A.sum(
            axis=1
        )
    ).ravel()


    component_sizes = (

        pd.Series(
            labels
        )
        .value_counts()
        .sort_values(
            ascending=False
        )
        .tolist()
    )


    undirected_edges = int(
        A.nnz
        //
        2
    )


    return {

        "components":
            int(
                components
            ),

        "isolates":
            int(
                np.sum(
                    degrees == 0
                )
            ),

        "undirected_edges":
            undirected_edges,

        "degree_min":
            float(
                degrees.min()
            ),

        "degree_mean":
            float(
                degrees.mean()
            ),

        "degree_max":
            float(
                degrees.max()
            ),

        "component_sizes":
            component_sizes
    }


def cluster_summary(labels):

    counts = np.bincount(
        labels
    )


    counts = counts[
        counts > 0
    ]


    minimum = int(
        counts.min()
    )


    maximum = int(
        counts.max()
    )


    balance = float(
        maximum
        /
        minimum
    )


    return (
        counts.tolist(),
        minimum,
        maximum,
        balance
    )


# ============================================================
# 3. FROZEN STAGE2A INPUTS
# ============================================================

STAGE2A_TRAINING_LOCK = (
    LOCK
    / "STAGE2A_VGAE_TRAINING_DESIGN_LOCK.json"
)

STAGE2A_OUTPUT_LOCK = (
    LOCK
    / "STAGE2A_VGAE_OUTPUT_LOCK.json"
)

MU_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.npy"
)

EMBEDDING_CSV = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.csv"
)

STAGE2A_SUMMARY = (
    AUDIT
    / "STAGE2A_VGAE_SUMMARY.json"
)

SUBJECT_ORDER = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)


EXPECTED_HASHES = {

    STAGE2A_TRAINING_LOCK:
        "146656ebb5b16cb338f78d4a91cb1de6788fcfb65f789fcf5acc6f015acce502",

    STAGE2A_OUTPUT_LOCK:
        "ab2abcbbda9ef7dfc77a1db3c3230d3927105b9fe193533718e56b02009a4912",

    MU_FILE:
        "89ef44d4aa3f718e59dc391f43df51baca891cda1178c05c23346f0c46963415",

    EMBEDDING_CSV:
        "959f02f60aaf01b74fe436a318ebc75b05b0a687b59a0a07abc145dc7280ff4d",

    STAGE2A_SUMMARY:
        "37484857b6b9f60c963a4c1e9c0a9b3fc8aa01f027d6afe1e191057157db1712",

    SUBJECT_ORDER:
        "8310ccc52f0d0698ed171ded6feb6f14cf4738787b21b93dbedcc0a1d1c99c6e"
}


print(
    "\n" + "=" * 120
)

print(
    "FROZEN STAGE2A INPUT HASH VERIFICATION"
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
            "Frozen Stage2A input changed. STOP."
        )


print(
    "\nALL STAGE2A INPUT HASHES PASS."
)


# ============================================================
# 4. VERIFY VGAE OUTPUT
# ============================================================

with open(
    STAGE2A_OUTPUT_LOCK,
    "r",
    encoding="utf-8"
) as f:

    vgae_lock = json.load(
        f
    )


assert vgae_lock[
    "status"
] == "FROZEN_BEFORE_UMAP"


assert vgae_lock[
    "N"
] == 85


assert vgae_lock[
    "architecture"
] == "188_to_64_to_16"


assert vgae_lock[
    "posterior_mu_rank"
] == 16


assert vgae_lock[
    "active_latent_dimensions"
] == 16


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


assert np.linalg.matrix_rank(
    mu
) == 16


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
    "\nFrozen VGAE posterior:"
)

print(
    "Shape:",
    mu.shape
)

print(
    "Rank:",
    np.linalg.matrix_rank(
        mu
    )
)


# ============================================================
# 5. FREEZE STAGE3A SELECTION DESIGN
#
# THIS OCCURS BEFORE ANY UMAP OR CLUSTERING RESULT.
# ============================================================

UMAP_SEEDS = [
    42,
    52,
    62,
    72,
    82
]

K_CANDIDATES = [
    2,
    3,
    4,
    5,
    6
]


UMAP_NEIGHBORS = 30

UMAP_COMPONENTS = 2

UMAP_MIN_DIST = 0.1

UMAP_METRIC = "euclidean"

UMAP_AFFINITY_K = 4


MIN_CLUSTER_SIZE_GUARD = 5

MAX_BALANCE_GUARD = 5.0


SELECTION_DESIGN_LOCK = (
    LOCK
    / "STAGE3A_K_SELECTION_DESIGN_LOCK.json"
)


selection_design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3A_K_SELECTION_DESIGN",

    "status":
        "FROZEN_BEFORE_UMAP_AND_CLUSTERING",

    "N":
        85,

    "input":
        "frozen_VGAE_posterior_mean_mu16",

    "UMAP":
        {

            "n_neighbors":
                30,

            "n_components":
                2,

            "min_dist":
                0.1,

            "metric":
                "euclidean",

            "init":
                "spectral",

            "seeds":
                UMAP_SEEDS,

            "n_jobs":
                1
        },

    "spectral_affinity":
        {

            "construction":
                "exact_4NN_on_each_UMAP_embedding",

            "n_neighbors":
                4,

            "metric":
                "euclidean",

            "mode":
                "connectivity",

            "symmetrization":
                "maximum_with_transpose",

            "binary":
                True,

            "isolates_allowed":
                False
        },

    "spectral_clustering":
        {

            "candidate_K":
                K_CANDIDATES,

            "affinity":
                "precomputed",

            "assign_labels":
                "kmeans",

            "random_state":
                42,

            "n_init":
                20
        },

    "eligibility_guards":
        {

            "minimum_cluster_size_each_seed":
                MIN_CLUSTER_SIZE_GUARD,

            "maximum_balance_ratio_each_seed":
                MAX_BALANCE_GUARD,

            "zero_affinity_isolates_each_seed":
                True,

            "automatic_relaxation_if_none_eligible":
                False
        },

    "selection_metrics":
        [

            "mean_silhouette_higher_better",

            "mean_Davies_Bouldin_lower_better",

            "mean_Calinski_Harabasz_higher_better",

            "mean_pairwise_seed_ARI_higher_better",

            "mean_pairwise_seed_NMI_higher_better",

            "mean_minimum_cluster_size_higher_better",

            "mean_balance_ratio_lower_better"
        ],

    "selection_rule":
        (

            "Among eligible K values, compute ordinal ranks "
            "for the seven frozen criteria with equal weight. "
            "Select the K with the smallest total rank."
        ),

    "tie_break_order":
        [

            "higher_mean_pairwise_ARI",

            "higher_mean_pairwise_NMI",

            "higher_mean_silhouette",

            "lower_K_for_parsimony"
        ],

    "clinical_variables_used":
        False,

    "diagnosis_used":
        False,

    "Experiment2_cluster_labels_used":
        False,

    "selected_K_known_before_lock":
        False,

    "authoritative_partition_frozen_in_this_stage":
        False,

    "input_hashes":
        {

            "Stage2A_training_lock":
                sha256_file(
                    STAGE2A_TRAINING_LOCK
                ),

            "Stage2A_output_lock":
                sha256_file(
                    STAGE2A_OUTPUT_LOCK
                ),

            "posterior_mu":
                sha256_file(
                    MU_FILE
                ),

            "subject_order":
                sha256_file(
                    SUBJECT_ORDER
                )
        }
}


if SELECTION_DESIGN_LOCK.exists():

    with open(
        SELECTION_DESIGN_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing_design = json.load(
            f
        )


    if existing_design != selection_design:

        raise RuntimeError(
            "Existing Stage3A K-selection design differs. STOP."
        )


    print(
        "\nExisting Stage3A K-selection design lock verified."
    )


else:

    with open(
        SELECTION_DESIGN_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selection_design,
            f,
            indent=2
        )


    print(
        "\nStage3A K-selection design frozen BEFORE UMAP/clustering."
    )


print(
    "Stage3A design SHA256:",
    sha256_file(
        SELECTION_DESIGN_LOCK
    )
)


# ============================================================
# 6. RUN FIVE FROZEN UMAP SEEDS
# ============================================================

all_result_rows = []

affinity_rows = []

labels_by_k = {
    K: {}
    for K in K_CANDIDATES
}


print(
    "\n" + "=" * 120
)

print(
    "MULTI-SEED UMAP + SPECTRAL AUDIT"
)

print(
    "=" * 120
)


for seed in UMAP_SEEDS:

    print(
        f"\n--- UMAP seed {seed} ---"
    )


    umap_model = UMAP(

        n_neighbors=UMAP_NEIGHBORS,

        n_components=UMAP_COMPONENTS,

        min_dist=UMAP_MIN_DIST,

        metric=UMAP_METRIC,

        init="spectral",

        random_state=seed,

        transform_seed=seed,

        n_jobs=1
    )


    Y = umap_model.fit_transform(
        mu
    ).astype(
        np.float32
    )


    assert Y.shape == (
        85,
        2
    )


    assert np.isfinite(
        Y
    ).all()


    UMAP_FILE = (
        STAGE3_DIR
        / f"STAGE3A_UMAP_SEED{seed}.npy"
    )


    np.save(
        UMAP_FILE,
        Y
    )


    # --------------------------------------------------------
    # exact frozen 4NN UMAP affinity
    # --------------------------------------------------------

    A_umap = build_umap_affinity(
        Y
    )


    diag = affinity_diagnostics(
        A_umap
    )


    if diag[
        "isolates"
    ] != 0:

        raise RuntimeError(
            f"UMAP seed {seed} produced isolated nodes. STOP."
        )


    AFFINITY_FILE = (
        STAGE3_DIR
        / f"STAGE3A_UMAP_4NN_AFFINITY_SEED{seed}.npz"
    )


    from scipy.sparse import save_npz

    save_npz(
        AFFINITY_FILE,
        A_umap
    )


    affinity_rows.append({

        "UMAP_seed":
            seed,

        "components":
            diag[
                "components"
            ],

        "isolates":
            diag[
                "isolates"
            ],

        "undirected_edges":
            diag[
                "undirected_edges"
            ],

        "degree_min":
            diag[
                "degree_min"
            ],

        "degree_mean":
            diag[
                "degree_mean"
            ],

        "degree_max":
            diag[
                "degree_max"
            ],

        "component_sizes":
            "|".join(
                map(
                    str,
                    diag[
                        "component_sizes"
                    ]
                )
            )
    })


    print(
        "4NN affinity:"
    )

    print(
        " components =",
        diag[
            "components"
        ]
    )

    print(
        " isolates =",
        diag[
            "isolates"
        ]
    )

    print(
        " edges =",
        diag[
            "undirected_edges"
        ]
    )


    # --------------------------------------------------------
    # Candidate K scan
    # --------------------------------------------------------

    for K in K_CANDIDATES:

        spectral = SpectralClustering(

            n_clusters=K,

            affinity="precomputed",

            assign_labels="kmeans",

            random_state=42,

            n_init=20
        )


        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )


            labels = spectral.fit_predict(
                A_umap
            ).astype(
                int
            )


        if len(
            np.unique(
                labels
            )
        ) != K:

            raise RuntimeError(
                f"Seed {seed}, K={K}: "
                "spectral clustering did not return K groups."
            )


        (
            sizes,
            min_size,
            max_size,
            balance
        ) = cluster_summary(
            labels
        )


        sil = float(

            silhouette_score(
                Y,
                labels,
                metric="euclidean"
            )
        )


        db = float(

            davies_bouldin_score(
                Y,
                labels
            )
        )


        ch = float(

            calinski_harabasz_score(
                Y,
                labels
            )
        )


        labels_by_k[
            K
        ][
            seed
        ] = labels.copy()


        LABEL_FILE = (
            STAGE3_DIR
            / f"STAGE3A_LABELS_SEED{seed}_K{K}.csv"
        )


        label_table = subjects[
            [
                "experiment3_row",
                "subject",
                "subject_norm"
            ]
        ].copy()


        label_table[
            "candidate_cluster"
        ] = labels


        label_table.to_csv(
            LABEL_FILE,
            index=False
        )


        eligible_run = bool(

            min_size
            >=
            MIN_CLUSTER_SIZE_GUARD

            and

            balance
            <=
            MAX_BALANCE_GUARD

            and

            diag[
                "isolates"
            ]
            ==
            0
        )


        all_result_rows.append({

            "UMAP_seed":
                seed,

            "K":
                K,

            "silhouette":
                sil,

            "Davies_Bouldin":
                db,

            "Calinski_Harabasz":
                ch,

            "minimum_cluster_size":
                min_size,

            "maximum_cluster_size":
                max_size,

            "balance_ratio":
                balance,

            "cluster_sizes":
                "|".join(
                    map(
                        str,
                        sizes
                    )
                ),

            "UMAP_affinity_components":
                diag[
                    "components"
                ],

            "UMAP_affinity_isolates":
                diag[
                    "isolates"
                ],

            "eligible_run":
                eligible_run
        })


        print(

            f"K={K} | "
            f"sizes={sizes} | "
            f"sil={sil:.6f} | "
            f"DB={db:.6f} | "
            f"CH={ch:.3f} | "
            f"min={min_size} | "
            f"balance={balance:.3f}"
        )


results = pd.DataFrame(
    all_result_rows
)


affinity_audit = pd.DataFrame(
    affinity_rows
)


PER_SEED_FILE = (
    AUDIT
    / "STAGE3A_PER_SEED_K_RESULTS.csv"
)


AFFINITY_AUDIT_FILE = (
    AUDIT
    / "STAGE3A_UMAP_4NN_AFFINITY_AUDIT.csv"
)


results.to_csv(
    PER_SEED_FILE,
    index=False
)


affinity_audit.to_csv(
    AFFINITY_AUDIT_FILE,
    index=False
)


# ============================================================
# 7. CROSS-SEED STABILITY FOR EACH K
#
# 5 seeds -> 10 seed-pair comparisons per K.
# ============================================================

pair_rows = []


for K in K_CANDIDATES:

    for seed_a, seed_b in combinations(
        UMAP_SEEDS,
        2
    ):

        labels_a = labels_by_k[
            K
        ][
            seed_a
        ]


        labels_b = labels_by_k[
            K
        ][
            seed_b
        ]


        ari = float(

            adjusted_rand_score(
                labels_a,
                labels_b
            )
        )


        nmi = float(

            normalized_mutual_info_score(
                labels_a,
                labels_b
            )
        )


        pair_rows.append({

            "K":
                K,

            "seed_a":
                seed_a,

            "seed_b":
                seed_b,

            "ARI":
                ari,

            "NMI":
                nmi
        })


pairwise = pd.DataFrame(
    pair_rows
)


PAIRWISE_FILE = (
    AUDIT
    / "STAGE3A_PAIRWISE_SEED_STABILITY.csv"
)


pairwise.to_csv(
    PAIRWISE_FILE,
    index=False
)


# ============================================================
# 8. AGGREGATE K-LEVEL RESULTS
# ============================================================

summary_rows = []


for K in K_CANDIDATES:

    r = results[
        results[
            "K"
        ]
        ==
        K
    ]


    p = pairwise[
        pairwise[
            "K"
        ]
        ==
        K
    ]


    all_seed_runs_eligible = bool(

        r[
            "eligible_run"
        ].all()
    )


    summary_rows.append({

        "K":
            K,

        "eligible":
            all_seed_runs_eligible,

        "mean_silhouette":
            float(
                r[
                    "silhouette"
                ].mean()
            ),

        "SD_silhouette":
            float(
                r[
                    "silhouette"
                ].std(
                    ddof=0
                )
            ),

        "mean_Davies_Bouldin":
            float(
                r[
                    "Davies_Bouldin"
                ].mean()
            ),

        "SD_Davies_Bouldin":
            float(
                r[
                    "Davies_Bouldin"
                ].std(
                    ddof=0
                )
            ),

        "mean_Calinski_Harabasz":
            float(
                r[
                    "Calinski_Harabasz"
                ].mean()
            ),

        "SD_Calinski_Harabasz":
            float(
                r[
                    "Calinski_Harabasz"
                ].std(
                    ddof=0
                )
            ),

        "mean_minimum_cluster_size":
            float(
                r[
                    "minimum_cluster_size"
                ].mean()
            ),

        "minimum_cluster_size_across_seeds":
            int(
                r[
                    "minimum_cluster_size"
                ].min()
            ),

        "mean_balance_ratio":
            float(
                r[
                    "balance_ratio"
                ].mean()
            ),

        "maximum_balance_ratio_across_seeds":
            float(
                r[
                    "balance_ratio"
                ].max()
            ),

        "mean_pairwise_ARI":
            float(
                p[
                    "ARI"
                ].mean()
            ),

        "SD_pairwise_ARI":
            float(
                p[
                    "ARI"
                ].std(
                    ddof=0
                )
            ),

        "minimum_pairwise_ARI":
            float(
                p[
                    "ARI"
                ].min()
            ),

        "mean_pairwise_NMI":
            float(
                p[
                    "NMI"
                ].mean()
            ),

        "SD_pairwise_NMI":
            float(
                p[
                    "NMI"
                ].std(
                    ddof=0
                )
            ),

        "minimum_pairwise_NMI":
            float(
                p[
                    "NMI"
                ].min()
            )
    })


k_summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 9. APPLY FROZEN EQUAL-WEIGHT RANK RULE
#
# ONLY eligible K values are ranked.
# ============================================================

eligible_summary = k_summary[

    k_summary[
        "eligible"
    ]

].copy()


if len(
    eligible_summary
) == 0:

    print(
        "\nNO K PASSED THE PROSPECTIVELY FROZEN "
        "MINIMUM-SIZE / BALANCE GUARDS."
    )

    print(
        "Do not relax thresholds automatically."
    )


    raise RuntimeError(
        "No eligible K. STOP before clinical opening."
    )


eligible_summary[
    "rank_silhouette"
] = eligible_summary[
    "mean_silhouette"
].rank(
    method="min",
    ascending=False
)


eligible_summary[
    "rank_DB"
] = eligible_summary[
    "mean_Davies_Bouldin"
].rank(
    method="min",
    ascending=True
)


eligible_summary[
    "rank_CH"
] = eligible_summary[
    "mean_Calinski_Harabasz"
].rank(
    method="min",
    ascending=False
)


eligible_summary[
    "rank_ARI"
] = eligible_summary[
    "mean_pairwise_ARI"
].rank(
    method="min",
    ascending=False
)


eligible_summary[
    "rank_NMI"
] = eligible_summary[
    "mean_pairwise_NMI"
].rank(
    method="min",
    ascending=False
)


eligible_summary[
    "rank_min_size"
] = eligible_summary[
    "mean_minimum_cluster_size"
].rank(
    method="min",
    ascending=False
)


eligible_summary[
    "rank_balance"
] = eligible_summary[
    "mean_balance_ratio"
].rank(
    method="min",
    ascending=True
)


rank_columns = [

    "rank_silhouette",

    "rank_DB",

    "rank_CH",

    "rank_ARI",

    "rank_NMI",

    "rank_min_size",

    "rank_balance"
]


eligible_summary[
    "total_equal_weight_rank"
] = eligible_summary[
    rank_columns
].sum(
    axis=1
)


# ============================================================
# 10. FROZEN TIE-BREAKING
# ============================================================

eligible_summary = eligible_summary.sort_values(

    by=[
        "total_equal_weight_rank",
        "mean_pairwise_ARI",
        "mean_pairwise_NMI",
        "mean_silhouette",
        "K"
    ],

    ascending=[
        True,
        False,
        False,
        False,
        True
    ]
).reset_index(
    drop=True
)


SELECTED_K = int(

    eligible_summary.iloc[
        0
    ][
        "K"
    ]
)


# Merge ranks back for complete table.

rank_merge_columns = [

    "K",
    *rank_columns,
    "total_equal_weight_rank"
]


k_summary = k_summary.merge(

    eligible_summary[
        rank_merge_columns
    ],

    on="K",

    how="left"
)


k_summary[
    "selected_K"
] = (

    k_summary[
        "K"
    ]
    ==
    SELECTED_K
)


K_SUMMARY_FILE = (
    AUDIT
    / "STAGE3A_K_SELECTION_SUMMARY.csv"
)


k_summary.to_csv(
    K_SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. PRIMARY-SEED RESULT FOR SELECTED K
#
# IMPORTANT:
# This is reported for inspection only.
#
# Stage3B will freeze the authoritative partition.
# ============================================================

primary_row = results[

    (
        results[
            "UMAP_seed"
        ]
        ==
        42
    )

    &

    (
        results[
            "K"
        ]
        ==
        SELECTED_K
    )

].iloc[
    0
]


selected_stability = eligible_summary[

    eligible_summary[
        "K"
    ]
    ==
    SELECTED_K

].iloc[
    0
]


# ============================================================
# 12. SAVE SELECTION RESULT LOCK
# ============================================================

SELECTION_RESULT_LOCK = (
    LOCK
    / "STAGE3A_K_SELECTION_RESULT_LOCK.json"
)


selection_result = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3A_K_SELECTION_RESULT",

    "status":
        "K_SELECTED_BEFORE_AUTHORITATIVE_PARTITION_FREEZE",

    "N":
        85,

    "candidate_K":
        K_CANDIDATES,

    "selected_K":
        SELECTED_K,

    "selection_basis":
        "internal_geometry_cluster_usability_and_cross_seed_stability_only",

    "clinical_variables_used":
        False,

    "diagnosis_used":
        False,

    "Experiment2_labels_used":
        False,

    "UMAP_seeds":
        UMAP_SEEDS,

    "primary_seed_for_next_partition_freeze":
        42,

    "selected_K_summary":
        {

            "mean_silhouette":
                float(
                    selected_stability[
                        "mean_silhouette"
                    ]
                ),

            "mean_Davies_Bouldin":
                float(
                    selected_stability[
                        "mean_Davies_Bouldin"
                    ]
                ),

            "mean_Calinski_Harabasz":
                float(
                    selected_stability[
                        "mean_Calinski_Harabasz"
                    ]
                ),

            "mean_minimum_cluster_size":
                float(
                    selected_stability[
                        "mean_minimum_cluster_size"
                    ]
                ),

            "mean_balance_ratio":
                float(
                    selected_stability[
                        "mean_balance_ratio"
                    ]
                ),

            "mean_pairwise_ARI":
                float(
                    selected_stability[
                        "mean_pairwise_ARI"
                    ]
                ),

            "mean_pairwise_NMI":
                float(
                    selected_stability[
                        "mean_pairwise_NMI"
                    ]
                ),

            "total_equal_weight_rank":
                float(
                    selected_stability[
                        "total_equal_weight_rank"
                    ]
                )
        },

    "primary_seed42_preview":
        {

            "cluster_sizes":
                primary_row[
                    "cluster_sizes"
                ],

            "silhouette":
                float(
                    primary_row[
                        "silhouette"
                    ]
                ),

            "Davies_Bouldin":
                float(
                    primary_row[
                        "Davies_Bouldin"
                    ]
                ),

            "Calinski_Harabasz":
                float(
                    primary_row[
                        "Calinski_Harabasz"
                    ]
                ),

            "minimum_cluster_size":
                int(
                    primary_row[
                        "minimum_cluster_size"
                    ]
                ),

            "balance_ratio":
                float(
                    primary_row[
                        "balance_ratio"
                    ]
                )
        },

    "authoritative_partition_frozen":
        False,

    "next_stage":
        "STAGE3B_FREEZE_PRIMARY_SEED42_PARTITION",

    "input_hashes":
        {

            "Stage3A_selection_design":
                sha256_file(
                    SELECTION_DESIGN_LOCK
                ),

            "Stage2A_output_lock":
                sha256_file(
                    STAGE2A_OUTPUT_LOCK
                ),

            "posterior_mu":
                sha256_file(
                    MU_FILE
                )
        },

    "output_hashes":
        {

            "per_seed_results":
                sha256_file(
                    PER_SEED_FILE
                ),

            "affinity_audit":
                sha256_file(
                    AFFINITY_AUDIT_FILE
                ),

            "pairwise_stability":
                sha256_file(
                    PAIRWISE_FILE
                ),

            "K_summary":
                sha256_file(
                    K_SUMMARY_FILE
                )
        }
}


if SELECTION_RESULT_LOCK.exists():

    with open(
        SELECTION_RESULT_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing_result = json.load(
            f
        )


    if existing_result != selection_result:

        raise RuntimeError(
            "Existing Stage3A K-selection result differs. STOP."
        )


    print(
        "\nExisting Stage3A K-selection result lock verified."
    )


else:

    with open(
        SELECTION_RESULT_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selection_result,
            f,
            indent=2
        )


    print(
        "\nStage3A K-selection result lock created."
    )


# ============================================================
# 13. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE3A_K_SELECTION_REPORT.txt"
)


report_lines = []


report_lines.append(
    "TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3"
)

report_lines.append(
    "STAGE 3A — MULTI-SEED UMAP STRUCTURAL AUDIT + K SELECTION"
)

report_lines.append(
    ""
)

report_lines.append(
    "Subjects: 85"
)

report_lines.append(
    "UMAP seeds: 42, 52, 62, 72, 82"
)

report_lines.append(
    "Candidate K: 2, 3, 4, 5, 6"
)

report_lines.append(
    ""
)

report_lines.append(
    "K-selection used internal geometry, cluster usability, "
    "and cross-seed stability only."
)

report_lines.append(
    "Diagnosis and held-out clinical variables were not used."
)

report_lines.append(
    ""
)

report_lines.append(
    f"SELECTED K: {SELECTED_K}"
)

report_lines.append(
    ""
)

report_lines.append(
    "Selected-K mean metrics:"
)

report_lines.append(
    f"Silhouette: "
    f"{selected_stability['mean_silhouette']:.10f}"
)

report_lines.append(
    f"Davies-Bouldin: "
    f"{selected_stability['mean_Davies_Bouldin']:.10f}"
)

report_lines.append(
    f"Calinski-Harabasz: "
    f"{selected_stability['mean_Calinski_Harabasz']:.10f}"
)

report_lines.append(
    f"Pairwise seed ARI: "
    f"{selected_stability['mean_pairwise_ARI']:.10f}"
)

report_lines.append(
    f"Pairwise seed NMI: "
    f"{selected_stability['mean_pairwise_NMI']:.10f}"
)

report_lines.append(
    f"Mean minimum cluster size: "
    f"{selected_stability['mean_minimum_cluster_size']:.4f}"
)

report_lines.append(
    f"Mean balance ratio: "
    f"{selected_stability['mean_balance_ratio']:.10f}"
)

report_lines.append(
    ""
)

report_lines.append(
    "Primary seed42 preview:"
)

report_lines.append(
    f"Cluster sizes: {primary_row['cluster_sizes']}"
)

report_lines.append(
    f"Silhouette: {primary_row['silhouette']:.10f}"
)

report_lines.append(
    f"Davies-Bouldin: {primary_row['Davies_Bouldin']:.10f}"
)

report_lines.append(
    f"Calinski-Harabasz: "
    f"{primary_row['Calinski_Harabasz']:.10f}"
)

report_lines.append(
    ""
)

report_lines.append(
    "The authoritative subject partition has NOT yet been frozen."
)

report_lines.append(
    "That occurs in Stage3B using the already-selected K "
    "and primary UMAP seed42."
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# 14. PRINT FINAL TABLE
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "K-LEVEL MULTI-SEED SUMMARY"
)

print(
    "=" * 120
)


display_columns = [

    "K",

    "eligible",

    "mean_silhouette",

    "mean_Davies_Bouldin",

    "mean_Calinski_Harabasz",

    "mean_minimum_cluster_size",

    "mean_balance_ratio",

    "mean_pairwise_ARI",

    "mean_pairwise_NMI",

    "total_equal_weight_rank",

    "selected_K"
]


print(

    k_summary[
        display_columns
    ].to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"
    )
)


print(
    "\n" + "=" * 120
)

print(
    "SELECTED K"
)

print(
    "=" * 120
)


print(
    "Selected K:",
    SELECTED_K
)


print(
    "\nMean pairwise seed ARI:",
    f"{selected_stability['mean_pairwise_ARI']:.6f}"
)

print(
    "Mean pairwise seed NMI:",
    f"{selected_stability['mean_pairwise_NMI']:.6f}"
)

print(
    "Mean silhouette:",
    f"{selected_stability['mean_silhouette']:.6f}"
)

print(
    "Mean DB:",
    f"{selected_stability['mean_Davies_Bouldin']:.6f}"
)

print(
    "Mean CH:",
    f"{selected_stability['mean_Calinski_Harabasz']:.6f}"
)


print(
    "\nPrimary seed42 preview:"
)

print(
    "Cluster sizes:",
    primary_row[
        "cluster_sizes"
    ]
)

print(
    "Silhouette:",
    f"{primary_row['silhouette']:.6f}"
)

print(
    "DB:",
    f"{primary_row['Davies_Bouldin']:.6f}"
)

print(
    "CH:",
    f"{primary_row['Calinski_Harabasz']:.6f}"
)


# ============================================================
# 15. HASHES
# ============================================================

print(
    "\nHASHES"
)


print(
    "Selection design:",
    sha256_file(
        SELECTION_DESIGN_LOCK
    )
)

print(
    "Per-seed results:",
    sha256_file(
        PER_SEED_FILE
    )
)

print(
    "UMAP affinity audit:",
    sha256_file(
        AFFINITY_AUDIT_FILE
    )
)

print(
    "Pairwise stability:",
    sha256_file(
        PAIRWISE_FILE
    )
)

print(
    "K summary:",
    sha256_file(
        K_SUMMARY_FILE
    )
)

print(
    "Selection result lock:",
    sha256_file(
        SELECTION_RESULT_LOCK
    )
)

print(
    "Report:",
    sha256_file(
        REPORT_FILE
    )
)


# ============================================================
# 16. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 3A PASS — "
    "EEG+PHENOTYPE K SELECTED"
)

print(
    "=" * 120
)


print(
    "\nSELECTED K =",
    SELECTED_K
)


print(
    "\nDiagnosis used for K selection: NO"
)

print(
    "ADHD/YBOCS/BDI used for K selection: NO"
)

print(
    "Experiment2 labels used for K selection: NO"
)


print(
    "\nIMPORTANT:"
)

print(
    "K is now selected and must not be changed "
    "because of later clinical results."
)

print(
    "The authoritative subject labels are NOT yet frozen."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 3B — FREEZE THE AUTHORITATIVE "
    "PRIMARY-SEED42 PARTITION AT THE SELECTED K."
)