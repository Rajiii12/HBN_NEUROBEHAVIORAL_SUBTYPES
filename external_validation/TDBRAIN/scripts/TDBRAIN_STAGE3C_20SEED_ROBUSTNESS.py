# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 3C
# 20-SEED STRUCTURAL ROBUSTNESS OF FROZEN K=6 PARTITION
#
# AUTHORITATIVE:
#   N = 85
#   K = 6
#   primary seed = 42
#   sizes = [27, 15, 18, 10, 9, 6]
#
# PURPOSE
# -------
# Test whether the already-frozen K=6 solution is robust
# to UMAP stochastic initialization.
#
# SPECIAL AUDIT
# -------------
# Primary seed42 has:
#   UMAP 4NN components = [70, 15]
#
# Authoritative Cluster 1 (N=15) exactly equals the
# 15-subject disconnected component.
#
# Therefore Stage3C also evaluates:
#   - frequency of disconnected UMAP supports
#   - component counts across seeds
#   - per-cluster subject-set stability
#   - whether each frozen cluster repeatedly forms an
#     entire connected component
#
# IMPORTANT:
#   K REMAINS 6 regardless of this result.
#
# NO:
#   diagnosis
#   ADHD scores
#   YBOCS
#   BDI
#   clinical naming
#   K reselection
# ============================================================

from pathlib import Path
import hashlib
import json
import warnings
from itertools import combinations

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

from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import save_npz


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

ROBUSTNESS_DIR = (
    EXP3
    / "06_ROBUSTNESS"
)


ROBUSTNESS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 3C — 20-SEED STRUCTURAL ROBUSTNESS OF FROZEN K=6")
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


def cluster_sizes(labels, K=6):

    return np.array(
        [
            int(
                np.sum(
                    labels == k
                )
            )

            for k in range(
                K
            )
        ],
        dtype=int
    )


def hungarian_align(
    reference,
    candidate,
    K=6
):

    overlap = np.zeros(
        (
            K,
            K
        ),
        dtype=int
    )


    for ref_k in range(
        K
    ):

        for cand_k in range(
            K
        ):

            overlap[
                ref_k,
                cand_k
            ] = int(

                np.sum(

                    (
                        reference
                        ==
                        ref_k
                    )

                    &

                    (
                        candidate
                        ==
                        cand_k
                    )
                )
            )


    row_ind, col_ind = linear_sum_assignment(
        -overlap
    )


    candidate_to_reference = {

        int(
            cand
        ):
            int(
                ref
            )

        for ref, cand in zip(
            row_ind,
            col_ind
        )
    }


    aligned = np.array(

        [

            candidate_to_reference[
                int(x)
            ]

            for x in candidate

        ],

        dtype=int
    )


    return (
        aligned,
        overlap,
        candidate_to_reference
    )


def jaccard_score_binary(
    a,
    b
):

    intersection = int(

        np.sum(
            a
            &
            b
        )
    )


    union = int(

        np.sum(
            a
            |
            b
        )
    )


    if union == 0:

        return np.nan


    return float(
        intersection
        /
        union
    )


def dice_score_binary(
    a,
    b
):

    intersection = int(

        np.sum(
            a
            &
            b
        )
    )


    denominator = int(

        np.sum(
            a
        )

        +

        np.sum(
            b
        )
    )


    if denominator == 0:

        return np.nan


    return float(
        2
        *
        intersection
        /
        denominator
    )


# ============================================================
# 3. FROZEN INPUTS
# ============================================================

PARTITION_FILE = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PRIMARY_PARTITION.csv"
)

PARTITION_LOCK = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PARTITION_LOCK.json"
)

STAGE3B_SUMMARY = (
    AUDIT
    / "STAGE3B_PRIMARY_PARTITION_SUMMARY.json"
)

STAGE3B_REPORT = (
    AUDIT
    / "STAGE3B_AUTHORITATIVE_PARTITION_REPORT.txt"
)

COMPONENT_ASSIGNMENTS = (
    AUDIT
    / "STAGE3B_UMAP_COMPONENT_ASSIGNMENTS.csv"
)

MU_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.npy"
)

SUBJECT_ORDER = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)


EXPECTED_HASHES = {

    PARTITION_FILE:
        "f11a53d12e6676c7175622edef55984686363af051f4158b899fc321df50a782",

    PARTITION_LOCK:
        "53006cf5d1c66e15fc64b5dd7b707a6c90e1bde9315ef45cda0ae0ab747e8e85",

    STAGE3B_SUMMARY:
        "5173863ad1816f44ff704af1a18aa2ffe14395038b165c0b117b87592a2fbbf4",

    STAGE3B_REPORT:
        "8d99e8aadf90e9719ee47f3bc028d9eae99d69efc2064cda89f04b258c9e9f47",

    COMPONENT_ASSIGNMENTS:
        "fd91aa4e159b4b305d55da285dfd3d7831cd20bdc486904d76f09c9f8cc5bbbb",

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
            "Frozen Stage3B input changed. STOP."
        )


print(
    "\nALL STAGE3B INPUT HASHES PASS."
)


# ============================================================
# 4. VERIFY AUTHORITATIVE PARTITION
# ============================================================

with open(
    PARTITION_LOCK,
    "r",
    encoding="utf-8"
) as f:

    partition_lock = json.load(
        f
    )


assert partition_lock[
    "status"
] == "FROZEN"


assert partition_lock[
    "N"
] == 85


assert partition_lock[
    "K"
] == 6


assert partition_lock[
    "primary_UMAP_seed"
] == 42


assert partition_lock[
    "cluster_names_assigned"
] is False


partition = pd.read_csv(
    PARTITION_FILE
)


assert len(
    partition
) == 85


assert partition[
    "subject"
].nunique() == 85


assert np.array_equal(

    partition[
        "experiment3_row"
    ].to_numpy(
        dtype=int
    ),

    np.arange(
        85
    )
)


reference_labels = partition[
    "EEG_PHENO_cluster"
].to_numpy(
    dtype=int
)


assert len(
    np.unique(
        reference_labels
    )
) == 6


REFERENCE_SIZES = cluster_sizes(
    reference_labels
)


assert REFERENCE_SIZES.tolist() == [
    27,
    15,
    18,
    10,
    9,
    6
]


print(
    "\nAuthoritative partition:"
)

print(
    "N = 85"
)

print(
    "K = 6"
)

print(
    "Cluster sizes =",
    REFERENCE_SIZES.tolist()
)


# ============================================================
# 5. LOAD FROZEN VGAE REPRESENTATION
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


assert np.linalg.matrix_rank(
    mu
) == 16


# ============================================================
# 6. FREEZE 20-SEED ROBUSTNESS DESIGN
#
# BEFORE running any new seed.
# ============================================================

ROBUSTNESS_SEEDS = [

    42,
    52,
    62,
    72,
    82,
    92,
    102,
    112,
    122,
    132,
    142,
    152,
    162,
    172,
    182,
    192,
    202,
    212,
    222,
    232
]


assert len(
    ROBUSTNESS_SEEDS
) == 20


K = 6

MIN_CLUSTER_SIZE_GUARD = 5

MAX_BALANCE_GUARD = 5.0


DESIGN_LOCK = (
    LOCK
    / "STAGE3C_20SEED_ROBUSTNESS_DESIGN_LOCK.json"
)


design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3C_20SEED_STRUCTURAL_ROBUSTNESS",

    "status":
        "FROZEN_BEFORE_ROBUSTNESS_RUN",

    "N":
        85,

    "K":
        6,

    "K_reselection_allowed":
        False,

    "authoritative_seed":
        42,

    "robustness_seeds":
        ROBUSTNESS_SEEDS,

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

            "n_jobs":
                1
        },

    "spectral_affinity":
        {

            "n_neighbors":
                4,

            "metric":
                "euclidean",

            "mode":
                "connectivity",

            "binary":
                True,

            "symmetrization":
                "maximum_with_transpose"
        },

    "spectral_clustering":
        {

            "K":
                6,

            "affinity":
                "precomputed",

            "assign_labels":
                "kmeans",

            "random_state":
                42,

            "n_init":
                20
        },

    "reported_robustness":
        [

            "ARI_vs_authoritative",

            "NMI_vs_authoritative",

            "label_invariant_exact_partition_match",

            "all_190_pairwise_seed_ARI",

            "all_190_pairwise_seed_NMI",

            "silhouette",

            "Davies_Bouldin",

            "Calinski_Harabasz",

            "minimum_cluster_size",

            "balance_ratio",

            "UMAP_affinity_component_count",

            "UMAP_affinity_component_sizes",

            "cluster_specific_Hungarian_Jaccard",

            "cluster_specific_Hungarian_Dice",

            "cluster_exact_component_frequency"
        ],

    "original_selection_guards":
        {

            "minimum_cluster_size":
                5,

            "maximum_balance_ratio":
                5.0
        },

    "failure_of_guard_changes_K":
        False,

    "diagnosis_used":
        False,

    "clinical_scores_used":
        False,

    "cluster_names_used":
        False,

    "input_hashes":
        {

            "authoritative_partition":
                sha256_file(
                    PARTITION_FILE
                ),

            "partition_lock":
                sha256_file(
                    PARTITION_LOCK
                ),

            "VGAE_mu":
                sha256_file(
                    MU_FILE
                )
        }
}


if DESIGN_LOCK.exists():

    with open(
        DESIGN_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing = json.load(
            f
        )


    if existing != design:

        raise RuntimeError(
            "Existing Stage3C design lock differs. STOP."
        )


    print(
        "\nExisting Stage3C robustness design lock verified."
    )


else:

    with open(
        DESIGN_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            design,
            f,
            indent=2
        )


    print(
        "\nStage3C robustness design frozen BEFORE analysis."
    )


print(
    "Stage3C design SHA256:",
    sha256_file(
        DESIGN_LOCK
    )
)


# ============================================================
# 7. 20-SEED ROBUSTNESS
# ============================================================

seed_rows = []

cluster_rows = []

component_pair_rows = []

labels_by_seed = {}


print(
    "\n" + "=" * 120
)

print(
    "20-SEED K=6 STRUCTURAL ROBUSTNESS"
)

print(
    "=" * 120
)


for seed_index, seed in enumerate(
    ROBUSTNESS_SEEDS,
    start=1
):

    print(
        "\n" + "=" * 120
    )

    print(
        f"ROBUSTNESS SEED {seed_index}/20: {seed}"
    )

    print(
        "=" * 120
    )


    # --------------------------------------------------------
    # UMAP
    # --------------------------------------------------------

    model = UMAP(

        n_neighbors=30,

        n_components=2,

        min_dist=0.1,

        metric="euclidean",

        init="spectral",

        random_state=seed,

        transform_seed=seed,

        n_jobs=1
    )


    Y = model.fit_transform(
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
        ROBUSTNESS_DIR
        / f"STAGE3C_UMAP_SEED{seed}.npy"
    )


    np.save(
        UMAP_FILE,
        Y
    )


    # --------------------------------------------------------
    # exact 4NN affinity
    # --------------------------------------------------------

    A = build_umap_affinity(
        Y
    )


    components, component_labels = (
        connected_components(

            A,

            directed=False,

            return_labels=True
        )
    )


    degree = np.asarray(

        A.sum(
            axis=1
        )

    ).ravel()


    isolates = int(

        np.sum(
            degree == 0
        )
    )


    if isolates != 0:

        raise RuntimeError(
            f"Seed {seed} has isolated UMAP nodes. STOP."
        )


    edges = int(
        A.nnz
        //
        2
    )


    component_sizes_array = np.array(

        [

            np.sum(
                component_labels
                ==
                c
            )

            for c in range(
                components
            )

        ],

        dtype=int
    )


    component_sizes_sorted = sorted(

        component_sizes_array.tolist(),

        reverse=True
    )


    AFFINITY_FILE = (
        ROBUSTNESS_DIR
        / f"STAGE3C_4NN_AFFINITY_SEED{seed}.npz"
    )


    save_npz(
        AFFINITY_FILE,
        A
    )


    # --------------------------------------------------------
    # fixed K=6 spectral clustering
    # --------------------------------------------------------

    spectral = SpectralClustering(

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


        raw_labels = spectral.fit_predict(
            A
        ).astype(
            int
        )


    if len(
        np.unique(
            raw_labels
        )
    ) != 6:

        raise RuntimeError(
            f"Seed {seed} did not return six clusters."
        )


    # --------------------------------------------------------
    # Hungarian alignment to authoritative numeric IDs
    # --------------------------------------------------------

    (
        aligned_labels,
        overlap_matrix,
        candidate_to_reference
    ) = hungarian_align(

        reference_labels,
        raw_labels,
        K=6
    )


    labels_by_seed[
        seed
    ] = aligned_labels.copy()


    # --------------------------------------------------------
    # Overall agreement
    # --------------------------------------------------------

    ari = float(

        adjusted_rand_score(
            reference_labels,
            raw_labels
        )
    )


    nmi = float(

        normalized_mutual_info_score(
            reference_labels,
            raw_labels
        )
    )


    exact_partition = bool(

        np.array_equal(
            reference_labels,
            aligned_labels
        )
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    sil = float(

        silhouette_score(
            Y,
            raw_labels,
            metric="euclidean"
        )
    )


    db = float(

        davies_bouldin_score(
            Y,
            raw_labels
        )
    )


    ch = float(

        calinski_harabasz_score(
            Y,
            raw_labels
        )
    )


    # --------------------------------------------------------
    # Sizes after alignment to authoritative cluster IDs
    # --------------------------------------------------------

    aligned_sizes = cluster_sizes(
        aligned_labels,
        K=6
    )


    min_size = int(
        aligned_sizes.min()
    )


    max_size = int(
        aligned_sizes.max()
    )


    balance = float(
        max_size
        /
        min_size
    )


    guard_satisfied = bool(

        min_size
        >=
        MIN_CLUSTER_SIZE_GUARD

        and

        balance
        <=
        MAX_BALANCE_GUARD
    )


    # --------------------------------------------------------
    # Per-authoritative-cluster stability
    # --------------------------------------------------------

    exact_component_reference_clusters = []


    for ref_k in range(
        6
    ):

        ref_mask = (
            reference_labels
            ==
            ref_k
        )


        candidate_mask = (
            aligned_labels
            ==
            ref_k
        )


        intersection = int(

            np.sum(
                ref_mask
                &
                candidate_mask
            )
        )


        jaccard = jaccard_score_binary(
            ref_mask,
            candidate_mask
        )


        dice = dice_score_binary(
            ref_mask,
            candidate_mask
        )


        exact_subject_set = bool(

            np.array_equal(
                ref_mask,
                candidate_mask
            )
        )


        # ----------------------------------------------------
        # Does this candidate cluster equal an entire
        # connected component of this seed's 4NN affinity?
        # ----------------------------------------------------

        exact_component = False

        matching_component = None


        for component_id in range(
            components
        ):

            component_mask = (
                component_labels
                ==
                component_id
            )


            if np.array_equal(
                candidate_mask,
                component_mask
            ):

                exact_component = True

                matching_component = int(
                    component_id
                )

                break


        if exact_component:

            exact_component_reference_clusters.append(
                ref_k
            )


        cluster_rows.append({

            "seed":
                seed,

            "authoritative_cluster":
                ref_k,

            "authoritative_N":
                int(
                    REFERENCE_SIZES[
                        ref_k
                    ]
                ),

            "seed_cluster_N":
                int(
                    aligned_sizes[
                        ref_k
                    ]
                ),

            "intersection_N":
                intersection,

            "Jaccard":
                jaccard,

            "Dice":
                dice,

            "exact_subject_set_match":
                exact_subject_set,

            "cluster_is_exact_affinity_component":
                exact_component,

            "matching_affinity_component":
                matching_component
        })


    # --------------------------------------------------------
    # Record every exact component <-> cluster relation
    # --------------------------------------------------------

    for component_id in range(
        components
    ):

        component_mask = (
            component_labels
            ==
            component_id
        )


        for ref_k in range(
            6
        ):

            cluster_mask = (
                aligned_labels
                ==
                ref_k
            )


            if np.array_equal(
                component_mask,
                cluster_mask
            ):

                component_pair_rows.append({

                    "seed":
                        seed,

                    "affinity_component":
                        int(
                            component_id
                        ),

                    "component_N":
                        int(
                            np.sum(
                                component_mask
                            )
                        ),

                    "authoritative_cluster":
                        ref_k,

                    "cluster_N":
                        int(
                            np.sum(
                                cluster_mask
                            )
                        )
                })


    # --------------------------------------------------------
    # Save aligned labels
    # --------------------------------------------------------

    label_file = (
        ROBUSTNESS_DIR
        / f"STAGE3C_ALIGNED_LABELS_SEED{seed}.csv"
    )


    label_df = partition[
        [
            "experiment3_row",
            "subject",
            "subject_norm"
        ]
    ].copy()


    label_df[
        "raw_seed_cluster"
    ] = raw_labels


    label_df[
        "aligned_authoritative_cluster"
    ] = aligned_labels


    label_df.to_csv(
        label_file,
        index=False
    )


    # --------------------------------------------------------
    # Seed summary
    # --------------------------------------------------------

    seed_rows.append({

        "seed":
            seed,

        "affinity_connected_components":
            int(
                components
            ),

        "affinity_component_sizes":
            "|".join(
                map(
                    str,
                    component_sizes_sorted
                )
            ),

        "affinity_isolates":
            isolates,

        "affinity_edges":
            edges,

        "cluster0_N":
            int(
                aligned_sizes[
                    0
                ]
            ),

        "cluster1_N":
            int(
                aligned_sizes[
                    1
                ]
            ),

        "cluster2_N":
            int(
                aligned_sizes[
                    2
                ]
            ),

        "cluster3_N":
            int(
                aligned_sizes[
                    3
                ]
            ),

        "cluster4_N":
            int(
                aligned_sizes[
                    4
                ]
            ),

        "cluster5_N":
            int(
                aligned_sizes[
                    5
                ]
            ),

        "minimum_cluster_size":
            min_size,

        "maximum_cluster_size":
            max_size,

        "balance_ratio":
            balance,

        "silhouette":
            sil,

        "Davies_Bouldin":
            db,

        "Calinski_Harabasz":
            ch,

        "ARI_vs_authoritative":
            ari,

        "NMI_vs_authoritative":
            nmi,

        "exact_partition_match":
            exact_partition,

        "original_size_balance_guards_satisfied":
            guard_satisfied,

        "number_clusters_equal_entire_affinity_component":
            len(
                exact_component_reference_clusters
            ),

        "clusters_equal_entire_affinity_component":
            "|".join(
                map(
                    str,
                    exact_component_reference_clusters
                )
            )
    })


    print(
        "Affinity components:",
        components
    )

    print(
        "Component sizes:",
        component_sizes_sorted
    )

    print(
        "Affinity edges:",
        edges
    )

    print(
        "Aligned cluster sizes:",
        aligned_sizes.tolist()
    )

    print(
        "Minimum cluster size:",
        min_size
    )

    print(
        "Balance ratio:",
        f"{balance:.6f}"
    )

    print(
        "Silhouette:",
        f"{sil:.6f}"
    )

    print(
        "DB:",
        f"{db:.6f}"
    )

    print(
        "CH:",
        f"{ch:.3f}"
    )

    print(
        "ARI vs authoritative:",
        f"{ari:.6f}"
    )

    print(
        "NMI vs authoritative:",
        f"{nmi:.6f}"
    )

    print(
        "Exact partition match:",
        exact_partition
    )

    print(
        "Clusters exactly equal to graph components:",
        exact_component_reference_clusters
    )


# ============================================================
# 8. SAVE PER-SEED RESULTS
# ============================================================

seed_results = pd.DataFrame(
    seed_rows
)


cluster_stability = pd.DataFrame(
    cluster_rows
)


component_pairs = pd.DataFrame(
    component_pair_rows
)


SEED_RESULTS_FILE = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_RESULTS.csv"
)

CLUSTER_STABILITY_FILE = (
    AUDIT
    / "STAGE3C_CLUSTER_SPECIFIC_STABILITY.csv"
)

COMPONENT_PAIRS_FILE = (
    AUDIT
    / "STAGE3C_COMPONENT_CLUSTER_ALIGNMENT.csv"
)


seed_results.to_csv(
    SEED_RESULTS_FILE,
    index=False
)


cluster_stability.to_csv(
    CLUSTER_STABILITY_FILE,
    index=False
)


component_pairs.to_csv(
    COMPONENT_PAIRS_FILE,
    index=False
)


# ============================================================
# 9. ALL 190 PAIRWISE SEED COMPARISONS
# ============================================================

pairwise_rows = []


for seed_a, seed_b in combinations(
    ROBUSTNESS_SEEDS,
    2
):

    labels_a = labels_by_seed[
        seed_a
    ]

    labels_b = labels_by_seed[
        seed_b
    ]


    pair_ari = float(

        adjusted_rand_score(
            labels_a,
            labels_b
        )
    )


    pair_nmi = float(

        normalized_mutual_info_score(
            labels_a,
            labels_b
        )
    )


    exact_pair = bool(

        np.isclose(
            pair_ari,
            1.0
        )

        and

        np.isclose(
            pair_nmi,
            1.0
        )
    )


    pairwise_rows.append({

        "seed_a":
            seed_a,

        "seed_b":
            seed_b,

        "ARI":
            pair_ari,

        "NMI":
            pair_nmi,

        "exact_partition_match":
            exact_pair
    })


pairwise = pd.DataFrame(
    pairwise_rows
)


assert len(
    pairwise
) == 190


PAIRWISE_FILE = (
    AUDIT
    / "STAGE3C_190_PAIRWISE_SEED_STABILITY.csv"
)


pairwise.to_csv(
    PAIRWISE_FILE,
    index=False
)


# ============================================================
# 10. CLUSTER-SPECIFIC SUMMARY
# ============================================================

cluster_summary_rows = []


for ref_k in range(
    6
):

    c = cluster_stability[

        cluster_stability[
            "authoritative_cluster"
        ]
        ==
        ref_k

    ]


    cluster_summary_rows.append({

        "authoritative_cluster":
            ref_k,

        "authoritative_N":
            int(
                REFERENCE_SIZES[
                    ref_k
                ]
            ),

        "mean_seed_cluster_N":
            float(
                c[
                    "seed_cluster_N"
                ].mean()
            ),

        "SD_seed_cluster_N":
            float(
                c[
                    "seed_cluster_N"
                ].std(
                    ddof=0
                )
            ),

        "mean_Jaccard":
            float(
                c[
                    "Jaccard"
                ].mean()
            ),

        "SD_Jaccard":
            float(
                c[
                    "Jaccard"
                ].std(
                    ddof=0
                )
            ),

        "minimum_Jaccard":
            float(
                c[
                    "Jaccard"
                ].min()
            ),

        "maximum_Jaccard":
            float(
                c[
                    "Jaccard"
                ].max()
            ),

        "mean_Dice":
            float(
                c[
                    "Dice"
                ].mean()
            ),

        "exact_subject_set_matches":
            int(
                c[
                    "exact_subject_set_match"
                ].sum()
            ),

        "times_cluster_is_entire_affinity_component":
            int(
                c[
                    "cluster_is_exact_affinity_component"
                ].sum()
            )
    })


cluster_summary = pd.DataFrame(
    cluster_summary_rows
)


CLUSTER_SUMMARY_FILE = (
    AUDIT
    / "STAGE3C_CLUSTER_SPECIFIC_STABILITY_SUMMARY.csv"
)


cluster_summary.to_csv(
    CLUSTER_SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. OVERALL SUMMARY STATISTICS
# ============================================================

mean_ari_authoritative = float(

    seed_results[
        "ARI_vs_authoritative"
    ].mean()
)


sd_ari_authoritative = float(

    seed_results[
        "ARI_vs_authoritative"
    ].std(
        ddof=0
    )
)


min_ari_authoritative = float(

    seed_results[
        "ARI_vs_authoritative"
    ].min()
)


max_ari_authoritative = float(

    seed_results[
        "ARI_vs_authoritative"
    ].max()
)


mean_nmi_authoritative = float(

    seed_results[
        "NMI_vs_authoritative"
    ].mean()
)


sd_nmi_authoritative = float(

    seed_results[
        "NMI_vs_authoritative"
    ].std(
        ddof=0
    )
)


min_nmi_authoritative = float(

    seed_results[
        "NMI_vs_authoritative"
    ].min()
)


max_nmi_authoritative = float(

    seed_results[
        "NMI_vs_authoritative"
    ].max()
)


exact_authoritative_count = int(

    seed_results[
        "exact_partition_match"
    ].sum()
)


mean_pairwise_ari = float(

    pairwise[
        "ARI"
    ].mean()
)


sd_pairwise_ari = float(

    pairwise[
        "ARI"
    ].std(
        ddof=0
    )
)


min_pairwise_ari = float(

    pairwise[
        "ARI"
    ].min()
)


mean_pairwise_nmi = float(

    pairwise[
        "NMI"
    ].mean()
)


sd_pairwise_nmi = float(

    pairwise[
        "NMI"
    ].std(
        ddof=0
    )
)


min_pairwise_nmi = float(

    pairwise[
        "NMI"
    ].min()
)


exact_pairwise_count = int(

    pairwise[
        "exact_partition_match"
    ].sum()
)


disconnected_count = int(

    np.sum(

        seed_results[
            "affinity_connected_components"
        ]
        >
        1
    )
)


connected_count = int(
    20
    -
    disconnected_count
)


guard_count = int(

    seed_results[
        "original_size_balance_guards_satisfied"
    ].sum()
)


mean_silhouette = float(

    seed_results[
        "silhouette"
    ].mean()
)


sd_silhouette = float(

    seed_results[
        "silhouette"
    ].std(
        ddof=0
    )
)


component_frequency = (

    seed_results[
        "affinity_connected_components"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


# ============================================================
# 12. VERIFY PRIMARY SEED42
# ============================================================

primary_result = seed_results[

    seed_results[
        "seed"
    ]
    ==
    42

].iloc[
    0
]


assert np.isclose(

    float(
        primary_result[
            "ARI_vs_authoritative"
        ]
    ),

    1.0
)


assert np.isclose(

    float(
        primary_result[
            "NMI_vs_authoritative"
        ]
    ),

    1.0
)


assert bool(

    primary_result[
        "exact_partition_match"
    ]
)


assert int(

    primary_result[
        "affinity_connected_components"
    ]

) == 2


# Authoritative Cluster 1 must be an exact component at seed42.

primary_cluster1 = cluster_stability[

    (
        cluster_stability[
            "seed"
        ]
        ==
        42
    )

    &

    (
        cluster_stability[
            "authoritative_cluster"
        ]
        ==
        1
    )

].iloc[
    0
]


assert bool(

    primary_cluster1[
        "cluster_is_exact_affinity_component"
    ]
)


# ============================================================
# 13. SUMMARY JSON
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_SUMMARY.json"
)


cluster_summary_json = {}


for _, row in cluster_summary.iterrows():

    k = str(
        int(
            row[
                "authoritative_cluster"
            ]
        )
    )


    cluster_summary_json[
        k
    ] = {

        "authoritative_N":
            int(
                row[
                    "authoritative_N"
                ]
            ),

        "mean_Jaccard":
            float(
                row[
                    "mean_Jaccard"
                ]
            ),

        "minimum_Jaccard":
            float(
                row[
                    "minimum_Jaccard"
                ]
            ),

        "mean_Dice":
            float(
                row[
                    "mean_Dice"
                ]
            ),

        "exact_subject_set_matches_out_of_20":
            int(
                row[
                    "exact_subject_set_matches"
                ]
            ),

        "times_cluster_is_entire_affinity_component_out_of_20":
            int(
                row[
                    "times_cluster_is_entire_affinity_component"
                ]
            )
    }


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3C_20SEED_STRUCTURAL_ROBUSTNESS",

    "status":
        "COMPUTATIONAL_PASS",

    "N":
        85,

    "K":
        6,

    "K_changed":
        False,

    "seeds":
        ROBUSTNESS_SEEDS,

    "ARI_vs_authoritative":
        {

            "mean":
                mean_ari_authoritative,

            "SD":
                sd_ari_authoritative,

            "minimum":
                min_ari_authoritative,

            "maximum":
                max_ari_authoritative
        },

    "NMI_vs_authoritative":
        {

            "mean":
                mean_nmi_authoritative,

            "SD":
                sd_nmi_authoritative,

            "minimum":
                min_nmi_authoritative,

            "maximum":
                max_nmi_authoritative
        },

    "exact_partition_matches_vs_authoritative":
        exact_authoritative_count,

    "pairwise_190":
        {

            "mean_ARI":
                mean_pairwise_ari,

            "SD_ARI":
                sd_pairwise_ari,

            "minimum_ARI":
                min_pairwise_ari,

            "mean_NMI":
                mean_pairwise_nmi,

            "SD_NMI":
                sd_pairwise_nmi,

            "minimum_NMI":
                min_pairwise_nmi,

            "exact_partition_matches":
                exact_pairwise_count
        },

    "UMAP_affinity_connectivity":
        {

            "connected_seed_count":
                connected_count,

            "disconnected_seed_count":
                disconnected_count,

            "component_count_frequency":
                {

                    str(k):
                        int(v)

                    for k, v in
                    component_frequency.items()
                }
        },

    "original_size_balance_guards_satisfied":
        {

            "count":
                guard_count,

            "out_of":
                20
        },

    "silhouette_across_seeds":
        {

            "mean":
                mean_silhouette,

            "SD":
                sd_silhouette,

            "minimum":
                float(
                    seed_results[
                        "silhouette"
                    ].min()
                ),

            "maximum":
                float(
                    seed_results[
                        "silhouette"
                    ].max()
                )
        },

    "cluster_specific_stability":
        cluster_summary_json,

    "special_primary_cluster1_observation":
        {

            "authoritative_N":
                15,

            "seed42_is_exact_disconnected_component":
                True,

            "frequency_as_exact_component_across_20_seeds":
                int(

                    cluster_summary.loc[

                        cluster_summary[
                            "authoritative_cluster"
                        ]
                        ==
                        1,

                        "times_cluster_is_entire_affinity_component"

                    ].iloc[
                        0
                    ]
                )
        },

    "clinical_variables_used":
        False,

    "cluster_names_assigned":
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
# 14. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_REPORT.txt"
)


report = f"""
TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
STAGE 3C — 20-SEED STRUCTURAL ROBUSTNESS

AUTHORITATIVE PARTITION
-----------------------
N:
85

K:
6

Cluster sizes:
{REFERENCE_SIZES.tolist()}


AGREEMENT WITH AUTHORITATIVE PARTITION
--------------------------------------
ARI mean:
{mean_ari_authoritative:.10f}

ARI SD:
{sd_ari_authoritative:.10f}

ARI minimum:
{min_ari_authoritative:.10f}

ARI maximum:
{max_ari_authoritative:.10f}

NMI mean:
{mean_nmi_authoritative:.10f}

NMI SD:
{sd_nmi_authoritative:.10f}

NMI minimum:
{min_nmi_authoritative:.10f}

NMI maximum:
{max_nmi_authoritative:.10f}

Exact partition matches:
{exact_authoritative_count} / 20


ALL 190 PAIRWISE SEED COMPARISONS
---------------------------------
Mean pairwise ARI:
{mean_pairwise_ari:.10f}

SD pairwise ARI:
{sd_pairwise_ari:.10f}

Minimum pairwise ARI:
{min_pairwise_ari:.10f}

Mean pairwise NMI:
{mean_pairwise_nmi:.10f}

SD pairwise NMI:
{sd_pairwise_nmi:.10f}

Minimum pairwise NMI:
{min_pairwise_nmi:.10f}

Exact pairwise partition matches:
{exact_pairwise_count} / 190


UMAP 4NN CONNECTIVITY
---------------------
Connected supports:
{connected_count} / 20

Disconnected supports:
{disconnected_count} / 20

Component-count frequency:
{component_frequency}


ORIGINAL K-SELECTION SIZE/BALANCE GUARDS
----------------------------------------
Satisfied:
{guard_count} / 20


SILHOUETTE
----------
Mean:
{mean_silhouette:.10f}

SD:
{sd_silhouette:.10f}

Minimum:
{float(seed_results["silhouette"].min()):.10f}

Maximum:
{float(seed_results["silhouette"].max()):.10f}


CLUSTER-SPECIFIC STABILITY
--------------------------
{cluster_summary.to_string(index=False)}


IMPORTANT PRIMARY STRUCTURAL FEATURE
------------------------------------
At the authoritative seed42 solution, Cluster 1
(N=15) exactly equals the complete 15-subject second
connected component of the UMAP 4NN affinity.

Stage3C explicitly measures how often that relationship
recurs across the twenty UMAP initializations.

This property is structural only. It does not provide
clinical or diagnostic meaning.


INTERPRETATION RULE
-------------------
K=6 remains permanently frozen regardless of these
robustness results.

Poorer stability, if observed, is reported as a limitation.
It is not used to search for a more clinically attractive K.


DESIGN PROTECTION
-----------------
Diagnosis used:
NO

ADHD/YBOCS/BDI used:
NO

Clinical names assigned:
NO

K changed:
NO


FINAL STATUS
------------
STAGE 3C COMPUTATIONAL PASS —
20-SEED STRUCTURAL ROBUSTNESS COMPLETE.
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
# 15. OUTPUT LOCK
# ============================================================

OUTPUT_LOCK = (
    LOCK
    / "STAGE3C_20SEED_ROBUSTNESS_LOCK.json"
)


lock_content = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3C_20SEED_ROBUSTNESS_LOCK",

    "status":
        "FROZEN",

    "N":
        85,

    "K":
        6,

    "K_remains_frozen":
        True,

    "robustness_seeds":
        ROBUSTNESS_SEEDS,

    "exact_matches_vs_authoritative":
        exact_authoritative_count,

    "mean_ARI_vs_authoritative":
        mean_ari_authoritative,

    "mean_NMI_vs_authoritative":
        mean_nmi_authoritative,

    "mean_pairwise_ARI":
        mean_pairwise_ari,

    "mean_pairwise_NMI":
        mean_pairwise_nmi,

    "disconnected_affinity_seeds":
        disconnected_count,

    "connected_affinity_seeds":
        connected_count,

    "size_balance_guard_passes":
        guard_count,

    "clinical_variables_used":
        False,

    "output_hashes":
        {

            "seed_results":
                sha256_file(
                    SEED_RESULTS_FILE
                ),

            "pairwise_results":
                sha256_file(
                    PAIRWISE_FILE
                ),

            "cluster_specific_results":
                sha256_file(
                    CLUSTER_STABILITY_FILE
                ),

            "cluster_specific_summary":
                sha256_file(
                    CLUSTER_SUMMARY_FILE
                ),

            "component_cluster_alignment":
                sha256_file(
                    COMPONENT_PAIRS_FILE
                ),

            "summary":
                sha256_file(
                    SUMMARY_FILE
                ),

            "report":
                sha256_file(
                    REPORT_FILE
                )
        }
}


if OUTPUT_LOCK.exists():

    with open(
        OUTPUT_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing = json.load(
            f
        )


    if existing != lock_content:

        raise RuntimeError(
            "Existing Stage3C output lock differs. STOP."
        )


    print(
        "\nExisting Stage3C robustness lock verified."
    )


else:

    with open(
        OUTPUT_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            lock_content,
            f,
            indent=2
        )


    print(
        "\nStage3C robustness output lock created."
    )


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 3C — 20-SEED ROBUSTNESS RESULTS"
)

print(
    "=" * 120
)


display_cols = [

    "seed",

    "affinity_connected_components",

    "affinity_component_sizes",

    "cluster0_N",

    "cluster1_N",

    "cluster2_N",

    "cluster3_N",

    "cluster4_N",

    "cluster5_N",

    "minimum_cluster_size",

    "balance_ratio",

    "silhouette",

    "Davies_Bouldin",

    "Calinski_Harabasz",

    "ARI_vs_authoritative",

    "NMI_vs_authoritative",

    "exact_partition_match",

    "original_size_balance_guards_satisfied"
]


print(

    seed_results[
        display_cols
    ].to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"
    )
)


print(
    "\n" + "=" * 120
)

print(
    "OVERALL ROBUSTNESS"
)

print(
    "=" * 120
)


print(
    "ARI vs authoritative:"
)

print(
    " mean =",
    f"{mean_ari_authoritative:.6f}"
)

print(
    " SD   =",
    f"{sd_ari_authoritative:.6f}"
)

print(
    " min  =",
    f"{min_ari_authoritative:.6f}"
)

print(
    " max  =",
    f"{max_ari_authoritative:.6f}"
)


print(
    "\nNMI vs authoritative:"
)

print(
    " mean =",
    f"{mean_nmi_authoritative:.6f}"
)

print(
    " SD   =",
    f"{sd_nmi_authoritative:.6f}"
)

print(
    " min  =",
    f"{min_nmi_authoritative:.6f}"
)

print(
    " max  =",
    f"{max_nmi_authoritative:.6f}"
)


print(
    "\nExact partition matches vs authoritative:",
    exact_authoritative_count,
    "/ 20"
)


print(
    "\nMean pairwise ARI:",
    f"{mean_pairwise_ari:.6f}"
)

print(
    "Mean pairwise NMI:",
    f"{mean_pairwise_nmi:.6f}"
)

print(
    "Exact pairwise matches:",
    exact_pairwise_count,
    "/ 190"
)


print(
    "\nAffinities with >1 connected component:",
    disconnected_count,
    "/ 20"
)


print(
    "Original size/balance guards satisfied:",
    guard_count,
    "/ 20"
)


print(
    "\n" + "=" * 120
)

print(
    "CLUSTER-SPECIFIC SUBJECT STABILITY"
)

print(
    "=" * 120
)


print(

    cluster_summary.to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 17. HASHES
# ============================================================

print(
    "\nHASHES"
)


print(
    "Stage3C design lock:",
    sha256_file(
        DESIGN_LOCK
    )
)

print(
    "20-seed results:",
    sha256_file(
        SEED_RESULTS_FILE
    )
)

print(
    "190 pairwise results:",
    sha256_file(
        PAIRWISE_FILE
    )
)

print(
    "Cluster-specific stability:",
    sha256_file(
        CLUSTER_STABILITY_FILE
    )
)

print(
    "Cluster-specific summary:",
    sha256_file(
        CLUSTER_SUMMARY_FILE
    )
)

print(
    "Component alignment:",
    sha256_file(
        COMPONENT_PAIRS_FILE
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
    "Output lock:",
    sha256_file(
        OUTPUT_LOCK
    )
)


# ============================================================
# 18. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 3C PASS — "
    "20-SEED K=6 STRUCTURAL ROBUSTNESS COMPLETE"
)

print(
    "=" * 120
)


print(
    "\nK REMAINS FROZEN AT 6."
)

print(
    "DO NOT RESELECT K."
)

print(
    "NO CLINICAL VARIABLES WERE USED."
)

print(
    "NO CLUSTER NAMES WERE ASSIGNED."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 3D — FIXED-LABEL PERMUTATION SIGNIFICANCE "
    "OF THE AUTHORITATIVE K=6 INTERNAL GEOMETRY."
)