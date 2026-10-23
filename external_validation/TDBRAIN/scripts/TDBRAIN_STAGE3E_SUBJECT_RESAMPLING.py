# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 3E
# SUBJECT-RESAMPLING ROBUSTNESS OF FROZEN K=6 SOLUTION
#
# PRIMARY:
#   N = 85
#   K = 6
#   UMAP seed = 42
#
# RESAMPLING:
#   A) 100 x 80% subsampling WITHOUT replacement
#      n = 68 each iteration
#
#   B) 100 x bootstrap WITH replacement
#      n = 85 draws each iteration
#
# IMPORTANT:
#   - use the already-frozen seed42 UMAP coordinates
#   - rebuild exact 4NN affinity on each sampled set
#   - spectral clustering K remains 6
#   - compare sampled labels with authoritative labels
#     for those sampled observations using ARI / NMI
#
# THIS IS FIXED-MANIFOLD DOWNSTREAM RESAMPLING.
#
# THIS STAGE DOES NOT:
#   retrain EEG
#   refit NEO scaling
#   rebuild EEG+phenotype fusion
#   retrain VGAE
#   refit UMAP
#   reselect K
#   use diagnosis
#   use ADHD/YBOCS/BDI
#   assign clinical names
# ============================================================

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

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

STAGE3_DIR = (
    EXP3
    / "05_UMAP_K_SELECTION"
)

RESAMPLING_DIR = (
    EXP3
    / "07_RESAMPLING"
)


RESAMPLING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 3E — SUBJECT-RESAMPLING ROBUSTNESS OF FROZEN K=6")
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


def find_file_by_sha256(
    root,
    expected_sha
):

    matches = []

    allowed_suffixes = {
        ".json",
        ".csv",
        ".txt",
        ".npy",
        ".npz"
    }


    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue


        try:

            actual = sha256_file(
                path
            )

        except Exception:

            continue


        if actual == expected_sha:

            matches.append(
                path
            )


    if len(matches) == 0:

        raise FileNotFoundError(
            "Could not locate file with SHA256:\n"
            + expected_sha
        )


    if len(matches) > 1:

        print(
            "\nWARNING: multiple identical files found for SHA256:"
        )

        print(
            expected_sha
        )

        for p in matches:

            print(
                " -",
                p
            )


    return matches[0]


def build_affinity(Y):

    n = len(
        Y
    )


    if n <= 4:

        raise RuntimeError(
            "Sample too small for frozen 4NN affinity."
        )


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


def graph_diagnostics(A):

    components, labels = connected_components(

        A,

        directed=False,

        return_labels=True
    )


    degree = np.asarray(

        A.sum(
            axis=1
        )

    ).ravel()


    component_sizes = sorted(

        [

            int(
                np.sum(
                    labels == c
                )
            )

            for c in range(
                components
            )

        ],

        reverse=True
    )


    return {

        "components":
            int(
                components
            ),

        "isolates":
            int(
                np.sum(
                    degree == 0
                )
            ),

        "edges":
            int(
                A.nnz
                //
                2
            ),

        "component_sizes":
            component_sizes
    }


def cluster_size_stats(
    labels,
    K=6
):

    counts = np.array(

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


    positive = counts[
        counts > 0
    ]


    if len(
        positive
    ) == 0:

        return (
            counts,
            0,
            0,
            np.nan
        )


    min_size = int(
        positive.min()
    )


    max_size = int(
        positive.max()
    )


    balance = float(

        max_size
        /
        min_size
    )


    return (
        counts,
        min_size,
        max_size,
        balance
    )


def empirical_ci(values):

    values = np.asarray(
        values,
        dtype=float
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return (
            np.nan,
            np.nan
        )


    return (

        float(
            np.percentile(
                values,
                2.5
            )
        ),

        float(
            np.percentile(
                values,
                97.5
            )
        )
    )


# ============================================================
# 3. AUTHORITATIVE INPUTS
# ============================================================

PARTITION_FILE = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PRIMARY_PARTITION.csv"
)

PARTITION_LOCK = (
    LOCK
    / "STAGE3B_AUTHORITATIVE_PARTITION_LOCK.json"
)

STAGE3C_LOCK = (
    LOCK
    / "STAGE3C_20SEED_ROBUSTNESS_LOCK.json"
)

STAGE3C_SUMMARY = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_SUMMARY.json"
)

PRIMARY_UMAP = (
    STAGE3_DIR
    / "STAGE3A_UMAP_SEED42.npy"
)


EXPECTED_KNOWN_HASHES = {

    PARTITION_FILE:
        "f11a53d12e6676c7175622edef55984686363af051f4158b899fc321df50a782",

    PARTITION_LOCK:
        "53006cf5d1c66e15fc64b5dd7b707a6c90e1bde9315ef45cda0ae0ab747e8e85",

    STAGE3C_LOCK:
        "fe34076b64b7dd59896a2ba977e3a60be44fe7a4845a2d1e8ce538902313476e",

    STAGE3C_SUMMARY:
        "545f604c12d9eb1affeb7af2b2a38cdbefe5b71fcfa70b82f81ffef7845a81f7"
}


# ============================================================
# 4. VERIFY KNOWN FROZEN INPUTS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FROZEN INPUT HASH VERIFICATION"
)

print(
    "=" * 120
)


for path, expected in EXPECTED_KNOWN_HASHES.items():

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
            "Frozen input changed. STOP."
        )


if not PRIMARY_UMAP.exists():

    raise FileNotFoundError(
        str(
            PRIMARY_UMAP
        )
    )


print(
    "PASS",
    PRIMARY_UMAP.name
)


# ============================================================
# 5. LOCATE + VERIFY STAGE3D OUTPUTS BY SHA256
#
# We do not guess their filenames.
# ============================================================

STAGE3D_EXPECTED = {

    "permutation_design":
        "87d9ddb059ce76738fc6b3353067192cc7444a11883c93f5c3ae0e73845c6299",

    "1000_permutations":
        "5f8925f7d8f84dd1accf325782a98470e85171d5e0c8e360ef097b4051962134",

    "result_table":
        "1dbf2a6859b6ba1b43c5f6c976353d1da86a7e5757911465d5c7d442862ee8fd",

    "summary":
        "50bc42da492ce95d2bc364e4d35a1d31bede848ec9dbe5772e6703d2548678d3",

    "report":
        "17e67cb04e2b5870b540556739cd5cdea22cc30c50ac96b71e01f8ebd5a2c51d",

    "output_lock":
        "f983c2c504b1d2329979a75d1e0ce27b0ecb8676e2adc96782c52a2345606efb"
}


print(
    "\n" + "=" * 120
)

print(
    "LOCATING STAGE3D OUTPUTS BY FROZEN SHA256"
)

print(
    "=" * 120
)


stage3d_paths = {}


for label, expected_sha in STAGE3D_EXPECTED.items():

    path = find_file_by_sha256(

        EXP3,

        expected_sha
    )


    stage3d_paths[
        label
    ] = path


    print(
        "PASS",
        label,
        "->",
        path.relative_to(
            EXP3
        )
    )


print(
    "\nALL STAGE3D FROZEN HASHES VERIFIED."
)


# ============================================================
# 6. LOAD AUTHORITATIVE PARTITION + UMAP
# ============================================================

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


reference_sizes = np.array(

    [

        int(
            np.sum(
                reference_labels == k
            )
        )

        for k in range(
            6
        )
    ],

    dtype=int
)


assert reference_sizes.tolist() == [
    27,
    15,
    18,
    10,
    9,
    6
]


Y_full = np.load(
    PRIMARY_UMAP
).astype(
    np.float64
)


assert Y_full.shape == (
    85,
    2
)


assert np.isfinite(
    Y_full
).all()


print(
    "\nAuthoritative state:"
)

print(
    "N = 85"
)

print(
    "K = 6"
)

print(
    "Cluster sizes =",
    reference_sizes.tolist()
)

print(
    "Fixed UMAP shape =",
    Y_full.shape
)


# ============================================================
# 7. FREEZE STAGE3E DESIGN BEFORE RESAMPLING
# ============================================================

N = 85

K = 6

SUBSAMPLE_FRACTION = 0.80

SUBSAMPLE_N = int(
    np.floor(
        SUBSAMPLE_FRACTION
        *
        N
    )
)


assert SUBSAMPLE_N == 68


N_SUBSAMPLE = 100

N_BOOTSTRAP = 100


SUBSAMPLE_RNG_SEED = 20261009

BOOTSTRAP_RNG_SEED = 20261010


SPECTRAL_RANDOM_STATE = 42

SPECTRAL_N_INIT = 20

AFFINITY_K = 4


DESIGN_LOCK = (
    LOCK
    / "STAGE3E_RESAMPLING_DESIGN_LOCK.json"
)


design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3E_SUBJECT_RESAMPLING_ROBUSTNESS",

    "status":
        "FROZEN_BEFORE_RESAMPLING",

    "N":
        N,

    "K":
        K,

    "K_reselection_allowed":
        False,

    "input_manifold":
        "frozen_primary_seed42_UMAP",

    "UMAP_refit_per_iteration":
        False,

    "VGAE_refit_per_iteration":
        False,

    "fusion_refit_per_iteration":
        False,

    "subsampling":
        {

            "iterations":
                N_SUBSAMPLE,

            "fraction":
                SUBSAMPLE_FRACTION,

            "sample_size":
                SUBSAMPLE_N,

            "replacement":
                False,

            "rng_seed":
                SUBSAMPLE_RNG_SEED
        },

    "bootstrap":
        {

            "iterations":
                N_BOOTSTRAP,

            "draw_size":
                N,

            "replacement":
                True,

            "rng_seed":
                BOOTSTRAP_RNG_SEED
        },

    "sampled_affinity":
        {

            "n_neighbors":
                AFFINITY_K,

            "metric":
                "euclidean",

            "mode":
                "connectivity",

            "symmetrization":
                "maximum_with_transpose",

            "binary":
                True
        },

    "spectral_clustering":
        {

            "K":
                K,

            "affinity":
                "precomputed",

            "assign_labels":
                "kmeans",

            "random_state":
                SPECTRAL_RANDOM_STATE,

            "n_init":
                SPECTRAL_N_INIT
        },

    "primary_agreement_metrics":
        [
            "ARI",
            "NMI"
        ],

    "summary_statistics":
        [
            "mean",
            "SD",
            "empirical_2.5_percentile",
            "empirical_97.5_percentile"
        ],

    "clinical_variables_used":
        False,

    "cluster_names_used":
        False,

    "input_hashes":
        {

            "partition":
                sha256_file(
                    PARTITION_FILE
                ),

            "partition_lock":
                sha256_file(
                    PARTITION_LOCK
                ),

            "Stage3C_lock":
                sha256_file(
                    STAGE3C_LOCK
                ),

            "primary_UMAP":
                sha256_file(
                    PRIMARY_UMAP
                ),

            "Stage3D_design":
                STAGE3D_EXPECTED[
                    "permutation_design"
                ],

            "Stage3D_output_lock":
                STAGE3D_EXPECTED[
                    "output_lock"
                ]
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
            "Existing Stage3E design differs. STOP."
        )


    print(
        "\nExisting Stage3E design lock verified."
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
        "\nStage3E design frozen BEFORE resampling."
    )


print(
    "Stage3E design SHA256:",
    sha256_file(
        DESIGN_LOCK
    )
)


# ============================================================
# 8. ONE RESAMPLING ITERATION
# ============================================================

def run_resample(
    indices,
    mode,
    iteration
):

    indices = np.asarray(
        indices,
        dtype=int
    )


    Y = Y_full[
        indices
    ]


    ref = reference_labels[
        indices
    ]


    A = build_affinity(
        Y
    )


    graph_info = graph_diagnostics(
        A
    )


    # No isolated nodes should occur under exact 4NN.
    if graph_info[
        "isolates"
    ] != 0:

        return {

            "mode":
                mode,

            "iteration":
                iteration,

            "status":
                "FAIL_ISOLATES",

            "draw_N":
                len(
                    indices
                ),

            "unique_subjects":
                len(
                    np.unique(
                        indices
                    )
                )
        }


    spectral = SpectralClustering(

        n_clusters=K,

        affinity="precomputed",

        assign_labels="kmeans",

        random_state=SPECTRAL_RANDOM_STATE,

        n_init=SPECTRAL_N_INIT
    )


    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )


            labels = spectral.fit_predict(
                A
            ).astype(
                int
            )


    except Exception as e:

        return {

            "mode":
                mode,

            "iteration":
                iteration,

            "status":
                "FAIL_SPECTRAL",

            "error":
                repr(
                    e
                ),

            "draw_N":
                len(
                    indices
                ),

            "unique_subjects":
                len(
                    np.unique(
                        indices
                    )
                )
        }


    if len(
        np.unique(
            labels
        )
    ) != K:

        return {

            "mode":
                mode,

            "iteration":
                iteration,

            "status":
                "FAIL_NOT_K6",

            "draw_N":
                len(
                    indices
                ),

            "unique_subjects":
                len(
                    np.unique(
                        indices
                    )
                ),

            "returned_clusters":
                int(
                    len(
                        np.unique(
                            labels
                        )
                    )
                )
        }


    ari = float(

        adjusted_rand_score(
            ref,
            labels
        )
    )


    nmi = float(

        normalized_mutual_info_score(
            ref,
            labels
        )
    )


    exact_match = bool(

        np.isclose(
            ari,
            1.0
        )

        and

        np.isclose(
            nmi,
            1.0
        )
    )


    (
        sizes,
        min_size,
        max_size,
        balance
    ) = cluster_size_stats(
        labels,
        K=K
    )


    reference_sample_sizes = np.array(

        [

            int(
                np.sum(
                    ref == k
                )
            )

            for k in range(
                K
            )
        ],

        dtype=int
    )


    # Geometry is calculated on the sampled fixed UMAP points.
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


    return {

        "mode":
            mode,

        "iteration":
            iteration,

        "status":
            "PASS",

        "draw_N":
            int(
                len(
                    indices
                )
            ),

        "unique_subjects":
            int(
                len(
                    np.unique(
                        indices
                    )
                )
            ),

        "reference_cluster0_N":
            int(
                reference_sample_sizes[
                    0
                ]
            ),

        "reference_cluster1_N":
            int(
                reference_sample_sizes[
                    1
                ]
            ),

        "reference_cluster2_N":
            int(
                reference_sample_sizes[
                    2
                ]
            ),

        "reference_cluster3_N":
            int(
                reference_sample_sizes[
                    3
                ]
            ),

        "reference_cluster4_N":
            int(
                reference_sample_sizes[
                    4
                ]
            ),

        "reference_cluster5_N":
            int(
                reference_sample_sizes[
                    5
                ]
            ),

        "reference_all_six_present":
            bool(
                np.all(
                    reference_sample_sizes
                    >
                    0
                )
            ),

        "cluster0_N":
            int(
                sizes[
                    0
                ]
            ),

        "cluster1_N":
            int(
                sizes[
                    1
                ]
            ),

        "cluster2_N":
            int(
                sizes[
                    2
                ]
            ),

        "cluster3_N":
            int(
                sizes[
                    3
                ]
            ),

        "cluster4_N":
            int(
                sizes[
                    4
                ]
            ),

        "cluster5_N":
            int(
                sizes[
                    5
                ]
            ),

        "minimum_cluster_size":
            min_size,

        "maximum_cluster_size":
            max_size,

        "balance_ratio":
            balance,

        "affinity_components":
            graph_info[
                "components"
            ],

        "affinity_component_sizes":
            "|".join(
                map(
                    str,
                    graph_info[
                        "component_sizes"
                    ]
                )
            ),

        "affinity_isolates":
            graph_info[
                "isolates"
            ],

        "affinity_edges":
            graph_info[
                "edges"
            ],

        "silhouette":
            sil,

        "Davies_Bouldin":
            db,

        "Calinski_Harabasz":
            ch,

        "ARI":
            ari,

        "NMI":
            nmi,

        "exact_partition_match":
            exact_match
    }


# ============================================================
# 9. GENERATE + SAVE EXACT RESAMPLING DRAWS
# ============================================================

sub_rng = np.random.default_rng(
    SUBSAMPLE_RNG_SEED
)

boot_rng = np.random.default_rng(
    BOOTSTRAP_RNG_SEED
)


draw_rows = []

subsample_draws = []

bootstrap_draws = []


for iteration in range(
    1,
    N_SUBSAMPLE + 1
):

    indices = sub_rng.choice(

        N,

        size=SUBSAMPLE_N,

        replace=False
    )


    subsample_draws.append(
        indices
    )


    for draw_position, subject_index in enumerate(
        indices
    ):

        draw_rows.append({

            "mode":
                "subsample_80pct",

            "iteration":
                iteration,

            "draw_position":
                draw_position,

            "subject_index":
                int(
                    subject_index
                ),

            "subject":
                partition.loc[
                    int(
                        subject_index
                    ),
                    "subject"
                ]
        })


for iteration in range(
    1,
    N_BOOTSTRAP + 1
):

    indices = boot_rng.choice(

        N,

        size=N,

        replace=True
    )


    bootstrap_draws.append(
        indices
    )


    for draw_position, subject_index in enumerate(
        indices
    ):

        draw_rows.append({

            "mode":
                "bootstrap",

            "iteration":
                iteration,

            "draw_position":
                draw_position,

            "subject_index":
                int(
                    subject_index
                ),

            "subject":
                partition.loc[
                    int(
                        subject_index
                    ),
                    "subject"
                ]
        })


DRAWS_FILE = (
    RESAMPLING_DIR
    / "STAGE3E_RESAMPLING_DRAWS.csv"
)


pd.DataFrame(
    draw_rows
).to_csv(
    DRAWS_FILE,
    index=False
)


print(
    "\nExact resampling draws saved BEFORE clustering."
)

print(
    "Draw file SHA256:",
    sha256_file(
        DRAWS_FILE
    )
)


# ============================================================
# 10. RUN 100 SUBSAMPLE ITERATIONS
# ============================================================

result_rows = []


print(
    "\n" + "=" * 120
)

print(
    "80% SUBJECT SUBSAMPLING — 100 ITERATIONS"
)

print(
    "=" * 120
)


for iteration, indices in enumerate(
    subsample_draws,
    start=1
):

    result = run_resample(

        indices=indices,

        mode="subsample_80pct",

        iteration=iteration
    )


    result_rows.append(
        result
    )


    if (
        iteration == 1
        or
        iteration % 10 == 0
        or
        iteration == 100
    ):

        if result[
            "status"
        ] == "PASS":

            print(

                f"Subsample {iteration:03d}/100 | "
                f"ARI={result['ARI']:.6f} | "
                f"NMI={result['NMI']:.6f} | "
                f"components={result['affinity_components']} | "
                f"min={result['minimum_cluster_size']} | "
                f"balance={result['balance_ratio']:.3f}"
            )

        else:

            print(

                f"Subsample {iteration:03d}/100 | "
                f"{result['status']}"
            )


# ============================================================
# 11. RUN 100 BOOTSTRAP ITERATIONS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "BOOTSTRAP WITH REPLACEMENT — 100 ITERATIONS"
)

print(
    "=" * 120
)


for iteration, indices in enumerate(
    bootstrap_draws,
    start=1
):

    result = run_resample(

        indices=indices,

        mode="bootstrap",

        iteration=iteration
    )


    result_rows.append(
        result
    )


    if (
        iteration == 1
        or
        iteration % 10 == 0
        or
        iteration == 100
    ):

        if result[
            "status"
        ] == "PASS":

            print(

                f"Bootstrap {iteration:03d}/100 | "
                f"ARI={result['ARI']:.6f} | "
                f"NMI={result['NMI']:.6f} | "
                f"unique={result['unique_subjects']} | "
                f"components={result['affinity_components']} | "
                f"min={result['minimum_cluster_size']} | "
                f"balance={result['balance_ratio']:.3f}"
            )

        else:

            print(

                f"Bootstrap {iteration:03d}/100 | "
                f"{result['status']}"
            )


# ============================================================
# 12. SAVE ITERATION-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(
    result_rows
)


RESULTS_FILE = (
    AUDIT
    / "STAGE3E_200_RESAMPLING_ITERATION_RESULTS.csv"
)


results.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 13. COMPUTE MODE-SPECIFIC SUMMARIES
# ============================================================

summary_rows = []


for mode in [
    "subsample_80pct",
    "bootstrap"
]:

    mode_all = results[
        results[
            "mode"
        ]
        ==
        mode
    ].copy()


    mode_pass = mode_all[
        mode_all[
            "status"
        ]
        ==
        "PASS"
    ].copy()


    n_total = len(
        mode_all
    )


    n_pass = len(
        mode_pass
    )


    if n_pass == 0:

        raise RuntimeError(
            f"No successful runs for {mode}. STOP."
        )


    ari_low, ari_high = empirical_ci(
        mode_pass[
            "ARI"
        ]
    )


    nmi_low, nmi_high = empirical_ci(
        mode_pass[
            "NMI"
        ]
    )


    summary_rows.append({

        "mode":
            mode,

        "iterations":
            n_total,

        "successful_iterations":
            n_pass,

        "failed_iterations":
            int(
                n_total
                -
                n_pass
            ),

        "mean_unique_subjects":
            float(
                mode_pass[
                    "unique_subjects"
                ].mean()
            ),

        "SD_unique_subjects":
            float(
                mode_pass[
                    "unique_subjects"
                ].std(
                    ddof=0
                )
            ),

        "mean_ARI":
            float(
                mode_pass[
                    "ARI"
                ].mean()
            ),

        "SD_ARI":
            float(
                mode_pass[
                    "ARI"
                ].std(
                    ddof=0
                )
            ),

        "ARI_2.5_percentile":
            ari_low,

        "ARI_97.5_percentile":
            ari_high,

        "minimum_ARI":
            float(
                mode_pass[
                    "ARI"
                ].min()
            ),

        "maximum_ARI":
            float(
                mode_pass[
                    "ARI"
                ].max()
            ),

        "mean_NMI":
            float(
                mode_pass[
                    "NMI"
                ].mean()
            ),

        "SD_NMI":
            float(
                mode_pass[
                    "NMI"
                ].std(
                    ddof=0
                )
            ),

        "NMI_2.5_percentile":
            nmi_low,

        "NMI_97.5_percentile":
            nmi_high,

        "minimum_NMI":
            float(
                mode_pass[
                    "NMI"
                ].min()
            ),

        "maximum_NMI":
            float(
                mode_pass[
                    "NMI"
                ].max()
            ),

        "exact_partition_matches":
            int(
                mode_pass[
                    "exact_partition_match"
                ].sum()
            ),

        "mean_silhouette":
            float(
                mode_pass[
                    "silhouette"
                ].mean()
            ),

        "SD_silhouette":
            float(
                mode_pass[
                    "silhouette"
                ].std(
                    ddof=0
                )
            ),

        "mean_Davies_Bouldin":
            float(
                mode_pass[
                    "Davies_Bouldin"
                ].mean()
            ),

        "mean_Calinski_Harabasz":
            float(
                mode_pass[
                    "Calinski_Harabasz"
                ].mean()
            ),

        "mean_minimum_cluster_size":
            float(
                mode_pass[
                    "minimum_cluster_size"
                ].mean()
            ),

        "mean_balance_ratio":
            float(
                mode_pass[
                    "balance_ratio"
                ].mean()
            ),

        "disconnected_affinity_runs":
            int(
                np.sum(
                    mode_pass[
                        "affinity_components"
                    ]
                    >
                    1
                )
            ),

        "all_reference_clusters_present_runs":
            int(
                mode_pass[
                    "reference_all_six_present"
                ].sum()
            )
    })


summary_table = pd.DataFrame(
    summary_rows
)


SUMMARY_TABLE_FILE = (
    AUDIT
    / "STAGE3E_RESAMPLING_SUMMARY_TABLE.csv"
)


summary_table.to_csv(
    SUMMARY_TABLE_FILE,
    index=False
)


# ============================================================
# 14. JSON SUMMARY
# ============================================================

sub_summary = summary_table[

    summary_table[
        "mode"
    ]
    ==
    "subsample_80pct"

].iloc[
    0
]


boot_summary = summary_table[

    summary_table[
        "mode"
    ]
    ==
    "bootstrap"

].iloc[
    0
]


SUMMARY_JSON = (
    AUDIT
    / "STAGE3E_RESAMPLING_SUMMARY.json"
)


summary_json = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3E_SUBJECT_RESAMPLING_ROBUSTNESS",

    "status":
        "COMPUTATIONAL_PASS",

    "N":
        85,

    "K":
        6,

    "K_changed":
        False,

    "input_manifold":
        "frozen_seed42_UMAP",

    "subsampling":
        {

            "iterations":
                100,

            "draw_N":
                68,

            "mean_ARI":
                float(
                    sub_summary[
                        "mean_ARI"
                    ]
                ),

            "SD_ARI":
                float(
                    sub_summary[
                        "SD_ARI"
                    ]
                ),

            "ARI_2.5_percentile":
                float(
                    sub_summary[
                        "ARI_2.5_percentile"
                    ]
                ),

            "ARI_97.5_percentile":
                float(
                    sub_summary[
                        "ARI_97.5_percentile"
                    ]
                ),

            "mean_NMI":
                float(
                    sub_summary[
                        "mean_NMI"
                    ]
                ),

            "SD_NMI":
                float(
                    sub_summary[
                        "SD_NMI"
                    ]
                ),

            "NMI_2.5_percentile":
                float(
                    sub_summary[
                        "NMI_2.5_percentile"
                    ]
                ),

            "NMI_97.5_percentile":
                float(
                    sub_summary[
                        "NMI_97.5_percentile"
                    ]
                ),

            "exact_matches":
                int(
                    sub_summary[
                        "exact_partition_matches"
                    ]
                ),

            "disconnected_affinity_runs":
                int(
                    sub_summary[
                        "disconnected_affinity_runs"
                    ]
                )
        },

    "bootstrap":
        {

            "iterations":
                100,

            "draw_N":
                85,

            "mean_unique_subjects":
                float(
                    boot_summary[
                        "mean_unique_subjects"
                    ]
                ),

            "mean_ARI":
                float(
                    boot_summary[
                        "mean_ARI"
                    ]
                ),

            "SD_ARI":
                float(
                    boot_summary[
                        "SD_ARI"
                    ]
                ),

            "ARI_2.5_percentile":
                float(
                    boot_summary[
                        "ARI_2.5_percentile"
                    ]
                ),

            "ARI_97.5_percentile":
                float(
                    boot_summary[
                        "ARI_97.5_percentile"
                    ]
                ),

            "mean_NMI":
                float(
                    boot_summary[
                        "mean_NMI"
                    ]
                ),

            "SD_NMI":
                float(
                    boot_summary[
                        "SD_NMI"
                    ]
                ),

            "NMI_2.5_percentile":
                float(
                    boot_summary[
                        "NMI_2.5_percentile"
                    ]
                ),

            "NMI_97.5_percentile":
                float(
                    boot_summary[
                        "NMI_97.5_percentile"
                    ]
                ),

            "exact_matches":
                int(
                    boot_summary[
                        "exact_partition_matches"
                    ]
                ),

            "disconnected_affinity_runs":
                int(
                    boot_summary[
                        "disconnected_affinity_runs"
                    ]
                )
        },

    "clinical_variables_used":
        False,

    "cluster_names_assigned":
        False,

    "interpretation":
        (
            "This stage evaluates downstream subject-composition "
            "robustness of the already-frozen seed42 manifold and "
            "K=6 partitioning procedure. It does not constitute "
            "clinical validation and does not permit K reselection."
        )
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=2
    )


# ============================================================
# 15. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE3E_RESAMPLING_REPORT.txt"
)


report = f"""
TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
STAGE 3E — SUBJECT-RESAMPLING ROBUSTNESS

AUTHORITATIVE ANALYSIS
----------------------
N:
85

K:
6

Input:
Frozen primary seed42 UMAP representation

UMAP refitted during resampling:
NO

VGAE refitted during resampling:
NO

Fusion refitted during resampling:
NO


80% SUBSAMPLING
---------------
Iterations:
100

Subjects per iteration:
68

Mean ARI:
{float(sub_summary["mean_ARI"]):.10f}

SD ARI:
{float(sub_summary["SD_ARI"]):.10f}

Empirical ARI 95% interval:
[{float(sub_summary["ARI_2.5_percentile"]):.10f},
 {float(sub_summary["ARI_97.5_percentile"]):.10f}]

Mean NMI:
{float(sub_summary["mean_NMI"]):.10f}

SD NMI:
{float(sub_summary["SD_NMI"]):.10f}

Empirical NMI 95% interval:
[{float(sub_summary["NMI_2.5_percentile"]):.10f},
 {float(sub_summary["NMI_97.5_percentile"]):.10f}]

Exact matches:
{int(sub_summary["exact_partition_matches"])} / 100

Disconnected sampled affinities:
{int(sub_summary["disconnected_affinity_runs"])} / 100


BOOTSTRAP
---------
Iterations:
100

Draws per iteration:
85

Mean unique subjects:
{float(boot_summary["mean_unique_subjects"]):.10f}

Mean ARI:
{float(boot_summary["mean_ARI"]):.10f}

SD ARI:
{float(boot_summary["SD_ARI"]):.10f}

Empirical ARI 95% interval:
[{float(boot_summary["ARI_2.5_percentile"]):.10f},
 {float(boot_summary["ARI_97.5_percentile"]):.10f}]

Mean NMI:
{float(boot_summary["mean_NMI"]):.10f}

SD NMI:
{float(boot_summary["SD_NMI"]):.10f}

Empirical NMI 95% interval:
[{float(boot_summary["NMI_2.5_percentile"]):.10f},
 {float(boot_summary["NMI_97.5_percentile"]):.10f}]

Exact matches:
{int(boot_summary["exact_partition_matches"])} / 100

Disconnected sampled affinities:
{int(boot_summary["disconnected_affinity_runs"])} / 100


INTERPRETATION
--------------
These analyses quantify subject-composition robustness of
the frozen final manifold and K=6 spectral partitioning stage.

They do not retrain the EEG representation, phenotype view,
fused subject graph, VGAE, or UMAP.

ARI and NMI quantify agreement with the authoritative
partition restricted to the corresponding sampled observations.

No clinical variables were used.

K remains permanently frozen at 6.

Poorer bootstrap or subsampling stability, if observed,
must be reported rather than corrected by changing K.


FINAL STATUS
------------
STAGE 3E COMPUTATIONAL PASS —
SUBJECT-RESAMPLING ROBUSTNESS COMPLETE.
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
# 16. OUTPUT LOCK
# ============================================================

OUTPUT_LOCK = (
    LOCK
    / "STAGE3E_RESAMPLING_OUTPUT_LOCK.json"
)


output_lock = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3E_RESAMPLING_OUTPUT_LOCK",

    "status":
        "FROZEN",

    "N":
        85,

    "K":
        6,

    "K_remains_frozen":
        True,

    "clinical_variables_used":
        False,

    "subsampling_mean_ARI":
        float(
            sub_summary[
                "mean_ARI"
            ]
        ),

    "subsampling_mean_NMI":
        float(
            sub_summary[
                "mean_NMI"
            ]
        ),

    "bootstrap_mean_ARI":
        float(
            boot_summary[
                "mean_ARI"
            ]
        ),

    "bootstrap_mean_NMI":
        float(
            boot_summary[
                "mean_NMI"
            ]
        ),

    "input_hashes":
        {

            "partition":
                sha256_file(
                    PARTITION_FILE
                ),

            "Stage3C_lock":
                sha256_file(
                    STAGE3C_LOCK
                ),

            "primary_UMAP":
                sha256_file(
                    PRIMARY_UMAP
                ),

            "Stage3D_output_lock":
                STAGE3D_EXPECTED[
                    "output_lock"
                ],

            "resampling_design":
                sha256_file(
                    DESIGN_LOCK
                ),

            "resampling_draws":
                sha256_file(
                    DRAWS_FILE
                )
        },

    "output_hashes":
        {

            "iteration_results":
                sha256_file(
                    RESULTS_FILE
                ),

            "summary_table":
                sha256_file(
                    SUMMARY_TABLE_FILE
                ),

            "summary_json":
                sha256_file(
                    SUMMARY_JSON
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


    if existing != output_lock:

        raise RuntimeError(
            "Existing Stage3E output lock differs. STOP."
        )


    print(
        "\nExisting Stage3E output lock verified."
    )


else:

    with open(
        OUTPUT_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_lock,
            f,
            indent=2
        )


    print(
        "\nStage3E resampling output lock created."
    )


# ============================================================
# 17. PRINT FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 3E — RESAMPLING SUMMARY"
)

print(
    "=" * 120
)


print(

    summary_table.to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"
    )
)


print(
    "\n" + "=" * 120
)

print(
    "PRIMARY RESAMPLING RESULTS"
)

print(
    "=" * 120
)


print(
    "\n80% SUBSAMPLING:"
)

print(
    "Mean ARI =",
    f"{float(sub_summary['mean_ARI']):.6f}"
)

print(
    "SD ARI =",
    f"{float(sub_summary['SD_ARI']):.6f}"
)

print(
    "Mean NMI =",
    f"{float(sub_summary['mean_NMI']):.6f}"
)

print(
    "SD NMI =",
    f"{float(sub_summary['SD_NMI']):.6f}"
)

print(
    "Exact matches =",
    int(
        sub_summary[
            "exact_partition_matches"
        ]
    ),
    "/ 100"
)


print(
    "\nBOOTSTRAP:"
)

print(
    "Mean unique subjects =",
    f"{float(boot_summary['mean_unique_subjects']):.2f}"
)

print(
    "Mean ARI =",
    f"{float(boot_summary['mean_ARI']):.6f}"
)

print(
    "SD ARI =",
    f"{float(boot_summary['SD_ARI']):.6f}"
)

print(
    "Mean NMI =",
    f"{float(boot_summary['mean_NMI']):.6f}"
)

print(
    "SD NMI =",
    f"{float(boot_summary['SD_NMI']):.6f}"
)

print(
    "Exact matches =",
    int(
        boot_summary[
            "exact_partition_matches"
        ]
    ),
    "/ 100"
)


# ============================================================
# 18. HASHES
# ============================================================

print(
    "\nHASHES"
)


print(
    "Stage3E design:",
    sha256_file(
        DESIGN_LOCK
    )
)

print(
    "Exact resampling draws:",
    sha256_file(
        DRAWS_FILE
    )
)

print(
    "Iteration results:",
    sha256_file(
        RESULTS_FILE
    )
)

print(
    "Summary table:",
    sha256_file(
        SUMMARY_TABLE_FILE
    )
)

print(
    "Summary JSON:",
    sha256_file(
        SUMMARY_JSON
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
# 19. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 3E PASS — "
    "SUBJECT-RESAMPLING ROBUSTNESS COMPLETE"
)

print(
    "=" * 120
)


print(
    "\nK REMAINS FROZEN AT 6."
)

print(
    "NO MODEL HYPERPARAMETER WAS RETUNED."
)

print(
    "NO CLINICAL VARIABLES WERE USED."
)

print(
    "NO CLUSTER NAMES WERE ASSIGNED."
)


print(
    "\nIMPORTANT:"
)

print(
    "This stage evaluates robustness to subject composition "
    "on the already-frozen seed42 manifold."
)

print(
    "It is not clinical validation."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 4A FOLLOWS THE FROZEN STRUCTURAL ANALYSES "
    "AND PREPARES THE HELD-OUT CHARACTERIZATION PLAN."
)