# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 3D
# FIXED-LABEL PERMUTATION SIGNIFICANCE
# OF THE AUTHORITATIVE K=6 INTERNAL GEOMETRY
#
# AUTHORITATIVE:
#   N = 85
#   K = 6
#   UMAP seed = 42
#   sizes = [27, 15, 18, 10, 9, 6]
#
# PERMUTATION TEST
# ----------------
# B = 1000
#
# Permute the authoritative labels across the fixed
# seed42 UMAP coordinates.
#
# Because labels are permuted rather than regenerated,
# every permutation preserves:
#   - N = 85
#   - K = 6
#   - exact cluster sizes
#
# FAVORABLE DIRECTIONS
# --------------------
# Silhouette:          higher
# Davies-Bouldin:      lower
# Calinski-Harabasz:   higher
#
# P-VALUE
# -------
# p = (1 + number at least as extreme) / (B + 1)
#
# IMPORTANT
# ---------
# This tests whether the frozen partition has stronger
# internal geometry than random assignments with the
# same cluster sizes.
#
# It is NOT:
#   clinical validation
#   diagnostic validation
#   external validation
#   evidence that K=6 is a uniquely true subtype count
#
# NO CLINICAL VARIABLES ARE OPENED.
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
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

STAGE3_DIR = (
    EXP3
    / "05_UMAP_K_SELECTION"
)

PERM_DIR = (
    EXP3
    / "07_PERMUTATION"
)


PERM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 3D — FIXED-LABEL PERMUTATION SIGNIFICANCE")
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

STAGE3B_SUMMARY = (
    AUDIT
    / "STAGE3B_PRIMARY_PARTITION_SUMMARY.json"
)

STAGE3C_DESIGN = (
    LOCK
    / "STAGE3C_20SEED_ROBUSTNESS_DESIGN_LOCK.json"
)

STAGE3C_RESULTS = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_RESULTS.csv"
)

STAGE3C_PAIRWISE = (
    AUDIT
    / "STAGE3C_190_PAIRWISE_SEED_STABILITY.csv"
)

STAGE3C_CLUSTER_SUMMARY = (
    AUDIT
    / "STAGE3C_CLUSTER_SPECIFIC_STABILITY_SUMMARY.csv"
)

STAGE3C_SUMMARY = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_SUMMARY.json"
)

STAGE3C_REPORT = (
    AUDIT
    / "STAGE3C_20SEED_ROBUSTNESS_REPORT.txt"
)

STAGE3C_LOCK = (
    LOCK
    / "STAGE3C_20SEED_ROBUSTNESS_LOCK.json"
)

PRIMARY_UMAP = (
    STAGE3_DIR
    / "STAGE3A_UMAP_SEED42.npy"
)


# ============================================================
# 4. VERIFY KNOWN FROZEN HASHES
# ============================================================

EXPECTED_HASHES = {

    PARTITION_FILE:
        "f11a53d12e6676c7175622edef55984686363af051f4158b899fc321df50a782",

    PARTITION_LOCK:
        "53006cf5d1c66e15fc64b5dd7b707a6c90e1bde9315ef45cda0ae0ab747e8e85",

    STAGE3B_SUMMARY:
        "5173863ad1816f44ff704af1a18aa2ffe14395038b165c0b117b87592a2fbbf4",

    STAGE3C_DESIGN:
        "ac0832fa037f036072286554d9161cf0ac9896ba95a20a51158f8246699ac7b2",

    STAGE3C_RESULTS:
        "58fb0dcc70f4cd5c5784652fe46c05ba3dc475a14240a423d713ecdaa1852ee9",

    STAGE3C_PAIRWISE:
        "198109115a09f6bdaf2db3f4daf8c530600e3f5e33e7828b5a52028ae174eaae",

    STAGE3C_CLUSTER_SUMMARY:
        "ddc2a925b80d8f7a54dc80a8e6d527691c3e5755a29ccd0b9c711be6d60a6ec8",

    STAGE3C_SUMMARY:
        "545f604c12d9eb1affeb7af2b2a38cdbefe5b71fcfa70b82f81ffef7845a81f7",

    STAGE3C_REPORT:
        "de5e4bb4a58a01aa565958fead3b3d8ec68bc3dd22364677c4b295fc3f3cd74a",

    STAGE3C_LOCK:
        "fe34076b64b7dd59896a2ba977e3a60be44fe7a4845a2d1e8ce538902313476e"
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
            "Frozen structural input changed. STOP."
        )


if not PRIMARY_UMAP.exists():

    raise FileNotFoundError(
        str(
            PRIMARY_UMAP
        )
    )


# ============================================================
# 5. VERIFY PRIMARY UMAP USING STAGE3B LOCK
#
# We do not guess its hash.
# The authoritative Stage3B lock already recorded it.
# ============================================================

with open(
    PARTITION_LOCK,
    "r",
    encoding="utf-8"
) as f:

    partition_lock = json.load(
        f
    )


expected_umap_hash = partition_lock[
    "input_hashes"
][
    "Stage3A_seed42_UMAP"
]


actual_umap_hash = sha256_file(
    PRIMARY_UMAP
)


print(
    "\nPrimary seed42 UMAP hash:",
    "PASS"
    if actual_umap_hash == expected_umap_hash
    else
    "FAIL"
)


if actual_umap_hash != expected_umap_hash:

    print(
        "Expected:",
        expected_umap_hash
    )

    print(
        "Actual:",
        actual_umap_hash
    )


    raise RuntimeError(
        "Primary seed42 UMAP changed. STOP."
    )


print(
    "\nALL FROZEN INPUT HASHES PASS."
)


# ============================================================
# 6. VERIFY FROZEN PARTITION STATE
# ============================================================

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
    "cluster_sizes"
] == [
    27,
    15,
    18,
    10,
    9,
    6
]


assert partition_lock[
    "diagnosis_used"
] is False


assert partition_lock[
    "ADHD_YBOCS_BDI_used"
] is False


with open(
    STAGE3C_LOCK,
    "r",
    encoding="utf-8"
) as f:

    robustness_lock = json.load(
        f
    )


assert robustness_lock[
    "status"
] == "FROZEN"


assert robustness_lock[
    "K"
] == 6


assert robustness_lock[
    "K_remains_frozen"
] is True


print(
    "\nAuthoritative structural state verified:"
)

print(
    "N = 85"
)

print(
    "K = 6"
)

print(
    "Primary UMAP seed = 42"
)

print(
    "K reselection allowed = NO"
)


# ============================================================
# 7. LOAD FIXED UMAP + AUTHORITATIVE LABELS
# ============================================================

Y = np.load(
    PRIMARY_UMAP
).astype(
    np.float64
)


partition = pd.read_csv(
    PARTITION_FILE
)


assert Y.shape == (
    85,
    2
)


assert np.isfinite(
    Y
).all()


assert len(
    partition
) == 85


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


labels = partition[
    "EEG_PHENO_cluster"
].to_numpy(
    dtype=int
)


assert len(
    labels
) == 85


assert set(
    np.unique(
        labels
    )
) == {
    0,
    1,
    2,
    3,
    4,
    5
}


sizes = np.bincount(
    labels,
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


print(
    "\nFixed UMAP shape:",
    Y.shape
)

print(
    "Authoritative cluster sizes:",
    sizes.tolist()
)


# ============================================================
# 8. OBSERVED INTERNAL GEOMETRY
# ============================================================

observed_silhouette = float(

    silhouette_score(
        Y,
        labels,
        metric="euclidean"
    )
)


observed_db = float(

    davies_bouldin_score(
        Y,
        labels
    )
)


observed_ch = float(

    calinski_harabasz_score(
        Y,
        labels
    )
)


# Match Stage3B values.

assert np.isclose(

    observed_silhouette,

    0.4654047984,

    rtol=1e-6,
    atol=1e-6
)


assert np.isclose(

    observed_db,

    0.5983619615,

    rtol=1e-6,
    atol=1e-6
)


assert np.isclose(

    observed_ch,

    163.4585647621,

    rtol=1e-6,
    atol=1e-5
)


print(
    "\n" + "=" * 120
)

print(
    "OBSERVED AUTHORITATIVE GEOMETRY"
)

print(
    "=" * 120
)


print(
    "Silhouette:",
    f"{observed_silhouette:.10f}"
)

print(
    "Davies-Bouldin:",
    f"{observed_db:.10f}"
)

print(
    "Calinski-Harabasz:",
    f"{observed_ch:.10f}"
)


# ============================================================
# 9. FREEZE PERMUTATION DESIGN BEFORE RUNNING
# ============================================================

B = 1000

PERMUTATION_SEED = 42


PERM_DESIGN_LOCK = (
    LOCK
    / "STAGE3D_PERMUTATION_DESIGN_LOCK.json"
)


perm_design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3D_FIXED_LABEL_PERMUTATION",

    "status":
        "FROZEN_BEFORE_PERMUTATIONS",

    "N":
        85,

    "K":
        6,

    "cluster_sizes":
        [
            27,
            15,
            18,
            10,
            9,
            6
        ],

    "coordinate_space":
        "authoritative_seed42_UMAP_2D",

    "coordinates_refit_per_permutation":
        False,

    "partition_refit_per_permutation":
        False,

    "permutation_method":
        (
            "randomly_permute_authoritative_labels_across_fixed_"
            "UMAP_coordinates_preserving_exact_cluster_sizes"
        ),

    "B":
        B,

    "permutation_seed":
        PERMUTATION_SEED,

    "metrics":
        {

            "silhouette":
                "higher_is_more_extreme",

            "Davies_Bouldin":
                "lower_is_more_extreme",

            "Calinski_Harabasz":
                "higher_is_more_extreme"
        },

    "p_value_formula":
        "(1 + number_at_least_as_extreme) / (B + 1)",

    "minimum_attainable_p":
        1.0
        /
        (
            B
            +
            1
        ),

    "diagnosis_used":
        False,

    "clinical_scores_used":
        False,

    "phenotype_profile_tested":
        False,

    "K_reselected":
        False,

    "interpretation":
        (
            "Tests internal geometry against random label assignments "
            "of identical cluster sizes. Does not establish clinical, "
            "biological, diagnostic, or external validity."
        ),

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

            "Stage3C_lock":
                sha256_file(
                    STAGE3C_LOCK
                ),

            "primary_seed42_UMAP":
                sha256_file(
                    PRIMARY_UMAP
                )
        }
}


if PERM_DESIGN_LOCK.exists():

    with open(
        PERM_DESIGN_LOCK,
        "r",
        encoding="utf-8"
    ) as f:

        existing = json.load(
            f
        )


    if existing != perm_design:

        raise RuntimeError(
            "Existing Stage3D permutation design differs. STOP."
        )


    print(
        "\nExisting Stage3D permutation design lock verified."
    )


else:

    with open(
        PERM_DESIGN_LOCK,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            perm_design,
            f,
            indent=2
        )


    print(
        "\nStage3D permutation design frozen BEFORE permutations."
    )


print(
    "Permutation-design SHA256:",
    sha256_file(
        PERM_DESIGN_LOCK
    )
)


# ============================================================
# 10. RUN 1000 LABEL PERMUTATIONS
# ============================================================

rng = np.random.default_rng(
    PERMUTATION_SEED
)


null_silhouette = np.empty(
    B,
    dtype=np.float64
)

null_db = np.empty(
    B,
    dtype=np.float64
)

null_ch = np.empty(
    B,
    dtype=np.float64
)


print(
    "\n" + "=" * 120
)

print(
    "RUNNING 1000 FIXED-LABEL PERMUTATIONS"
)

print(
    "=" * 120
)


for b in range(
    B
):

    permuted_labels = rng.permutation(
        labels
    )


    # --------------------------------------------------------
    # Cluster-size preservation must be exact.
    # --------------------------------------------------------

    perm_sizes = np.bincount(
        permuted_labels,
        minlength=6
    )


    if not np.array_equal(
        perm_sizes,
        sizes
    ):

        raise RuntimeError(
            "Permutation failed to preserve cluster sizes."
        )


    null_silhouette[
        b
    ] = silhouette_score(

        Y,
        permuted_labels,
        metric="euclidean"
    )


    null_db[
        b
    ] = davies_bouldin_score(

        Y,
        permuted_labels
    )


    null_ch[
        b
    ] = calinski_harabasz_score(

        Y,
        permuted_labels
    )


    if (
        b == 0
        or
        (
            b
            +
            1
        )
        %
        100
        ==
        0
    ):

        print(
            f"Permutation {b + 1:04d}/{B}"
        )


# ============================================================
# 11. FINITE-CHECK
# ============================================================

if not np.isfinite(
    null_silhouette
).all():

    raise RuntimeError(
        "Non-finite silhouette null result detected."
    )


if not np.isfinite(
    null_db
).all():

    raise RuntimeError(
        "Non-finite DB null result detected."
    )


if not np.isfinite(
    null_ch
).all():

    raise RuntimeError(
        "Non-finite CH null result detected."
    )


# ============================================================
# 12. ONE-SIDED PERMUTATION P-VALUES
# ============================================================

sil_extreme_count = int(

    np.sum(
        null_silhouette
        >=
        observed_silhouette
    )
)


db_extreme_count = int(

    np.sum(
        null_db
        <=
        observed_db
    )
)


ch_extreme_count = int(

    np.sum(
        null_ch
        >=
        observed_ch
    )
)


p_silhouette = (

    1
    +
    sil_extreme_count

) / (

    B
    +
    1
)


p_db = (

    1
    +
    db_extreme_count

) / (

    B
    +
    1
)


p_ch = (

    1
    +
    ch_extreme_count

) / (

    B
    +
    1
)


# ============================================================
# 13. NULL DISTRIBUTION SUMMARIES
# ============================================================

def summarize_null(
    observed,
    null_values,
    favorable_direction
):

    mean = float(
        np.mean(
            null_values
        )
    )


    sd = float(
        np.std(
            null_values,
            ddof=0
        )
    )


    if sd > 0:

        raw_z = float(
            (
                observed
                -
                mean
            )
            /
            sd
        )

    else:

        raw_z = np.nan


    if favorable_direction == "higher":

        favorable_z = raw_z

    elif favorable_direction == "lower":

        favorable_z = -raw_z

    else:

        raise ValueError(
            favorable_direction
        )


    return {

        "observed":
            float(
                observed
            ),

        "null_mean":
            mean,

        "null_SD":
            sd,

        "null_min":
            float(
                np.min(
                    null_values
                )
            ),

        "null_max":
            float(
                np.max(
                    null_values
                )
            ),

        "null_q025":
            float(
                np.quantile(
                    null_values,
                    0.025
                )
            ),

        "null_q975":
            float(
                np.quantile(
                    null_values,
                    0.975
                )
            ),

        "raw_z_observed_minus_null_mean":
            raw_z,

        "favorable_direction_z":
            favorable_z
    }


sil_summary = summarize_null(

    observed_silhouette,
    null_silhouette,
    "higher"
)


db_summary = summarize_null(

    observed_db,
    null_db,
    "lower"
)


ch_summary = summarize_null(

    observed_ch,
    null_ch,
    "higher"
)


# ============================================================
# 14. SAVE ALL 1000 PERMUTATIONS
# ============================================================

perm_df = pd.DataFrame({

    "permutation":
        np.arange(
            1,
            B + 1
        ),

    "silhouette":
        null_silhouette,

    "Davies_Bouldin":
        null_db,

    "Calinski_Harabasz":
        null_ch
})


PERMUTATIONS_FILE = (
    PERM_DIR
    / "STAGE3D_1000_FIXED_LABEL_PERMUTATIONS.csv"
)


perm_df.to_csv(
    PERMUTATIONS_FILE,
    index=False
)


# ============================================================
# 15. RESULT TABLE
# ============================================================

result_table = pd.DataFrame([

    {

        "metric":
            "silhouette",

        "favorable_direction":
            "higher",

        "observed":
            observed_silhouette,

        "null_mean":
            sil_summary[
                "null_mean"
            ],

        "null_SD":
            sil_summary[
                "null_SD"
            ],

        "null_2.5_percentile":
            sil_summary[
                "null_q025"
            ],

        "null_97.5_percentile":
            sil_summary[
                "null_q975"
            ],

        "favorable_z":
            sil_summary[
                "favorable_direction_z"
            ],

        "extreme_permutations":
            sil_extreme_count,

        "B":
            B,

        "permutation_p":
            p_silhouette
    },


    {

        "metric":
            "Davies_Bouldin",

        "favorable_direction":
            "lower",

        "observed":
            observed_db,

        "null_mean":
            db_summary[
                "null_mean"
            ],

        "null_SD":
            db_summary[
                "null_SD"
            ],

        "null_2.5_percentile":
            db_summary[
                "null_q025"
            ],

        "null_97.5_percentile":
            db_summary[
                "null_q975"
            ],

        "favorable_z":
            db_summary[
                "favorable_direction_z"
            ],

        "extreme_permutations":
            db_extreme_count,

        "B":
            B,

        "permutation_p":
            p_db
    },


    {

        "metric":
            "Calinski_Harabasz",

        "favorable_direction":
            "higher",

        "observed":
            observed_ch,

        "null_mean":
            ch_summary[
                "null_mean"
            ],

        "null_SD":
            ch_summary[
                "null_SD"
            ],

        "null_2.5_percentile":
            ch_summary[
                "null_q025"
            ],

        "null_97.5_percentile":
            ch_summary[
                "null_q975"
            ],

        "favorable_z":
            ch_summary[
                "favorable_direction_z"
            ],

        "extreme_permutations":
            ch_extreme_count,

        "B":
            B,

        "permutation_p":
            p_ch
    }
])


RESULT_FILE = (
    AUDIT
    / "STAGE3D_PERMUTATION_SIGNIFICANCE_RESULTS.csv"
)


result_table.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 16. SUMMARY JSON
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE3D_PERMUTATION_SIGNIFICANCE_SUMMARY.json"
)


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3D_FIXED_LABEL_PERMUTATION_SIGNIFICANCE",

    "status":
        "COMPUTATIONAL_PASS",

    "N":
        85,

    "K":
        6,

    "cluster_sizes":
        [
            27,
            15,
            18,
            10,
            9,
            6
        ],

    "B":
        B,

    "permutation_seed":
        PERMUTATION_SEED,

    "minimum_attainable_p":
        1.0
        /
        (
            B
            +
            1
        ),

    "observed":
        {

            "silhouette":
                observed_silhouette,

            "Davies_Bouldin":
                observed_db,

            "Calinski_Harabasz":
                observed_ch
        },

    "permutation_p":
        {

            "silhouette":
                p_silhouette,

            "Davies_Bouldin":
                p_db,

            "Calinski_Harabasz":
                p_ch
        },

    "extreme_permutation_counts":
        {

            "silhouette":
                sil_extreme_count,

            "Davies_Bouldin":
                db_extreme_count,

            "Calinski_Harabasz":
                ch_extreme_count
        },

    "null_distribution":
        {

            "silhouette":
                sil_summary,

            "Davies_Bouldin":
                db_summary,

            "Calinski_Harabasz":
                ch_summary
        },

    "interpretation":
        (
            "The test evaluates whether the authoritative "
            "K=6 partition has stronger internal geometry "
            "than random label assignments preserving the "
            "same six cluster sizes. It does not establish "
            "clinical, diagnostic, biological, or external validity."
        ),

    "clinical_variables_used":
        False,

    "K_changed":
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
# 17. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE3D_PERMUTATION_SIGNIFICANCE_REPORT.txt"
)


report = f"""
TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
STAGE 3D — FIXED-LABEL PERMUTATION SIGNIFICANCE

AUTHORITATIVE PARTITION
-----------------------
N:
85

K:
6

Cluster sizes:
[27, 15, 18, 10, 9, 6]

Primary UMAP seed:
42


PERMUTATION DESIGN
------------------
Permutations:
{B}

Permutation RNG seed:
{PERMUTATION_SEED}

Null construction:
Authoritative K=6 labels randomly permuted over the
fixed seed42 UMAP coordinates.

Exact cluster sizes are preserved in every permutation.

Minimum attainable p-value:
{1.0 / (B + 1):.10f}


OBSERVED INTERNAL GEOMETRY
--------------------------
Silhouette:
{observed_silhouette:.10f}

Davies-Bouldin:
{observed_db:.10f}

Calinski-Harabasz:
{observed_ch:.10f}


SILHOUETTE NULL
---------------
Null mean:
{sil_summary["null_mean"]:.10f}

Null SD:
{sil_summary["null_SD"]:.10f}

Extreme permutations:
{sil_extreme_count} / {B}

Permutation p:
{p_silhouette:.10f}


DAVIES-BOULDIN NULL
-------------------
Null mean:
{db_summary["null_mean"]:.10f}

Null SD:
{db_summary["null_SD"]:.10f}

Extreme permutations:
{db_extreme_count} / {B}

Permutation p:
{p_db:.10f}


CALINSKI-HARABASZ NULL
----------------------
Null mean:
{ch_summary["null_mean"]:.10f}

Null SD:
{ch_summary["null_SD"]:.10f}

Extreme permutations:
{ch_extreme_count} / {B}

Permutation p:
{p_ch:.10f}


INTERPRETATION
--------------
The permutation analysis tests whether the fixed
authoritative K=6 assignment has stronger internal
geometry than random label assignments with exactly
the same cluster sizes.

A significant result supports non-random geometric
organization of the frozen partition.

It does NOT independently establish:
- clinical subtype validity
- diagnosis separation
- biological mechanism
- external reproducibility
- a uniquely true value of K


DESIGN PROTECTION
-----------------
Diagnosis used:
NO

ADHD/YBOCS/BDI used:
NO

K changed:
NO

Clinical names assigned:
NO


FINAL STATUS
------------
STAGE 3D COMPUTATIONAL PASS —
FIXED-LABEL PERMUTATION SIGNIFICANCE COMPLETE.
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
# 18. OUTPUT LOCK
# ============================================================

OUTPUT_LOCK = (
    LOCK
    / "STAGE3D_PERMUTATION_SIGNIFICANCE_LOCK.json"
)


output_lock = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE3D_PERMUTATION_SIGNIFICANCE_LOCK",

    "status":
        "FROZEN",

    "N":
        85,

    "K":
        6,

    "B":
        B,

    "permutation_seed":
        PERMUTATION_SEED,

    "p_values":
        {

            "silhouette":
                p_silhouette,

            "Davies_Bouldin":
                p_db,

            "Calinski_Harabasz":
                p_ch
        },

    "K_remains_frozen":
        True,

    "clinical_variables_used":
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

            "permutation_design":
                sha256_file(
                    PERM_DESIGN_LOCK
                )
        },

    "output_hashes":
        {

            "all_permutations":
                sha256_file(
                    PERMUTATIONS_FILE
                ),

            "result_table":
                sha256_file(
                    RESULT_FILE
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


    if existing != output_lock:

        raise RuntimeError(
            "Existing Stage3D output lock differs. STOP."
        )


    print(
        "\nExisting Stage3D output lock verified."
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
        "\nStage3D permutation output lock created."
    )


# ============================================================
# 19. PRINT RESULT
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 3D — PERMUTATION SIGNIFICANCE RESULTS"
)

print(
    "=" * 120
)


print(

    result_table.to_string(

        index=False,

        float_format=lambda x: f"{x:.10f}"
    )
)


print(
    "\nObserved silhouette:",
    f"{observed_silhouette:.10f}"
)

print(
    "Permutation p:",
    f"{p_silhouette:.10f}"
)


print(
    "\nObserved Davies-Bouldin:",
    f"{observed_db:.10f}"
)

print(
    "Permutation p:",
    f"{p_db:.10f}"
)


print(
    "\nObserved Calinski-Harabasz:",
    f"{observed_ch:.10f}"
)

print(
    "Permutation p:",
    f"{p_ch:.10f}"
)


# ============================================================
# 20. HASHES
# ============================================================

print(
    "\nHASHES"
)


print(
    "Permutation design:",
    sha256_file(
        PERM_DESIGN_LOCK
    )
)

print(
    "1000 permutations:",
    sha256_file(
        PERMUTATIONS_FILE
    )
)

print(
    "Result table:",
    sha256_file(
        RESULT_FILE
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
# 21. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 3D PASS — "
    "FIXED-LABEL PERMUTATION SIGNIFICANCE COMPLETE"
)

print(
    "=" * 120
)


print(
    "\nK REMAINS FROZEN AT 6."
)

print(
    "NO CLINICAL VARIABLES WERE USED."
)

print(
    "NO CLUSTER NAMES WERE ASSIGNED."
)


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "Permutation significance means the observed internal "
    "geometry is stronger than random assignments with the "
    "same cluster sizes."
)

print(
    "It does NOT mean that six clinical subtypes have been validated."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 3E — SUBJECT-RESAMPLING ROBUSTNESS "
    "OF THE FROZEN K=6 SOLUTION."
)