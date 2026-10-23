# ==================================================================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# STAGE 4E — EXACT MDD BDI PAIRWISE POST-HOC ANALYSIS
# ==================================================================================================
#
# PURPOSE
# -------
# Complete the pairwise decomposition of the significant Stage4C
# MDD-only BDI omnibus result.
#
# THIS ANALYSIS WAS ALREADY FROZEN IN STAGE4D.
#
# Population:
#     MDD only
#
# Endpoint:
#     BDI_pre only
#
# Authoritative K:
#     K = 6
#
# MDD cluster counts:
#     C0 = 10
#     C1 = 5
#     C2 = 7
#     C3 = 5
#     C4 = 4
#     C5 = 0
#
# All ten possible comparisons among C0-C4 are required.
#
# Test:
#     exact two-sided permutation Mann-Whitney rank test
#
# Exact statistic:
#     pooled average ranks
#     U_A = R_A - nA(nA+1)/2
#     D_obs = |U_A - nA*nB/2|
#
# Exact p-value:
#     number of assignments with D >= D_obs / total assignments
#
# IMPORTANT:
#     NO +1 correction because every possible label assignment is enumerated.
#
# Effect:
#     rank-biserial correlation
#     RBC = 2U_A/(nA*nB) - 1
#
# Positive RBC:
#     Cluster A has higher BDI ranks.
#
# Multiplicity:
#     Holm FWER correction across ALL 10 comparisons.
#
# NO:
#     extra endpoints
#     pair selection
#     confidence intervals
#     post-treatment variables
#     cluster renaming
#     K changes
#
# ==================================================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
import os
import math

from itertools import combinations
from scipy.stats import rankdata


print("=" * 120)
print("TDBRAIN EXPERIMENT 3")
print("STAGE 4E — EXACT MDD BDI PAIRWISE POST-HOC ANALYSIS")
print("=" * 120)


# ==================================================================================================
# 1. ROOTS
# ==================================================================================================

ROOT = Path(
    "/content/drive/MyDrive/"
    "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION"
)

CHAR_DIR = ROOT / "08_CHARACTERIZATION"
LOCK_DIR = ROOT / "01_LOCK"

CHAR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================================================================
# 2. SHA256
# ==================================================================================================

def sha256_file(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


# ==================================================================================================
# 3. FROZEN INPUT HASHES
# ==================================================================================================

EXPECTED = {

    # Stage4A analysis table
    "stage4a_analysis":
        "580854ea8ef6ebe8ff6ae5ef8a31ce9cae3e53dfba07bc0f4f2da24515efcddc",

    # Stage4D frozen post-hoc design
    "stage4d_design":
        "7753f795459b2dd036ff406495bbfcccda0a864f2726fa2a8a40c806585d2d5d",

    # Stage4D feasibility table
    "stage4d_feasibility":
        "ba97fbb823bbbb705952d893da5c7373dcb613330fa94053689d8a1274a9ba80",

    # Stage4D report
    "stage4d_report":
        "d92fbdc375a1ce9b5ff5eeafe1e4472d35c7652da4eac67fed2004fb01f0ea52",

    # Stage4D output lock
    "stage4d_output_lock":
        "4728fdab90a56d4d4c3e6fd6b9a51d986d69e34adbd34f5f6060010705da27f4",
}


# ==================================================================================================
# 4. LOCATE EXISTING FROZEN FILES BY SHA256
# ==================================================================================================

SEARCH_DIRS = [
    ROOT / "00_METADATA_AUDIT",
    ROOT / "01_LOCK",
    ROOT / "02_INPUTS",
    ROOT / "03_GRAPH",
    ROOT / "04_VGAE",
    ROOT / "05_UMAP_K_SELECTION",
    ROOT / "06_ROBUSTNESS",
    ROOT / "07_PERMUTATION",
    ROOT / "08_CHARACTERIZATION",
    ROOT / "09_EXTERNAL_VALIDATION",
]


def locate_by_hash(expected_hash):

    hits = []

    for base in SEARCH_DIRS:

        if not base.exists():
            continue

        for dirpath, _, filenames in os.walk(base):

            for filename in filenames:

                p = Path(dirpath) / filename

                if p.suffix.lower() not in {
                    ".csv",
                    ".json",
                    ".txt",
                    ".md",
                    ".npy"
                }:
                    continue

                try:

                    if sha256_file(p) == expected_hash:
                        hits.append(p)

                except Exception:
                    pass


    if len(hits) == 0:

        raise RuntimeError(
            "Frozen input not found.\n"
            f"SHA256 = {expected_hash}"
        )


    if len(hits) > 1:

        print("\nIdentical-hash copies found:")

        for p in hits:
            print("   ", p)


    return hits[0]


print("\n" + "=" * 120)
print("FROZEN INPUT VERIFICATION")
print("=" * 120)

located = {}

for name, digest in EXPECTED.items():

    p = locate_by_hash(
        digest
    )

    located[name] = p

    print("PASS", name)
    print("    ", p)
    print("     SHA256 =", sha256_file(p))


# ==================================================================================================
# 5. LOAD FROZEN STAGE4A TABLE
# ==================================================================================================

df = pd.read_csv(
    located["stage4a_analysis"]
)


required_columns = [
    "EEG_PHENO_cluster",
    "diagnosis_group_standardized",
    "BDI_pre",
]


for col in required_columns:

    if col not in df.columns:

        raise RuntimeError(
            f"Required Stage4A column missing: {col}"
        )


print("\nPASS — required Stage4A columns verified.")


# ==================================================================================================
# 6. FILTER EXACT FROZEN POPULATION
# ==================================================================================================

mdd = df[
    (
        df["diagnosis_group_standardized"] == "MDD"
    )
    &
    (
        df["BDI_pre"].notna()
    )
].copy()


mdd["EEG_PHENO_cluster"] = (
    mdd["EEG_PHENO_cluster"]
    .astype(int)
)


if len(mdd) != 31:

    raise RuntimeError(
        f"Expected MDD BDI N=31, found {len(mdd)}."
    )


cluster_counts = (
    mdd["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
    .to_dict()
)


expected_cluster_counts = {
    0: 10,
    1: 5,
    2: 7,
    3: 5,
    4: 4,
}


if cluster_counts != expected_cluster_counts:

    raise RuntimeError(
        "Frozen MDD cluster counts do not match Stage4D.\n"
        f"Expected = {expected_cluster_counts}\n"
        f"Observed = {cluster_counts}"
    )


if (
    mdd["EEG_PHENO_cluster"] == 5
).any():

    raise RuntimeError(
        "Unexpected MDD subject found in Cluster 5."
    )


print("\n" + "=" * 120)
print("MDD BDI POPULATION VERIFIED")
print("=" * 120)

print("MDD BDI N =", len(mdd))

print(
    mdd[
        "EEG_PHENO_cluster"
    ]
    .value_counts()
    .sort_index()
)

print("\nPASS — frozen Stage4D MDD counts reproduced exactly.")


# ==================================================================================================
# 7. FROZEN PAIR SET
# ==================================================================================================

CLUSTERS = [
    0,
    1,
    2,
    3,
    4
]


PAIR_ORDER = list(
    combinations(
        CLUSTERS,
        2
    )
)


if len(PAIR_ORDER) != 10:

    raise RuntimeError(
        "Expected exactly 10 cluster pairs."
    )


EXPECTED_ASSIGNMENTS = {

    (0, 1): 3003,

    (0, 2): 19448,

    (0, 3): 3003,

    (0, 4): 1001,

    (1, 2): 792,

    (1, 3): 252,

    (1, 4): 126,

    (2, 3): 792,

    (2, 4): 330,

    (3, 4): 126,
}


print("\n" + "=" * 120)
print("FROZEN PAIR SET")
print("=" * 120)

for a, b in PAIR_ORDER:

    nA = expected_cluster_counts[a]
    nB = expected_cluster_counts[b]

    total = math.comb(
        nA + nB,
        nA
    )

    expected_total = EXPECTED_ASSIGNMENTS[
        (a, b)
    ]

    if total != expected_total:

        raise RuntimeError(
            f"C{a} vs C{b}: assignment-count mismatch. "
            f"Calculated={total}, expected={expected_total}"
        )

    print(
        f"C{a} vs C{b}: "
        f"nA={nA}, nB={nB}, "
        f"exact assignments={total}"
    )


print(
    "\nPASS — all ten Stage4D exact-assignment totals reproduced."
)


# ==================================================================================================
# 8. CLUSTER DESCRIPTIVES
# ==================================================================================================

descriptive_rows = []


for c in CLUSTERS:

    x = (
        mdd.loc[
            mdd["EEG_PHENO_cluster"] == c,
            "BDI_pre"
        ]
        .astype(float)
        .to_numpy()
    )


    descriptive_rows.append(
        {
            "cluster":
                c,

            "N":
                int(len(x)),

            "mean":
                float(np.mean(x)),

            "SD":
                (
                    float(
                        np.std(
                            x,
                            ddof=1
                        )
                    )
                    if len(x) > 1
                    else np.nan
                ),

            "median":
                float(
                    np.median(x)
                ),

            "Q1":
                float(
                    np.quantile(
                        x,
                        0.25
                    )
                ),

            "Q3":
                float(
                    np.quantile(
                        x,
                        0.75
                    )
                ),

            "minimum":
                float(
                    np.min(x)
                ),

            "maximum":
                float(
                    np.max(x)
                ),
        }
    )


descriptives = pd.DataFrame(
    descriptive_rows
)


print("\n" + "=" * 120)
print("MDD BDI DESCRIPTIVES BY AUTHORITATIVE K=6 CLUSTER")
print("=" * 120)

print(
    descriptives.to_string(
        index=False
    )
)


# ==================================================================================================
# 9. EXACT PAIRWISE TEST FUNCTION
# ==================================================================================================

def exact_rank_permutation_test(
    values_A,
    values_B
):

    values_A = np.asarray(
        values_A,
        dtype=float
    )

    values_B = np.asarray(
        values_B,
        dtype=float
    )


    nA = len(values_A)
    nB = len(values_B)

    pooled = np.concatenate(
        [
            values_A,
            values_B
        ]
    )


    # Average ranks explicitly handle tied BDI values.
    ranks = rankdata(
        pooled,
        method="average"
    )


    # Observed group A occupies first nA pooled positions.
    R_A = float(
        np.sum(
            ranks[:nA]
        )
    )


    U_A = (
        R_A
        -
        nA * (nA + 1) / 2.0
    )


    U_center = (
        nA
        *
        nB
        /
        2.0
    )


    D_obs = abs(
        U_A
        -
        U_center
    )


    total_expected = math.comb(
        nA + nB,
        nA
    )


    extreme_count = 0
    enumerated = 0


    # ------------------------------------------------------------------
    # EXACT enumeration of every possible allocation of pooled ranks
    # into a group of size nA.
    # ------------------------------------------------------------------

    for idx_A in combinations(
        range(nA + nB),
        nA
    ):

        R_perm = float(
            np.sum(
                ranks[
                    list(idx_A)
                ]
            )
        )


        U_perm = (
            R_perm
            -
            nA * (nA + 1) / 2.0
        )


        D_perm = abs(
            U_perm
            -
            U_center
        )


        if D_perm >= (
            D_obs - 1e-12
        ):

            extreme_count += 1


        enumerated += 1


    if enumerated != total_expected:

        raise RuntimeError(
            "Exact enumeration count mismatch."
        )


    # NO +1 correction:
    # every possible assignment was enumerated.
    p_exact = (
        extreme_count
        /
        total_expected
    )


    rank_biserial = (
        2.0
        *
        U_A
        /
        (
            nA
            *
            nB
        )
        -
        1.0
    )


    return {

        "nA":
            int(nA),

        "nB":
            int(nB),

        "R_A":
            float(R_A),

        "U_A":
            float(U_A),

        "U_center":
            float(U_center),

        "D_observed":
            float(D_obs),

        "rank_biserial":
            float(rank_biserial),

        "extreme_assignments":
            int(extreme_count),

        "total_assignments":
            int(total_expected),

        "exact_p":
            float(p_exact),
    }


# ==================================================================================================
# 10. RUN ALL TEN FROZEN COMPARISONS
# ==================================================================================================

pairwise_rows = []


for cluster_A, cluster_B in PAIR_ORDER:

    values_A = (
        mdd.loc[
            mdd["EEG_PHENO_cluster"]
            ==
            cluster_A,
            "BDI_pre"
        ]
        .astype(float)
        .to_numpy()
    )


    values_B = (
        mdd.loc[
            mdd["EEG_PHENO_cluster"]
            ==
            cluster_B,
            "BDI_pre"
        ]
        .astype(float)
        .to_numpy()
    )


    result = exact_rank_permutation_test(
        values_A,
        values_B
    )


    expected_assignments = (
        EXPECTED_ASSIGNMENTS[
            (
                cluster_A,
                cluster_B
            )
        ]
    )


    if (
        result[
            "total_assignments"
        ]
        !=
        expected_assignments
    ):

        raise RuntimeError(
            f"C{cluster_A} vs C{cluster_B}: "
            "exact assignment total changed unexpectedly."
        )


    rbc = result[
        "rank_biserial"
    ]


    if rbc > 0:

        direction = (
            f"C{cluster_A}_higher_BDI_ranks"
        )

    elif rbc < 0:

        direction = (
            f"C{cluster_B}_higher_BDI_ranks"
        )

    else:

        direction = "equal_rank_location"


    pairwise_rows.append(
        {
            "endpoint":
                "BDI_pre",

            "population":
                "MDD_ONLY",

            "cluster_A":
                cluster_A,

            "cluster_B":
                cluster_B,

            "N_A":
                result["nA"],

            "N_B":
                result["nB"],

            "mean_A":
                float(
                    np.mean(
                        values_A
                    )
                ),

            "mean_B":
                float(
                    np.mean(
                        values_B
                    )
                ),

            "median_A":
                float(
                    np.median(
                        values_A
                    )
                ),

            "median_B":
                float(
                    np.median(
                        values_B
                    )
                ),

            "rank_sum_A":
                result["R_A"],

            "U_A":
                result["U_A"],

            "D_observed":
                result["D_observed"],

            "effect_name":
                "rank_biserial",

            "rank_biserial":
                result["rank_biserial"],

            "effect_direction":
                direction,

            "extreme_assignments":
                result["extreme_assignments"],

            "total_assignments":
                result["total_assignments"],

            "exact_p":
                result["exact_p"],
        }
    )


pairwise = pd.DataFrame(
    pairwise_rows
)


# ==================================================================================================
# 11. HOLM FWER CORRECTION ACROSS ALL TEN PAIRS
# ==================================================================================================

m = len(
    pairwise
)


if m != 10:

    raise RuntimeError(
        f"Expected 10 tests, found {m}."
    )


pvals = (
    pairwise[
        "exact_p"
    ]
    .astype(float)
    .to_numpy()
)


order = np.argsort(
    pvals,
    kind="mergesort"
)


adjusted_sorted = np.empty(
    m,
    dtype=float
)


running_max = 0.0


for rank0, original_idx in enumerate(
    order
):

    multiplier = (
        m
        -
        rank0
    )


    candidate = (
        multiplier
        *
        pvals[
            original_idx
        ]
    )


    running_max = max(
        running_max,
        candidate
    )


    adjusted_sorted[
        rank0
    ] = min(
        1.0,
        running_max
    )


holm_adjusted = np.empty(
    m,
    dtype=float
)


for rank0, original_idx in enumerate(
    order
):

    holm_adjusted[
        original_idx
    ] = adjusted_sorted[
        rank0
    ]


pairwise[
    "Holm_adjusted_p"
] = holm_adjusted


pairwise[
    "Holm_significant_0.05"
] = (
    pairwise[
        "Holm_adjusted_p"
    ]
    <=
    0.05
)


pairwise[
    "multiplicity"
] = (
    "Holm_FWER_across_all_10_BDI_pairs"
)


# ==================================================================================================
# 12. ENUMERATION AUDIT TABLE
# ==================================================================================================

enumeration_audit = pairwise[
    [
        "cluster_A",
        "cluster_B",
        "N_A",
        "N_B",
        "extreme_assignments",
        "total_assignments",
        "exact_p"
    ]
].copy()


# ==================================================================================================
# 13. PRINT EXACT RESULTS
# ==================================================================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    300
)


print("\n" + "=" * 120)
print("STAGE4E EXACT MDD BDI PAIRWISE RESULTS")
print("=" * 120)

print(
    pairwise.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("HOLM-SIGNIFICANT PAIRS")
print("=" * 120)

sig = pairwise[
    pairwise[
        "Holm_significant_0.05"
    ]
].copy()


if len(sig) == 0:

    print(
        "No BDI pair remains significant after Holm correction."
    )

else:

    print(
        sig[
            [
                "cluster_A",
                "cluster_B",
                "N_A",
                "N_B",
                "median_A",
                "median_B",
                "rank_biserial",
                "exact_p",
                "Holm_adjusted_p"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================================================================
# 14. SAVE RESULTS
# ==================================================================================================

RESULT_FILE = (
    CHAR_DIR
    / "STAGE4E_BDI_EXACT_PAIRWISE_RESULTS.csv"
)


DESCRIPTIVE_FILE = (
    CHAR_DIR
    / "STAGE4E_BDI_MDD_CLUSTER_DESCRIPTIVES.csv"
)


ENUMERATION_FILE = (
    CHAR_DIR
    / "STAGE4E_BDI_EXACT_ENUMERATION_AUDIT.csv"
)


pairwise.to_csv(
    RESULT_FILE,
    index=False
)


descriptives.to_csv(
    DESCRIPTIVE_FILE,
    index=False
)


enumeration_audit.to_csv(
    ENUMERATION_FILE,
    index=False
)


# ==================================================================================================
# 15. SUMMARY
# ==================================================================================================

summary = {

    "stage":
        "STAGE4E_BDI_EXACT_POSTHOC",

    "status":
        "PASS",

    "population":
        "MDD_ONLY",

    "endpoint":
        "BDI_pre",

    "N":
        31,

    "authoritative_K":
        6,

    "cluster_counts":
        {
            "0": 10,
            "1": 5,
            "2": 7,
            "3": 5,
            "4": 4,
            "5": 0,
        },

    "pair_count":
        10,

    "test":
        "exact_two_sided_permutation_Mann_Whitney_rank_test",

    "rank_method":
        "average_ranks_for_ties",

    "two_sided_statistic":
        "abs(U_A - nA*nB/2)",

    "exact_p_definition":
        "extreme_assignments / total_assignments",

    "plus_one_correction":
        False,

    "effect_size":
        "rank_biserial_2U_A_over_nAnB_minus_1",

    "effect_direction":
        "positive_means_cluster_A_higher",

    "multiplicity":
        "Holm_FWER_across_all_10_pairs",

    "alpha":
        0.05,

    "significant_pair_count":
        int(
            pairwise[
                "Holm_significant_0.05"
            ].sum()
        ),

    "significant_pairs":
        [
            f"C{int(row.cluster_A)}-C{int(row.cluster_B)}"
            for row in pairwise[
                pairwise[
                    "Holm_significant_0.05"
                ]
            ].itertuples()
        ],

    "confidence_intervals_computed":
        False,

    "other_endpoints_tested":
        False,

    "post_treatment_used":
        False,

    "K_changed":
        False,

    "clusters_renamed":
        False,

    "input_hashes":
        EXPECTED,
}


SUMMARY_FILE = (
    CHAR_DIR
    / "STAGE4E_BDI_EXACT_POSTHOC_SUMMARY.json"
)


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


# ==================================================================================================
# 16. REPORT
# ==================================================================================================

REPORT_FILE = (
    CHAR_DIR
    / "STAGE4E_BDI_EXACT_POSTHOC_REPORT.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TDBRAIN EXPERIMENT 3\n"
    )

    f.write(
        "STAGE 4E — EXACT MDD BDI PAIRWISE POST-HOC\n\n"
    )

    f.write(
        "Population: MDD only\n"
    )

    f.write(
        "Endpoint: BDI_pre\n"
    )

    f.write(
        "N = 31\n"
    )

    f.write(
        "Authoritative solution: K=6\n\n"
    )

    f.write(
        "Cluster counts:\n"
        "C0=10, C1=5, C2=7, C3=5, C4=4, C5=0\n\n"
    )

    f.write(
        "All ten C0-C4 pairs were evaluated.\n"
    )

    f.write(
        "No pair was selected from observed BDI values.\n\n"
    )

    f.write(
        "Test: exact two-sided permutation Mann-Whitney rank test.\n"
    )

    f.write(
        "Ties were assigned average pooled ranks.\n"
    )

    f.write(
        "Exact p = extreme assignments / all assignments.\n"
    )

    f.write(
        "No +1 correction was used because the permutation space was fully enumerated.\n"
    )

    f.write(
        "Effect size: rank-biserial correlation.\n"
    )

    f.write(
        "Multiplicity: Holm FWER correction across all ten pairs.\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write(
        pairwise.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Holm-significant pair count: "
        +
        str(
            int(
                pairwise[
                    "Holm_significant_0.05"
                ].sum()
            )
        )
        +
        "\n"
    )


# ==================================================================================================
# 17. OUTPUT LOCK
# ==================================================================================================

OUTPUT_LOCK = {

    "stage":
        "STAGE4E_BDI_EXACT_POSTHOC",

    "status":
        "PASS",

    "results": {
        "path":
            str(
                RESULT_FILE
            ),

        "sha256":
            sha256_file(
                RESULT_FILE
            ),
    },

    "descriptives": {
        "path":
            str(
                DESCRIPTIVE_FILE
            ),

        "sha256":
            sha256_file(
                DESCRIPTIVE_FILE
            ),
    },

    "enumeration_audit": {
        "path":
            str(
                ENUMERATION_FILE
            ),

        "sha256":
            sha256_file(
                ENUMERATION_FILE
            ),
    },

    "summary": {
        "path":
            str(
                SUMMARY_FILE
            ),

        "sha256":
            sha256_file(
                SUMMARY_FILE
            ),
    },

    "report": {
        "path":
            str(
                REPORT_FILE
            ),

        "sha256":
            sha256_file(
                REPORT_FILE
            ),
    },

    "primary_K":
        6,

    "primary_K_changed":
        False,

    "post_treatment_used":
        False,

    "extra_endpoints_used":
        False,

    "clusters_renamed":
        False,
}


OUTPUT_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4E_BDI_EXACT_POSTHOC_OUTPUT_LOCK.json"
)


with open(
    OUTPUT_LOCK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        OUTPUT_LOCK,
        f,
        indent=2
    )


# ==================================================================================================
# 18. HASHES
# ==================================================================================================

print("\n" + "=" * 120)
print("STAGE4E HASHES")
print("=" * 120)

print("Pairwise results:")
print(
    sha256_file(
        RESULT_FILE
    )
)

print("\nMDD cluster descriptives:")
print(
    sha256_file(
        DESCRIPTIVE_FILE
    )
)

print("\nExact enumeration audit:")
print(
    sha256_file(
        ENUMERATION_FILE
    )
)

print("\nSummary:")
print(
    sha256_file(
        SUMMARY_FILE
    )
)

print("\nReport:")
print(
    sha256_file(
        REPORT_FILE
    )
)

print("\nOutput lock:")
print(
    sha256_file(
        OUTPUT_LOCK_FILE
    )
)


# ==================================================================================================
# 19. FINAL STATUS
# ==================================================================================================

print("\n" + "=" * 120)
print("FINAL STATUS")
print("=" * 120)

print("""
STAGE 4E PASS IF NO ERROR APPEARED ABOVE.

THIS STAGE COMPLETED THE ALREADY-FROZEN PRIMARY K=6
MDD BDI POST-HOC ANALYSIS.

REQUIRED CONDITIONS:
- MDD BDI N = 31
- C0 = 10
- C1 = 5
- C2 = 7
- C3 = 5
- C4 = 4
- C5 = 0
- exactly 10 pairwise comparisons
- exact assignment totals reproduced
- exact two-sided rank permutation tests
- rank-biserial effect sizes
- Holm correction across all 10 pairs

NO:
- new endpoint
- confidence interval
- post-treatment variable
- pair selection
- K change
- cluster renaming

PRIMARY ANALYSIS COMPLETE:
The frozen K=6 MDD BDI pairwise post-hoc tests are complete.
Any subsequent structural follow-up is a separate analysis.
""")