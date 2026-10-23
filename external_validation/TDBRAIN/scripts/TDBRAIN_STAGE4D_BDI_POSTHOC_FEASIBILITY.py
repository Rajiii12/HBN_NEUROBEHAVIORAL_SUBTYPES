# ==================================================================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# STAGE 4D — BDI WITHIN-MDD POST-HOC DESIGN + FEASIBILITY LOCK
# ==================================================================================================
#
# SCIENTIFIC TRIGGER:
#   Stage4C found a significant prespecified omnibus BDI_pre association
#   within the MDD subgroup after secondary-family BH-FDR.
#
# PURPOSE:
#   Freeze the follow-up pairwise design BEFORE examining pairwise BDI results.
#
# THIS STAGE:
#   - verifies Stage4C frozen outputs
#   - verifies the significant BDI omnibus result
#   - identifies MDD-represented authoritative clusters
#   - freezes ALL possible pairwise comparisons
#   - checks whether exact label-permutation enumeration is computationally feasible
#   - freezes Holm family-wise correction across all pairwise comparisons
#
# THIS STAGE DOES NOT:
#   - compute pairwise BDI test statistics
#   - compute pairwise p-values
#   - inspect/select pairs based on BDI means or medians
#   - change K
#   - rename clusters
#   - open additional clinical variables
#   - use post-treatment variables
# ==================================================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
import math
from itertools import combinations
from datetime import datetime, timezone


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 4D — BDI WITHIN-MDD POST-HOC DESIGN + FEASIBILITY LOCK")
print("=" * 120)


# ==================================================================================================
# 1. ROOTS
# ==================================================================================================

ROOT = Path(
    "/content/drive/MyDrive/"
    "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION"
)

LOCK_DIR = ROOT / "01_LOCK"
CHAR_DIR = ROOT / "08_CHARACTERIZATION"

if not ROOT.exists():
    raise RuntimeError(
        f"Experiment-3 root missing:\n{ROOT}"
    )


# ==================================================================================================
# 2. HASH HELPER
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
# 3. FROZEN STAGE4C INPUTS
# ==================================================================================================

FILES = {

    "stage4a_analysis":
        CHAR_DIR / "STAGE4A_R1_BASELINE_HELDOUT_AUDIT_TABLE.csv",

    "stage4c_implementation":
        LOCK_DIR / "STAGE4C_ASSOCIATION_IMPLEMENTATION_LOCK.json",

    "stage4c_results":
        CHAR_DIR / "STAGE4C_HELDOUT_ASSOCIATION_RESULTS.csv",

    "stage4c_numeric_descriptives":
        CHAR_DIR / "STAGE4C_NUMERIC_CLUSTER_DESCRIPTIVES.csv",

    "stage4c_categorical_counts":
        CHAR_DIR / "STAGE4C_CATEGORICAL_CONTINGENCY_COUNTS.csv",

    "stage4c_descriptive_only":
        CHAR_DIR / "STAGE4C_DESCRIPTIVE_ONLY_ENDPOINTS.csv",

    "stage4c_nulls":
        CHAR_DIR / "STAGE4C_10000_PERMUTATION_NULLS.csv",

    "stage4c_summary":
        CHAR_DIR / "STAGE4C_HELDOUT_ASSOCIATION_SUMMARY.json",

    "stage4c_report":
        CHAR_DIR / "STAGE4C_HELDOUT_ASSOCIATION_REPORT.txt",

    "stage4c_output_lock":
        LOCK_DIR / "STAGE4C_HELDOUT_ASSOCIATION_OUTPUT_LOCK.json",
}


EXPECTED_HASHES = {

    "stage4a_analysis":
        "580854ea8ef6ebe8ff6ae5ef8a31ce9cae3e53dfba07bc0f4f2da24515efcddc",

    "stage4c_implementation":
        "d97d83c413cbdc29c0921ca30e08b4c4afddca74c196b156d1e1f06d5ad822a3",

    "stage4c_results":
        "620505a526b170bc182f0c225ccea386f110353f72a6ad1488a0a7c0311420e4",

    "stage4c_numeric_descriptives":
        "88fb61028eaf3b1ac7d445e499e7d4d0086fe8425390473e4de2a05e7ee35a47",

    "stage4c_categorical_counts":
        "57acafa8c119298a77edecf5548eda461c77baa0f769c30064ff89f00309f32b",

    "stage4c_descriptive_only":
        "c72e079c1cf114813e6e078e1288c0ac8bcb0497c3d10d9a448513c36f59f217",

    "stage4c_nulls":
        "82ca4cebb881e0e577aca4d32ee66319cd4c27c91ab7bbe77e4c07d066b840d9",

    "stage4c_summary":
        "b0aa2b17836be4e08d9d4920ac1e93e706b2c6b760be49923d8e92824ce555d2",

    "stage4c_report":
        "6a13f709e0ac11d2d6eba3a1b48b37fda0537c994297c487acdbf32579c39e95",

    "stage4c_output_lock":
        "98a6df6e5c7cb4b15155e69bf1c3d98c7b4296e5534729adeef4caaa3bb67baf",
}


print("\n" + "=" * 120)
print("FROZEN STAGE4C INPUT VERIFICATION")
print("=" * 120)

for name, path in FILES.items():

    if not path.exists():

        raise RuntimeError(
            f"Missing required frozen Stage4C file:\n"
            f"{name}\n{path}"
        )

    actual = sha256_file(path)
    expected = EXPECTED_HASHES[name]

    if actual != expected:

        raise RuntimeError(
            f"HASH FAILURE: {name}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "STOP."
        )

    print(f"PASS {name}")
    print("    ", path)


# ==================================================================================================
# 4. LOAD ONLY FILES REQUIRED FOR DESIGN / COUNTS
# ==================================================================================================

analysis = pd.read_csv(
    FILES["stage4a_analysis"]
)

stage4c_results = pd.read_csv(
    FILES["stage4c_results"]
)


# ==================================================================================================
# 5. VERIFY AUTHORITATIVE STATE
# ==================================================================================================

if len(analysis) != 85:

    raise RuntimeError(
        f"Expected N=85, observed N={len(analysis)}."
    )

if analysis["subject_key"].duplicated().any():

    raise RuntimeError(
        "Duplicate subject_key found."
    )

analysis["EEG_PHENO_cluster"] = pd.to_numeric(
    analysis["EEG_PHENO_cluster"],
    errors="raise"
).astype(int)

expected_cluster_sizes = {
    0: 27,
    1: 15,
    2: 18,
    3: 10,
    4: 9,
    5: 6,
}

actual_cluster_sizes = {
    int(k): int(v)
    for k, v in
    analysis["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
    .items()
}

if actual_cluster_sizes != expected_cluster_sizes:

    raise RuntimeError(
        "Authoritative K=6 partition mismatch."
    )

print("\nPASS — authoritative N=85, K=6 partition verified.")


# ==================================================================================================
# 6. VERIFY THE STAGE4C BDI OMNIBUS RESULT THAT JUSTIFIES POST-HOC TESTING
# ==================================================================================================

bdi_result = stage4c_results[
    (
        stage4c_results["family"]
        == "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY"
    )
    &
    (
        stage4c_results["endpoint"]
        == "BDI_pre"
    )
    &
    (
        stage4c_results["population"]
        == "MDD_ONLY"
    )
].copy()

if len(bdi_result) != 1:

    raise RuntimeError(
        f"Expected exactly one frozen BDI_pre MDD result; found {len(bdi_result)}."
    )

bdi_row = bdi_result.iloc[0]

if bool(bdi_row["significant_q05"]) is not True:

    raise RuntimeError(
        "Frozen Stage4C BDI result is not FDR-significant. "
        "Post-hoc testing is not justified."
    )

observed_H = float(
    bdi_row["observed_statistic"]
)

epsilon_squared = float(
    bdi_row["effect_size"]
)

permutation_p = float(
    bdi_row["permutation_p"]
)

adjusted_p = float(
    bdi_row["p_adjusted"]
)

print("\n" + "=" * 120)
print("STAGE4C BDI OMNIBUS TRIGGER VERIFIED")
print("=" * 120)

print("Population           = MDD_ONLY")
print("Endpoint             = BDI_pre")
print("N                    =", int(bdi_row["N"]))
print("Kruskal-Wallis H     =", observed_H)
print("epsilon_squared      =", epsilon_squared)
print("permutation p        =", permutation_p)
print("secondary-family q   =", adjusted_p)
print("FDR significant      =", bool(bdi_row["significant_q05"]))

if int(bdi_row["N"]) != 31:

    raise RuntimeError(
        "Frozen MDD BDI N is not 31."
    )


# ==================================================================================================
# 7. BUILD MDD SUBGROUP USING DIAGNOSIS ONLY
#
# IMPORTANT:
# We do NOT inspect BDI values here.
# Only cluster membership and availability counts are used.
# ==================================================================================================

mdd = analysis[
    analysis["diagnosis_group_standardized"] == "MDD"
].copy()

if len(mdd) != 31:

    raise RuntimeError(
        f"Expected 31 MDD subjects, found {len(mdd)}."
    )

if mdd["BDI_pre"].notna().sum() != 31:

    raise RuntimeError(
        "BDI_pre is not complete across the frozen MDD subgroup."
    )


mdd_cluster_counts = {
    int(k): int(v)
    for k, v in
    mdd["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
    .items()
}

EXPECTED_MDD_CLUSTER_COUNTS = {
    0: 10,
    1: 5,
    2: 7,
    3: 5,
    4: 4,
}

if mdd_cluster_counts != EXPECTED_MDD_CLUSTER_COUNTS:

    raise RuntimeError(
        "MDD cluster-count mismatch.\n"
        f"Expected: {EXPECTED_MDD_CLUSTER_COUNTS}\n"
        f"Observed: {mdd_cluster_counts}"
    )

if 5 in mdd_cluster_counts:

    raise RuntimeError(
        "Cluster 5 unexpectedly contains an MDD subject."
    )

print("\n" + "=" * 120)
print("MDD CLUSTER FEASIBILITY COUNTS")
print("=" * 120)

for cluster_id, n in mdd_cluster_counts.items():

    print(
        f"C{cluster_id}: N={n}"
    )

print("C5: N=0 — excluded from MDD pairwise testing.")


# ==================================================================================================
# 8. FREEZE ALL POSSIBLE MDD CLUSTER PAIRS
# ==================================================================================================

represented_clusters = sorted(
    mdd_cluster_counts.keys()
)

pair_list = list(
    combinations(
        represented_clusters,
        2
    )
)

if len(pair_list) != 10:

    raise RuntimeError(
        f"Expected 10 pairwise comparisons, found {len(pair_list)}."
    )


# ==================================================================================================
# 9. EXACT-PERMUTATION ENUMERATION FEASIBILITY
#
# For a pair with n1 and n2 subjects, the complete number of unique label
# assignments is C(n1+n2, n1).
#
# No BDI values are used here.
# ==================================================================================================

MAX_EXACT_ENUMERATIONS = 200000

feasibility_rows = []

for cluster_a, cluster_b in pair_list:

    n_a = mdd_cluster_counts[cluster_a]
    n_b = mdd_cluster_counts[cluster_b]

    n_total = n_a + n_b

    exact_assignments = math.comb(
        n_total,
        n_a
    )

    exact_feasible = (
        exact_assignments
        <= MAX_EXACT_ENUMERATIONS
    )

    feasibility_rows.append({

        "endpoint":
            "BDI_pre",

        "population":
            "MDD_ONLY",

        "cluster_A":
            int(cluster_a),

        "cluster_B":
            int(cluster_b),

        "N_A":
            int(n_a),

        "N_B":
            int(n_b),

        "pair_N":
            int(n_total),

        "exact_label_assignments":
            int(exact_assignments),

        "maximum_allowed_exact_assignments":
            int(MAX_EXACT_ENUMERATIONS),

        "exact_permutation_feasible":
            bool(exact_feasible),

        "planned_test":
            "two_sided_exact_permutation_Mann_Whitney_rank_test",

        "effect_size":
            "rank_biserial_correlation",

        "multiplicity":
            "Holm_familywise_across_all_10_pairs",
    })


feasibility = pd.DataFrame(
    feasibility_rows
)


if not feasibility[
    "exact_permutation_feasible"
].all():

    print(
        feasibility.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "At least one frozen pair is not feasible for exact enumeration. "
        "STOP before choosing another method."
    )


# ==================================================================================================
# 10. FREEZE EXACT POST-HOC STATISTICAL METHOD
# ==================================================================================================

POSTHOC_DESIGN = {

    "stage":
        "STAGE4D_BDI_POSTHOC_DESIGN",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "scientific_trigger": {

        "source_stage":
            "STAGE4C",

        "endpoint":
            "BDI_pre",

        "population":
            "MDD_ONLY",

        "N":
            31,

        "omnibus_test":
            "permutation_Kruskal_Wallis",

        "observed_H":
            observed_H,

        "epsilon_squared":
            epsilon_squared,

        "permutation_p":
            permutation_p,

        "BH_adjusted_p":
            adjusted_p,

        "significant_q05":
            True,
    },

    "authoritative_model_state": {

        "N_total":
            85,

        "K":
            6,

        "clusters":
            [0, 1, 2, 3, 4, 5],

        "MDD_represented_clusters":
            represented_clusters,

        "MDD_cluster_counts":
            {
                str(k): int(v)
                for k, v in mdd_cluster_counts.items()
            },

        "cluster_5_MDD_N":
            0,
    },

    "posthoc_scope": {

        "endpoint":
            "BDI_pre",

        "population":
            "MDD_ONLY",

        "number_of_pairwise_comparisons":
            10,

        "all_pairs_required":
            True,

        "pairs":
            [
                [int(a), int(b)]
                for a, b in pair_list
            ],

        "pair_selection_based_on_BDI_values":
            False,
    },

    "statistical_method": {

        "test":
            "two-sided exact permutation Mann-Whitney rank test",

        "permutation_type":
            "complete enumeration of all unique group-label assignments",

        "test_statistic":
            (
                "absolute deviation of Mann-Whitney U from its null center "
                "n_A*n_B/2"
            ),

        "two_sided_rule":
            (
                "A permutation is at least as extreme when "
                "|U_perm - n_A*n_B/2| >= |U_obs - n_A*n_B/2|."
            ),

        "exact_p_value":
            (
                "number of exact label assignments at least as extreme as observed "
                "divided by the total number of exact label assignments"
            ),

        "ties":
            (
                "Ranks are computed on the pooled observed BDI values using average "
                "ranks; the complete label-permutation distribution therefore "
                "preserves any ties."
            ),

        "effect_size":
            "rank_biserial_correlation = 2*U_A/(n_A*n_B) - 1",

        "direction":
            (
                "Effect-size sign refers to cluster_A relative to cluster_B; "
                "positive values indicate larger rank tendency in cluster_A."
            ),

        "descriptives_to_report":
            [
                "N",
                "mean",
                "SD",
                "median",
                "Q1",
                "Q3",
                "minimum",
                "maximum",
            ],
    },

    "multiplicity": {

        "family":
            "all_10_BDI_within_MDD_pairwise_cluster_comparisons",

        "method":
            "Holm step-down family-wise error correction",

        "familywise_alpha":
            0.05,

        "number_of_tests":
            10,
    },

    "interpretation_rules": [

        "Post-hoc inference is restricted to BDI_pre within the MDD subgroup.",

        "All 10 possible comparisons among MDD-represented clusters 0-4 must be tested.",

        "Cluster 5 is not tested because it contains zero MDD subjects.",

        "No pair may be omitted or added after examining BDI values.",

        "Holm-adjusted p-values determine pairwise statistical significance.",

        "Raw pairwise p-values alone must not be used to claim significant contrasts.",

        "Pairwise BDI differences do not justify renaming the full multimodal clusters as depression subtypes.",

        "Diagnosis remains non-significantly associated with the full K=6 partition in Stage4C.",

        "No new endpoint is opened in this stage.",

        "No model parameter or cluster assignment may be changed.",

        "Post-treatment variables remain sealed.",
    ],

    "not_authorized_in_stage4d": [

        "pairwise hypothesis testing",

        "pairwise p-value calculation",

        "clinical cluster naming",

        "new endpoint testing",

        "K selection",

        "model retuning",

        "post-treatment analysis",
    ],
}


# ==================================================================================================
# 11. SAVE DESIGN + FEASIBILITY
# ==================================================================================================

DESIGN_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4D_BDI_POSTHOC_DESIGN_LOCK.json"
)

FEASIBILITY_FILE = (
    CHAR_DIR
    / "STAGE4D_BDI_POSTHOC_FEASIBILITY.csv"
)

REPORT_FILE = (
    CHAR_DIR
    / "STAGE4D_BDI_POSTHOC_DESIGN_REPORT.txt"
)


with open(
    DESIGN_LOCK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        POSTHOC_DESIGN,
        f,
        indent=2
    )


feasibility.to_csv(
    FEASIBILITY_FILE,
    index=False
)


# ==================================================================================================
# 12. TEXT REPORT
# ==================================================================================================

report = []

report.append(
    "TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3"
)

report.append(
    "STAGE 4D — BDI WITHIN-MDD POST-HOC DESIGN + FEASIBILITY LOCK"
)

report.append(
    "=" * 110
)

report.append("")
report.append("SCIENTIFIC TRIGGER")

report.append(
    f"Stage4C MDD-only BDI omnibus H = {observed_H:.10f}"
)

report.append(
    f"epsilon_squared = {epsilon_squared:.10f}"
)

report.append(
    f"permutation p = {permutation_p:.10f}"
)

report.append(
    f"BH-adjusted p = {adjusted_p:.10f}"
)

report.append("")

report.append(
    "MDD cluster counts = "
    + str(mdd_cluster_counts)
)

report.append(
    "Cluster 5 has no MDD subjects and is excluded."
)

report.append("")

report.append(
    "ALL 10 pairwise comparisons are frozen before pairwise BDI testing."
)

report.append("")

report.append(
    feasibility.to_string(
        index=False
    )
)

report.append("")

report.append(
    "PAIRWISE METHOD: two-sided exact permutation Mann-Whitney rank test."
)

report.append(
    "MULTIPLICITY: Holm FWER correction across all 10 comparisons."
)

report.append("")

report.append(
    "NO PAIRWISE TEST STATISTIC OR PAIRWISE P-VALUE WAS COMPUTED IN STAGE4D."
)

report.append(
    "K REMAINS FROZEN AT 6."
)

report.append(
    "NO CLUSTER NAME WAS ASSIGNED."
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# ==================================================================================================
# 13. OUTPUT LOCK
# ==================================================================================================

OUTPUT_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4D_BDI_POSTHOC_OUTPUT_LOCK.json"
)

output_lock = {

    "stage":
        "STAGE4D",

    "status":
        "PASS",

    "stage4c_source_hashes":
        EXPECTED_HASHES,

    "design_lock": {

        "path":
            str(DESIGN_LOCK_FILE),

        "sha256":
            sha256_file(
                DESIGN_LOCK_FILE
            ),
    },

    "feasibility": {

        "path":
            str(FEASIBILITY_FILE),

        "sha256":
            sha256_file(
                FEASIBILITY_FILE
            ),
    },

    "report": {

        "path":
            str(REPORT_FILE),

        "sha256":
            sha256_file(
                REPORT_FILE
            ),
    },

    "pairwise_tests_run":
        False,

    "pairwise_p_values_computed":
        False,

    "BDI_values_used_for_pair_selection":
        False,

    "new_endpoints_opened":
        False,

    "K_changed":
        False,

    "cluster_names_assigned":
        False,

    "post_treatment_used":
        False,
}


with open(
    OUTPUT_LOCK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_lock,
        f,
        indent=2
    )


# ==================================================================================================
# 14. PRINT RESULTS
# ==================================================================================================

print("\n" + "=" * 120)
print("FROZEN BDI POST-HOC PAIRS + EXACT ENUMERATION FEASIBILITY")
print("=" * 120)

print(
    feasibility.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("FROZEN POST-HOC METHOD")
print("=" * 120)

print("""
Endpoint:
    BDI_pre

Population:
    MDD_ONLY

Represented clusters:
    0, 1, 2, 3, 4

Cluster 5:
    N=0 MDD subjects — excluded

Number of pairwise comparisons:
    10 / 10 possible pairs

Test:
    Two-sided exact permutation Mann-Whitney rank test

Effect size:
    Rank-biserial correlation

Multiplicity:
    Holm family-wise correction across all 10 pairs

Family-wise alpha:
    0.05

No pair was selected from BDI means, medians, or significance.
""")


# ==================================================================================================
# 15. HASHES
# ==================================================================================================

print("\n" + "=" * 120)
print("STAGE4D HASHES")
print("=" * 120)

print("Design lock:")
print(
    sha256_file(
        DESIGN_LOCK_FILE
    )
)

print("\nPairwise feasibility:")
print(
    sha256_file(
        FEASIBILITY_FILE
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
# 16. FINAL STATUS
# ==================================================================================================

print("\n" + "=" * 120)
print("FINAL STATUS")
print("=" * 120)

print("""
STAGE 4D PASS IF NO ERROR APPEARED ABOVE.

IMPORTANT:

- Stage4C BDI significance was verified from the frozen result.
- The MDD-only subgroup remains N=31.
- MDD occurs in clusters 0,1,2,3,4 only.
- Cluster 5 contains no MDD participants.
- ALL 10 possible pairwise comparisons have been frozen.
- Exact permutation enumeration is required for every pair.
- Holm correction across all 10 pairs is frozen.
- No pairwise BDI statistic was calculated.
- No pairwise p-value was calculated.
- No BDI-driven pair selection occurred.
- No additional clinical endpoint was opened.
- K remains frozen at 6.
- Clusters remain numeric 0-5.
- No post-treatment variable was used.

NEXT PIPELINE STAGE:
Stage 4E implements the prespecified exact MDD BDI pairwise tests.
""")