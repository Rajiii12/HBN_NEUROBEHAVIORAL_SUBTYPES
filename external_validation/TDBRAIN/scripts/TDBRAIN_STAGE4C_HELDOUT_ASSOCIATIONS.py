# ==================================================================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# STAGE 4C — FROZEN HELD-OUT ASSOCIATION TESTING
# ==================================================================================================
#
# AUTHORIZED ONLY AFTER STAGE4B PASSED.
#
# Executes ONLY the inferential endpoints frozen in Stage4B.
#
# PRIMARY:
#   diagnosis_group_standardized
#       permutation Pearson chi-square
#       Cramer's V
#
# TRANSDIAGNOSTIC HELD-OUT FAMILY:
#   age
#   gender
#   education
#   avg_rt_oddb_CP
#   n_oddb_CP
#   n_oddb_FP
#   n_oddb_CN
#   n_oddb_FN
#
# DIAGNOSIS-SPECIFIC SECONDARY:
#   ADHD_pre_Att_leading within ADHD_ADD
#   ADHD_pre_Hyp_leading within ADHD_ADD
#   BDI_pre within MDD
#
# DESCRIPTIVE ONLY:
#   YBOCS_pre
#   avg_rt_oddb_FP
#   NEO-FFI
#
# NO POST-HOC TESTS.
# NO MODEL RETUNING.
# NO K CHANGE.
# NO CLUSTER RENAMING.
# NO POST-TREATMENT VARIABLES.
# ==================================================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from scipy.stats import kruskal, chi2_contingency
from datetime import datetime, timezone

print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 4C — FROZEN HELD-OUT ASSOCIATION TESTING")
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

LOCK_DIR.mkdir(parents=True, exist_ok=True)
CHAR_DIR.mkdir(parents=True, exist_ok=True)


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
# 3. FROZEN STAGE4B INPUTS
# ==================================================================================================

FILES = {

    "stage4a_analysis":
        CHAR_DIR / "STAGE4A_R1_BASELINE_HELDOUT_AUDIT_TABLE.csv",

    "stage4b_design":
        LOCK_DIR / "STAGE4B_STATISTICAL_CHARACTERIZATION_DESIGN_LOCK.json",

    "stage4b_feasibility":
        CHAR_DIR / "STAGE4B_ENDPOINT_FEASIBILITY.csv",

    "stage4b_multiplicity":
        CHAR_DIR / "STAGE4B_MULTIPLICITY_PLAN.csv",

    "stage4b_report":
        CHAR_DIR / "STAGE4B_STATISTICAL_CHARACTERIZATION_DESIGN_REPORT.txt",

    "stage4b_output_lock":
        LOCK_DIR / "STAGE4B_STATISTICAL_CHARACTERIZATION_OUTPUT_LOCK.json",
}


EXPECTED_HASHES = {

    "stage4a_analysis":
        "580854ea8ef6ebe8ff6ae5ef8a31ce9cae3e53dfba07bc0f4f2da24515efcddc",

    "stage4b_design":
        "af978707c3d2223dea151d5a802ec86f2ec1f716097b153eef5cc9d51e8891f4",

    "stage4b_feasibility":
        "9e89d58c59005e44f035bd0e495c110d90b5cd9afac20801af5a6724bee02230",

    "stage4b_multiplicity":
        "0bf58a4606077b3db8ed07dd636c882266f385b2c4e309957a8c0812721fb895",

    "stage4b_report":
        "fa4d5168d32750e37ddae437870ed715d2520895d1c49ac0a096228cf21e39d4",

    "stage4b_output_lock":
        "0311a46f9899a535ce850489cb15a8b7d7cfdfbf6eab71c3767572da067bfa1b",
}


print("\n" + "=" * 120)
print("FROZEN STAGE4B INPUT VERIFICATION")
print("=" * 120)

for name, path in FILES.items():

    if not path.exists():

        raise RuntimeError(
            f"Missing required frozen file:\n{name}\n{path}"
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
# 4. LOAD FROZEN DATA / DESIGN
# ==================================================================================================

analysis = pd.read_csv(
    FILES["stage4a_analysis"]
)

feasibility = pd.read_csv(
    FILES["stage4b_feasibility"]
)

with open(
    FILES["stage4b_design"],
    "r",
    encoding="utf-8"
) as f:

    design = json.load(f)


# ==================================================================================================
# 5. VERIFY AUTHORITATIVE STATE
# ==================================================================================================

if len(analysis) != 85:

    raise RuntimeError(
        f"Expected N=85, found {len(analysis)}."
    )

if analysis["subject_key"].duplicated().any():

    raise RuntimeError(
        "Duplicate subject_key detected."
    )

analysis["EEG_PHENO_cluster"] = pd.to_numeric(
    analysis["EEG_PHENO_cluster"],
    errors="raise"
).astype(int)

expected_sizes = {
    0: 27,
    1: 15,
    2: 18,
    3: 10,
    4: 9,
    5: 6,
}

actual_sizes = {
    int(k): int(v)
    for k, v in
    analysis["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
    .items()
}

if actual_sizes != expected_sizes:

    raise RuntimeError(
        f"Authoritative partition mismatch:\n{actual_sizes}"
    )

print("\nPASS — authoritative N=85, K=6 partition verified.")


# ==================================================================================================
# 6. GLOBAL FROZEN SETTINGS
# ==================================================================================================

N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20261009
FDR_Q = 0.05

if design["global_settings"]["permutations"] != N_PERMUTATIONS:

    raise RuntimeError("Permutation-count mismatch with Stage4B.")

if design["global_settings"]["permutation_seed"] != PERMUTATION_SEED:

    raise RuntimeError("Permutation-seed mismatch with Stage4B.")

if design["global_settings"]["FDR_q"] != FDR_Q:

    raise RuntimeError("FDR threshold mismatch with Stage4B.")

if design["global_settings"]["posthoc_testing_authorized"] is not False:

    raise RuntimeError("Unexpected post-hoc authorization.")

print("PASS — Stage4B inferential settings reproduced exactly.")


# ==================================================================================================
# 7. VERIFY EXACT FEASIBILITY DECISIONS
# ==================================================================================================

expected_decisions = {

    "diagnosis_group_standardized":
        "INFERENTIAL",

    "age":
        "INFERENTIAL",

    "gender":
        "INFERENTIAL",

    "education":
        "INFERENTIAL",

    "avg_rt_oddb_CP":
        "INFERENTIAL",

    "n_oddb_CP":
        "INFERENTIAL",

    "n_oddb_FP":
        "INFERENTIAL",

    "n_oddb_CN":
        "INFERENTIAL",

    "n_oddb_FN":
        "INFERENTIAL",

    "ADHD_pre_Att_leading":
        "INFERENTIAL_SECONDARY",

    "ADHD_pre_Hyp_leading":
        "INFERENTIAL_SECONDARY",

    "YBOCS_pre":
        "DESCRIPTIVE_ONLY",

    "BDI_pre":
        "INFERENTIAL_SECONDARY",
}


for endpoint, expected_decision in expected_decisions.items():

    row = feasibility[
        feasibility["endpoint"] == endpoint
    ]

    if len(row) != 1:

        raise RuntimeError(
            f"Unexpected feasibility rows for {endpoint}: {len(row)}"
        )

    actual_decision = row.iloc[0]["decision"]

    if actual_decision != expected_decision:

        raise RuntimeError(
            f"Stage4B decision mismatch for {endpoint}.\n"
            f"Expected: {expected_decision}\n"
            f"Observed: {actual_decision}"
        )

print("PASS — all Stage4B endpoint decisions reproduced exactly.")


# ==================================================================================================
# 8. FREEZE STAGE4C IMPLEMENTATION BEFORE TESTING
# ==================================================================================================

IMPLEMENTATION_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4C_ASSOCIATION_IMPLEMENTATION_LOCK.json"
)

implementation_lock = {

    "stage":
        "STAGE4C",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "authoritative_N":
        85,

    "authoritative_K":
        6,

    "cluster_field":
        "EEG_PHENO_cluster",

    "permutations":
        N_PERMUTATIONS,

    "permutation_seed":
        PERMUTATION_SEED,

    "empirical_p_formula":
        "(1 + number_of_permuted_statistics_at_least_as_extreme_as_observed) / (B + 1)",

    "numeric_test":
        "Kruskal-Wallis H with permutation p-value",

    "numeric_permutation_rule":
        (
            "Within the prespecified complete-case analysis population, "
            "shuffle endpoint values relative to the fixed authoritative cluster labels."
        ),

    "numeric_effect_size":
        "epsilon_squared = max(0, (H - k + 1) / (n - k))",

    "categorical_test":
        "Pearson chi-square with permutation p-value",

    "categorical_permutation_rule":
        (
            "Shuffle categorical endpoint labels relative to the fixed "
            "authoritative cluster labels."
        ),

    "categorical_effect_size":
        "Cramers_V = sqrt(chi2 / (n * min(r-1,c-1)))",

    "primary_endpoint":
        "diagnosis_group_standardized",

    "transdiagnostic_FDR_family": [
        "age",
        "gender",
        "education",
        "avg_rt_oddb_CP",
        "n_oddb_CP",
        "n_oddb_FP",
        "n_oddb_CN",
        "n_oddb_FN",
    ],

    "diagnosis_specific_FDR_family": [
        "ADHD_pre_Att_leading within ADHD_ADD",
        "ADHD_pre_Hyp_leading within ADHD_ADD",
        "BDI_pre within MDD",
    ],

    "descriptive_only": [
        "YBOCS_pre",
        "avg_rt_oddb_FP",
        "neoFFI_q1 ... neoFFI_q60",
    ],

    "BH_FDR_q":
        FDR_Q,

    "posthoc_testing":
        False,

    "cluster_renaming":
        False,

    "model_retuning":
        False,

    "K_change":
        False,

    "post_treatment_variables":
        False,
}


with open(
    IMPLEMENTATION_LOCK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        implementation_lock,
        f,
        indent=2
    )

IMPLEMENTATION_HASH = sha256_file(
    IMPLEMENTATION_LOCK_FILE
)

print("\n" + "=" * 120)
print("STAGE4C IMPLEMENTATION LOCKED BEFORE TEST STATISTICS")
print("=" * 120)

print(IMPLEMENTATION_LOCK_FILE)
print("SHA256 =", IMPLEMENTATION_HASH)


# ==================================================================================================
# 9. STATISTICAL HELPERS
# ==================================================================================================

def cramers_v_from_table(table):

    table = np.asarray(table)

    n = table.sum()

    if n <= 0:
        return np.nan

    chi2, _, _, _ = chi2_contingency(
        table,
        correction=False
    )

    r, c = table.shape

    denom = min(r - 1, c - 1)

    if denom <= 0:
        return np.nan

    v = np.sqrt(
        chi2 / (n * denom)
    )

    return float(v)


def epsilon_squared_kw(H, n, k):

    if n <= k:
        return np.nan

    eps = (
        (H - k + 1)
        /
        (n - k)
    )

    return float(
        max(0.0, eps)
    )


def bh_fdr(p_values):

    p = np.asarray(
        p_values,
        dtype=float
    )

    m = len(p)

    if m == 0:
        return np.array([])

    order = np.argsort(p)

    ranked = p[order]

    adjusted = np.empty(
        m,
        dtype=float
    )

    running = 1.0

    for i in range(m - 1, -1, -1):

        rank = i + 1

        candidate = (
            ranked[i] * m / rank
        )

        running = min(
            running,
            candidate
        )

        adjusted[order[i]] = min(
            running,
            1.0
        )

    return adjusted


# ==================================================================================================
# 10. PERMUTATION KRUSKAL-WALLIS
# ==================================================================================================

def permutation_kw(
    dataframe,
    endpoint,
    cluster_col,
    rng,
    n_perm
):

    df = dataframe[
        [cluster_col, endpoint]
    ].dropna().copy()

    clusters = sorted(
        df[cluster_col].unique()
    )

    values = df[endpoint].to_numpy(
        dtype=float
    )

    labels = df[cluster_col].to_numpy()

    groups = [
        values[labels == c]
        for c in clusters
    ]

    if len(groups) < 2:

        raise RuntimeError(
            f"{endpoint}: fewer than two represented clusters."
        )

    observed_H = float(
        kruskal(
            *groups
        ).statistic
    )

    null = np.empty(
        n_perm,
        dtype=float
    )

    for i in range(n_perm):

        permuted_values = rng.permutation(
            values
        )

        perm_groups = [
            permuted_values[labels == c]
            for c in clusters
        ]

        null[i] = float(
            kruskal(
                *perm_groups
            ).statistic
        )

    extreme = int(
        np.sum(
            null >= observed_H
        )
    )

    p_perm = (
        extreme + 1
    ) / (
        n_perm + 1
    )

    eps2 = epsilon_squared_kw(
        observed_H,
        n=len(df),
        k=len(clusters)
    )

    cluster_n = {
        int(c):
            int(
                np.sum(
                    labels == c
                )
            )
        for c in clusters
    }

    return {

        "N":
            int(len(df)),

        "represented_clusters":
            int(len(clusters)),

        "observed_statistic":
            observed_H,

        "effect_size":
            eps2,

        "permutation_extreme":
            extreme,

        "permutation_p":
            float(p_perm),

        "null_mean":
            float(np.mean(null)),

        "null_sd":
            float(np.std(null, ddof=1)),

        "null_2.5_percentile":
            float(np.percentile(null, 2.5)),

        "null_97.5_percentile":
            float(np.percentile(null, 97.5)),

        "cluster_N":
            cluster_n,

        "null":
            null,
    }


# ==================================================================================================
# 11. PERMUTATION PEARSON CHI-SQUARE
# ==================================================================================================

def permutation_chi_square(
    dataframe,
    endpoint,
    cluster_col,
    rng,
    n_perm
):

    df = dataframe[
        [cluster_col, endpoint]
    ].dropna().copy()

    labels = df[cluster_col].to_numpy()

    outcome = df[endpoint].to_numpy()

    observed_table = pd.crosstab(
        df[cluster_col],
        df[endpoint]
    )

    observed_chi2 = float(
        chi2_contingency(
            observed_table.values,
            correction=False
        )[0]
    )

    observed_v = cramers_v_from_table(
        observed_table.values
    )

    null = np.empty(
        n_perm,
        dtype=float
    )

    for i in range(n_perm):

        permuted_outcome = rng.permutation(
            outcome
        )

        perm_table = pd.crosstab(
            labels,
            permuted_outcome
        )

        # Ensure same complete category columns
        perm_table = perm_table.reindex(
            index=observed_table.index,
            columns=observed_table.columns,
            fill_value=0
        )

        null[i] = float(
            chi2_contingency(
                perm_table.values,
                correction=False
            )[0]
        )

    extreme = int(
        np.sum(
            null >= observed_chi2
        )
    )

    p_perm = (
        extreme + 1
    ) / (
        n_perm + 1
    )

    return {

        "N":
            int(len(df)),

        "represented_clusters":
            int(
                df[cluster_col].nunique()
            ),

        "observed_statistic":
            observed_chi2,

        "effect_size":
            observed_v,

        "permutation_extreme":
            extreme,

        "permutation_p":
            float(p_perm),

        "null_mean":
            float(np.mean(null)),

        "null_sd":
            float(np.std(null, ddof=1)),

        "null_2.5_percentile":
            float(np.percentile(null, 2.5)),

        "null_97.5_percentile":
            float(np.percentile(null, 97.5)),

        "observed_table":
            observed_table,

        "null":
            null,
    }


# ==================================================================================================
# 12. DESCRIPTIVE HELPERS
# ==================================================================================================

def numeric_descriptives(
    dataframe,
    endpoint,
    cluster_col
):

    rows = []

    for cluster_id, g in dataframe.groupby(
        cluster_col,
        sort=True
    ):

        x = pd.to_numeric(
            g[endpoint],
            errors="coerce"
        ).dropna()

        rows.append({

            "endpoint":
                endpoint,

            "cluster":
                int(cluster_id),

            "N":
                int(len(x)),

            "mean":
                float(x.mean())
                if len(x) else np.nan,

            "SD":
                float(x.std(ddof=1))
                if len(x) > 1 else np.nan,

            "median":
                float(x.median())
                if len(x) else np.nan,

            "Q1":
                float(x.quantile(0.25))
                if len(x) else np.nan,

            "Q3":
                float(x.quantile(0.75))
                if len(x) else np.nan,

            "minimum":
                float(x.min())
                if len(x) else np.nan,

            "maximum":
                float(x.max())
                if len(x) else np.nan,
        })

    return rows


# ==================================================================================================
# 13. EXECUTE PRIMARY DIAGNOSIS TEST
# ==================================================================================================

results = []
null_rows = []
numeric_desc_rows = []
categorical_tables = {}

CLUSTER = "EEG_PHENO_cluster"


print("\n" + "=" * 120)
print("PRIMARY CLINICAL ENDPOINT")
print("=" * 120)

rng = np.random.default_rng(
    PERMUTATION_SEED
)

diag_res = permutation_chi_square(
    analysis,
    "diagnosis_group_standardized",
    CLUSTER,
    rng,
    N_PERMUTATIONS
)

results.append({

    "family":
        "PRIMARY_CLINICAL_SINGLE_ENDPOINT",

    "endpoint":
        "diagnosis_group_standardized",

    "population":
        "ALL_N85",

    "test":
        "permutation_Pearson_chi_square",

    "effect_size_name":
        "Cramers_V",

    **{
        k: v
        for k, v in diag_res.items()
        if k not in {
            "null",
            "observed_table"
        }
    },
})

categorical_tables[
    "diagnosis_group_standardized"
] = diag_res["observed_table"]

for i, val in enumerate(
    diag_res["null"],
    1
):

    null_rows.append({

        "family":
            "PRIMARY_CLINICAL_SINGLE_ENDPOINT",

        "endpoint":
            "diagnosis_group_standardized",

        "iteration":
            i,

        "null_statistic":
            float(val),
    })


print("Observed chi-square =", diag_res["observed_statistic"])
print("Cramer's V         =", diag_res["effect_size"])
print("Permutation p      =", diag_res["permutation_p"])


# ==================================================================================================
# 14. EXECUTE TRANSDIAGNOSTIC HELD-OUT FAMILY
# ==================================================================================================

TRANSDIAGNOSTIC_SPECS = [

    ("age", "kw"),
    ("gender", "chi"),
    ("education", "kw"),
    ("avg_rt_oddb_CP", "kw"),
    ("n_oddb_CP", "kw"),
    ("n_oddb_FP", "kw"),
    ("n_oddb_CN", "kw"),
    ("n_oddb_FN", "kw"),
]


print("\n" + "=" * 120)
print("TRANSDIAGNOSTIC HELD-OUT FAMILY")
print("=" * 120)

for endpoint_index, (endpoint, test_type) in enumerate(
    TRANSDIAGNOSTIC_SPECS
):

    # Deterministic independent RNG stream per endpoint.
    rng = np.random.default_rng(
        PERMUTATION_SEED + 100 + endpoint_index
    )

    if test_type == "kw":

        res = permutation_kw(
            analysis,
            endpoint,
            CLUSTER,
            rng,
            N_PERMUTATIONS
        )

        effect_name = "epsilon_squared"
        test_name = "permutation_Kruskal_Wallis"

        numeric_desc_rows.extend(
            numeric_descriptives(
                analysis,
                endpoint,
                CLUSTER
            )
        )

    else:

        res = permutation_chi_square(
            analysis,
            endpoint,
            CLUSTER,
            rng,
            N_PERMUTATIONS
        )

        effect_name = "Cramers_V"
        test_name = "permutation_Pearson_chi_square"

        categorical_tables[
            endpoint
        ] = res["observed_table"]

    results.append({

        "family":
            "TRANSDIAGNOSTIC_HELDOUT_FAMILY",

        "endpoint":
            endpoint,

        "population":
            "ALL_N85",

        "test":
            test_name,

        "effect_size_name":
            effect_name,

        **{
            k: v
            for k, v in res.items()
            if k not in {
                "null",
                "observed_table"
            }
        },
    })

    for i, val in enumerate(
        res["null"],
        1
    ):

        null_rows.append({

            "family":
                "TRANSDIAGNOSTIC_HELDOUT_FAMILY",

            "endpoint":
                endpoint,

            "iteration":
                i,

            "null_statistic":
                float(val),
        })

    print(
        f"{endpoint:20s} "
        f"stat={res['observed_statistic']:.6f} "
        f"effect={res['effect_size']:.6f} "
        f"p={res['permutation_p']:.8f}"
    )


# ==================================================================================================
# 15. EXECUTE DIAGNOSIS-SPECIFIC SECONDARY FAMILY
# ==================================================================================================

SECONDARY_SPECS = [

    (
        "ADHD_pre_Att_leading",
        "ADHD_ADD"
    ),

    (
        "ADHD_pre_Hyp_leading",
        "ADHD_ADD"
    ),

    (
        "BDI_pre",
        "MDD"
    ),
]


print("\n" + "=" * 120)
print("DIAGNOSIS-SPECIFIC SECONDARY FAMILY")
print("=" * 120)

for endpoint_index, (
    endpoint,
    diagnosis
) in enumerate(
    SECONDARY_SPECS
):

    subset = analysis[
        analysis[
            "diagnosis_group_standardized"
        ] == diagnosis
    ].copy()

    rng = np.random.default_rng(
        PERMUTATION_SEED + 200 + endpoint_index
    )

    res = permutation_kw(
        subset,
        endpoint,
        CLUSTER,
        rng,
        N_PERMUTATIONS
    )

    results.append({

        "family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "endpoint":
            endpoint,

        "population":
            f"{diagnosis}_ONLY",

        "test":
            "permutation_Kruskal_Wallis",

        "effect_size_name":
            "epsilon_squared",

        **{
            k: v
            for k, v in res.items()
            if k != "null"
        },
    })

    desc = numeric_descriptives(
        subset,
        endpoint,
        CLUSTER
    )

    for row in desc:
        row["population"] = f"{diagnosis}_ONLY"

    numeric_desc_rows.extend(
        desc
    )

    for i, val in enumerate(
        res["null"],
        1
    ):

        null_rows.append({

            "family":
                "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

            "endpoint":
                endpoint,

            "iteration":
                i,

            "null_statistic":
                float(val),
        })

    print(
        f"{endpoint:25s} "
        f"subset={diagnosis:8s} "
        f"N={res['N']:2d} "
        f"stat={res['observed_statistic']:.6f} "
        f"effect={res['effect_size']:.6f} "
        f"p={res['permutation_p']:.8f}"
    )


# ==================================================================================================
# 16. RESULTS DATAFRAME
# ==================================================================================================

results_df = pd.DataFrame(
    results
)

# JSON-safe cluster counts
if "cluster_N" in results_df.columns:

    results_df["cluster_N"] = (
        results_df["cluster_N"]
        .apply(
            lambda x:
                json.dumps(x, sort_keys=True)
                if isinstance(x, dict)
                else x
        )
    )


# ==================================================================================================
# 17. APPLY PRE-FROZEN MULTIPLICITY CORRECTION
# ==================================================================================================

results_df["adjustment"] = ""
results_df["p_adjusted"] = np.nan
results_df["significant_q05"] = False


# PRIMARY — one endpoint, no correction
mask_primary = (
    results_df["family"]
    ==
    "PRIMARY_CLINICAL_SINGLE_ENDPOINT"
)

results_df.loc[
    mask_primary,
    "adjustment"
] = "none_single_prespecified_endpoint"

results_df.loc[
    mask_primary,
    "p_adjusted"
] = results_df.loc[
    mask_primary,
    "permutation_p"
]

results_df.loc[
    mask_primary,
    "significant_q05"
] = (
    results_df.loc[
        mask_primary,
        "permutation_p"
    ]
    < 0.05
)


# TRANSDIAGNOSTIC FAMILY — BH across exactly 8
mask_trans = (
    results_df["family"]
    ==
    "TRANSDIAGNOSTIC_HELDOUT_FAMILY"
)

trans_p = results_df.loc[
    mask_trans,
    "permutation_p"
].to_numpy()

if len(trans_p) != 8:

    raise RuntimeError(
        f"Expected exactly 8 transdiagnostic tests, found {len(trans_p)}."
    )

trans_adj = bh_fdr(
    trans_p
)

results_df.loc[
    mask_trans,
    "adjustment"
] = "Benjamini-Hochberg_FDR_q0.05"

results_df.loc[
    mask_trans,
    "p_adjusted"
] = trans_adj

results_df.loc[
    mask_trans,
    "significant_q05"
] = (
    trans_adj <= FDR_Q
)


# SECONDARY DIAGNOSIS-SPECIFIC — BH across exactly 3 feasible tests
mask_secondary = (
    results_df["family"]
    ==
    "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY"
)

secondary_p = results_df.loc[
    mask_secondary,
    "permutation_p"
].to_numpy()

if len(secondary_p) != 3:

    raise RuntimeError(
        f"Expected exactly 3 feasible secondary tests, found {len(secondary_p)}."
    )

secondary_adj = bh_fdr(
    secondary_p
)

results_df.loc[
    mask_secondary,
    "adjustment"
] = "Benjamini-Hochberg_FDR_q0.05_among_3_feasible_tests"

results_df.loc[
    mask_secondary,
    "p_adjusted"
] = secondary_adj

results_df.loc[
    mask_secondary,
    "significant_q05"
] = (
    secondary_adj <= FDR_Q
)


# ==================================================================================================
# 18. YBOCS + FALSE-POSITIVE RT DESCRIPTIVES ONLY
# ==================================================================================================

descriptive_only_rows = []

# YBOCS only within OCD, but NO inference.
ocd = analysis[
    analysis["diagnosis_group_standardized"]
    == "OCD"
].copy()

for row in numeric_descriptives(
    ocd,
    "YBOCS_pre",
    CLUSTER
):

    row["population"] = "OCD_ONLY"
    row["reason"] = (
        "Failed Stage4B diagnosis-specific feasibility rule."
    )

    descriptive_only_rows.append(
        row
    )


# avg_rt_oddb_FP full cohort descriptive only
for row in numeric_descriptives(
    analysis,
    "avg_rt_oddb_FP",
    CLUSTER
):

    row["population"] = "ALL_N85"
    row["reason"] = (
        "Low and uneven availability; inferential testing prohibited by Stage4B."
    )

    descriptive_only_rows.append(
        row
    )


descriptive_only_df = pd.DataFrame(
    descriptive_only_rows
)


# ==================================================================================================
# 19. NUMERIC DESCRIPTIVES
# ==================================================================================================

numeric_descriptives_df = pd.DataFrame(
    numeric_desc_rows
)

if "population" not in numeric_descriptives_df.columns:
    numeric_descriptives_df["population"] = "ALL_N85"

numeric_descriptives_df["population"] = (
    numeric_descriptives_df["population"]
    .fillna("ALL_N85")
)


# ==================================================================================================
# 20. SAVE CATEGORICAL TABLES
# ==================================================================================================

categorical_long_rows = []

for endpoint, table in categorical_tables.items():

    for cluster_id in table.index:

        for category in table.columns:

            categorical_long_rows.append({

                "endpoint":
                    endpoint,

                "cluster":
                    cluster_id,

                "category":
                    category,

                "count":
                    int(
                        table.loc[
                            cluster_id,
                            category
                        ]
                    ),
            })

categorical_long_df = pd.DataFrame(
    categorical_long_rows
)


# ==================================================================================================
# 21. NULL DISTRIBUTIONS
# ==================================================================================================

null_df = pd.DataFrame(
    null_rows
)


# ==================================================================================================
# 22. SAVE OUTPUTS
# ==================================================================================================

RESULTS_FILE = (
    CHAR_DIR
    / "STAGE4C_HELDOUT_ASSOCIATION_RESULTS.csv"
)

NUMERIC_DESC_FILE = (
    CHAR_DIR
    / "STAGE4C_NUMERIC_CLUSTER_DESCRIPTIVES.csv"
)

CATEGORICAL_FILE = (
    CHAR_DIR
    / "STAGE4C_CATEGORICAL_CONTINGENCY_COUNTS.csv"
)

DESCRIPTIVE_ONLY_FILE = (
    CHAR_DIR
    / "STAGE4C_DESCRIPTIVE_ONLY_ENDPOINTS.csv"
)

NULL_FILE = (
    CHAR_DIR
    / "STAGE4C_10000_PERMUTATION_NULLS.csv"
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

numeric_descriptives_df.to_csv(
    NUMERIC_DESC_FILE,
    index=False
)

categorical_long_df.to_csv(
    CATEGORICAL_FILE,
    index=False
)

descriptive_only_df.to_csv(
    DESCRIPTIVE_ONLY_FILE,
    index=False
)

null_df.to_csv(
    NULL_FILE,
    index=False
)


# ==================================================================================================
# 23. SUMMARY JSON
# ==================================================================================================

SUMMARY_FILE = (
    CHAR_DIR
    / "STAGE4C_HELDOUT_ASSOCIATION_SUMMARY.json"
)

summary = {

    "stage":
        "STAGE4C",

    "status":
        "PASS",

    "N":
        85,

    "K":
        6,

    "permutations_per_test":
        N_PERMUTATIONS,

    "permutation_seed_base":
        PERMUTATION_SEED,

    "number_primary_tests":
        int(mask_primary.sum()),

    "number_transdiagnostic_tests":
        int(mask_trans.sum()),

    "number_secondary_tests":
        int(mask_secondary.sum()),

    "primary_significant":
        int(
            results_df.loc[
                mask_primary,
                "significant_q05"
            ].sum()
        ),

    "transdiagnostic_FDR_significant":
        int(
            results_df.loc[
                mask_trans,
                "significant_q05"
            ].sum()
        ),

    "secondary_FDR_significant":
        int(
            results_df.loc[
                mask_secondary,
                "significant_q05"
            ].sum()
        ),

    "YBOCS_inferential_test_run":
        False,

    "avg_rt_oddb_FP_inferential_test_run":
        False,

    "NEO_inferential_validation_test_run":
        False,

    "posthoc_tests_run":
        False,

    "cluster_names_assigned":
        False,

    "K_changed":
        False,

    "post_treatment_variables_used":
        False,

    "implementation_lock_sha256":
        IMPLEMENTATION_HASH,

    "results_sha256":
        sha256_file(RESULTS_FILE),

    "numeric_descriptives_sha256":
        sha256_file(NUMERIC_DESC_FILE),

    "categorical_counts_sha256":
        sha256_file(CATEGORICAL_FILE),

    "descriptive_only_sha256":
        sha256_file(DESCRIPTIVE_ONLY_FILE),

    "permutation_nulls_sha256":
        sha256_file(NULL_FILE),
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

SUMMARY_HASH = sha256_file(
    SUMMARY_FILE
)


# ==================================================================================================
# 24. TEXT REPORT
# ==================================================================================================

REPORT_FILE = (
    CHAR_DIR
    / "STAGE4C_HELDOUT_ASSOCIATION_REPORT.txt"
)

report = []

report.append(
    "TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3"
)

report.append(
    "STAGE 4C — FROZEN HELD-OUT ASSOCIATION TESTING"
)

report.append(
    "=" * 110
)

report.append("")
report.append(
    results_df.to_string(index=False)
)

report.append("")
report.append(
    "YBOCS_pre: DESCRIPTIVE ONLY."
)

report.append(
    "avg_rt_oddb_FP: DESCRIPTIVE ONLY."
)

report.append(
    "NEO-FFI: MODEL INPUT; NOT INDEPENDENT VALIDATION."
)

report.append(
    "NO POST-HOC TESTS WERE PERFORMED."
)

report.append(
    "NO CLUSTER NAMES WERE ASSIGNED."
)

report.append(
    "K REMAINED FROZEN AT 6."
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
# 25. OUTPUT LOCK
# ==================================================================================================

OUTPUT_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4C_HELDOUT_ASSOCIATION_OUTPUT_LOCK.json"
)

output_lock = {

    "stage":
        "STAGE4C",

    "status":
        "PASS",

    "stage4b_design_sha256":
        EXPECTED_HASHES["stage4b_design"],

    "stage4b_feasibility_sha256":
        EXPECTED_HASHES["stage4b_feasibility"],

    "implementation_lock": {
        "path":
            str(IMPLEMENTATION_LOCK_FILE),

        "sha256":
            IMPLEMENTATION_HASH,
    },

    "results": {
        "path":
            str(RESULTS_FILE),

        "sha256":
            sha256_file(RESULTS_FILE),
    },

    "numeric_descriptives": {
        "path":
            str(NUMERIC_DESC_FILE),

        "sha256":
            sha256_file(NUMERIC_DESC_FILE),
    },

    "categorical_counts": {
        "path":
            str(CATEGORICAL_FILE),

        "sha256":
            sha256_file(CATEGORICAL_FILE),
    },

    "descriptive_only": {
        "path":
            str(DESCRIPTIVE_ONLY_FILE),

        "sha256":
            sha256_file(DESCRIPTIVE_ONLY_FILE),
    },

    "permutation_nulls": {
        "path":
            str(NULL_FILE),

        "sha256":
            sha256_file(NULL_FILE),
    },

    "summary": {
        "path":
            str(SUMMARY_FILE),

        "sha256":
            SUMMARY_HASH,
    },

    "report": {
        "path":
            str(REPORT_FILE),

        "sha256":
            sha256_file(REPORT_FILE),
    },

    "K_changed":
        False,

    "cluster_names_assigned":
        False,

    "posthoc_tests_run":
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

OUTPUT_LOCK_HASH = sha256_file(
    OUTPUT_LOCK_FILE
)


# ==================================================================================================
# 26. PRINT FINAL RESULTS
# ==================================================================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


print("\n" + "=" * 120)
print("STAGE 4C — FINAL INFERENTIAL RESULTS")
print("=" * 120)

display_columns = [

    "family",
    "endpoint",
    "population",
    "N",
    "test",
    "observed_statistic",
    "effect_size_name",
    "effect_size",
    "permutation_p",
    "adjustment",
    "p_adjusted",
    "significant_q05",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("PRIMARY DIAGNOSIS CONTINGENCY TABLE")
print("=" * 120)

print(
    categorical_tables[
        "diagnosis_group_standardized"
    ].to_string()
)


print("\n" + "=" * 120)
print("DESCRIPTIVE-ONLY ENDPOINTS")
print("=" * 120)

print(
    descriptive_only_df.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("HASHES")
print("=" * 120)

print("Implementation lock:")
print(IMPLEMENTATION_HASH)

print("\nInferential results:")
print(sha256_file(RESULTS_FILE))

print("\nNumeric descriptives:")
print(sha256_file(NUMERIC_DESC_FILE))

print("\nCategorical counts:")
print(sha256_file(CATEGORICAL_FILE))

print("\nDescriptive-only endpoints:")
print(sha256_file(DESCRIPTIVE_ONLY_FILE))

print("\nPermutation nulls:")
print(sha256_file(NULL_FILE))

print("\nSummary:")
print(SUMMARY_HASH)

print("\nReport:")
print(sha256_file(REPORT_FILE))

print("\nOutput lock:")
print(OUTPUT_LOCK_HASH)


print("\n" + "=" * 120)
print("FINAL STATUS")
print("=" * 120)

print("""
STAGE 4C COMPLETE IF NO ERROR APPEARED ABOVE.

IMPORTANT:

- K remained frozen at 6.
- Authoritative clusters remained numeric 0-5.
- No clinical variable influenced model selection.
- Diagnosis was tested only after the partition was frozen.
- The eight prespecified whole-cohort endpoints used one BH-FDR family.
- Only the three Stage4B-feasible diagnosis-specific symptom tests were included
  in the secondary BH-FDR family.
- YBOCS_pre was NOT tested inferentially.
- avg_rt_oddb_FP was NOT tested inferentially.
- NEO-FFI was NOT treated as independent validation.
- No post-hoc pairwise tests were performed.
- No post-treatment variable was used.
- No cluster was clinically named.

NEXT PIPELINE STAGE:
Post-hoc analysis, if required, follows a separately frozen statistical plan.
""")