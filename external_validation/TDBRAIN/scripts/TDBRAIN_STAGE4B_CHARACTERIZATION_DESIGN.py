# ==================================================================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# STAGE 4B — STATISTICAL CHARACTERIZATION DESIGN + FEASIBILITY LOCK
# ==================================================================================================
#
# PURPOSE:
#   Freeze the statistical characterization plan BEFORE any clinical p-value is calculated.
#
# THIS STAGE:
#   - verifies Stage4A-R1 frozen outputs
#   - defines primary / secondary endpoint families
#   - defines statistical tests
#   - defines permutation count and random seed
#   - defines effect sizes
#   - defines FDR families
#   - checks sample-size / cluster-coverage feasibility
#
# THIS STAGE DOES NOT:
#   - calculate p-values
#   - calculate test statistics from clinical values
#   - rename clusters
#   - change K
#   - use NEO-FFI as independent validation
#   - use post-treatment variables
# ==================================================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json

print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 4B — STATISTICAL CHARACTERIZATION DESIGN + FEASIBILITY LOCK")
print("=" * 120)


# ==================================================================================================
# 1. ROOT
# ==================================================================================================

ROOT = Path(
    "/content/drive/MyDrive/"
    "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION"
)

LOCK_DIR = ROOT / "01_LOCK"
CHAR_DIR = ROOT / "08_CHARACTERIZATION"

if not ROOT.exists():
    raise RuntimeError(f"Experiment-3 root missing:\n{ROOT}")


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
# 3. VERIFY STAGE4A-R1 FROZEN OUTPUTS
# ==================================================================================================

STAGE4A_FILES = {

    "design_lock":
        LOCK_DIR / "STAGE4A_R1_CHARACTERIZATION_PREANALYSIS_LOCK.json",

    "analysis_table":
        CHAR_DIR / "STAGE4A_R1_BASELINE_HELDOUT_AUDIT_TABLE.csv",

    "overall_availability":
        CHAR_DIR / "STAGE4A_R1_VARIABLE_AVAILABILITY.csv",

    "diagnosis_availability":
        CHAR_DIR / "STAGE4A_R1_DIAGNOSIS_SPECIFIC_AVAILABILITY.csv",

    "cluster_availability":
        CHAR_DIR / "STAGE4A_R1_CLUSTER_VARIABLE_AVAILABILITY.csv",

    "cluster_diagnosis":
        CHAR_DIR / "STAGE4A_R1_CLUSTER_DIAGNOSIS_COUNTS.csv",

    "summary":
        CHAR_DIR / "STAGE4A_R1_AVAILABILITY_AUDIT_SUMMARY.json",

    "output_lock":
        LOCK_DIR / "STAGE4A_R1_AVAILABILITY_AUDIT_LOCK.json",
}


EXPECTED_STAGE4A_HASHES = {

    "design_lock":
        "53d78f30332dc671796aa443cdc57aa9e020d5d266bb364cb124c70668f0db00",

    "analysis_table":
        "580854ea8ef6ebe8ff6ae5ef8a31ce9cae3e53dfba07bc0f4f2da24515efcddc",

    "overall_availability":
        "baa8dcf43f351da4f13a90cc89a01870ba10045543c334a877732f0aa2c28323",

    "diagnosis_availability":
        "e8abdf48f2c9d3afac45f6faaf70e4d27e91ab764d0e02a474e06ebd7bbf0543",

    "cluster_availability":
        "7d5e0ec094e8e5265d898059d1afa91a60a6298a313f9f2cbe3956078601d45f",

    "cluster_diagnosis":
        "4364ac629c24a357cd9e05bfc3bd4b1a71019e3cd5ca3e1679c91aeabe8c16e4",

    "summary":
        "b21345007eb87f7072fe9536a7f1a93738df369b667b040be65f3dae4c9df1df",

    "output_lock":
        "7664d9fd106575ca3e2aa8dea39b05a11e4e7b653a9def04e5b1c6611033717d",
}


print("\n" + "=" * 120)
print("STAGE4A-R1 FROZEN INPUT VERIFICATION")
print("=" * 120)

for name, path in STAGE4A_FILES.items():

    if not path.exists():
        raise RuntimeError(
            f"Missing Stage4A-R1 frozen file:\n{name}\n{path}"
        )

    actual = sha256_file(path)
    expected = EXPECTED_STAGE4A_HASHES[name]

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
# 4. LOAD STAGE4A-R1 AUDIT TABLE
# ==================================================================================================

analysis = pd.read_csv(
    STAGE4A_FILES["analysis_table"]
)

required_columns = [

    "subject_key",
    "EEG_PHENO_cluster",
    "diagnosis_group_standardized",

    "age",
    "gender",
    "education",

    "ADHD_pre_Att_leading",
    "ADHD_pre_Hyp_leading",
    "YBOCS_pre",
    "BDI_pre",

    "avg_rt_oddb_CP",
    "avg_rt_oddb_FP",

    "n_oddb_CP",
    "n_oddb_FP",
    "n_oddb_CN",
    "n_oddb_FN",
]

missing_columns = [
    c for c in required_columns
    if c not in analysis.columns
]

if missing_columns:

    raise RuntimeError(
        "Required Stage4A columns missing:\n"
        + "\n".join(missing_columns)
    )


# ==================================================================================================
# 5. VERIFY AUTHORITATIVE STATE AGAIN
# ==================================================================================================

if len(analysis) != 85:

    raise RuntimeError(
        f"Unexpected N={len(analysis)}, expected 85."
    )

if analysis["subject_key"].duplicated().any():

    raise RuntimeError(
        "Duplicate subjects in Stage4A audit table."
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

observed_cluster_sizes = {
    int(k): int(v)
    for k, v in
    analysis["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
    .items()
}

if observed_cluster_sizes != expected_cluster_sizes:

    raise RuntimeError(
        "Authoritative cluster-size mismatch."
    )

expected_diagnosis_counts = {
    "MDD": 31,
    "ADHD_ADD": 28,
    "OCD": 26,
}

observed_diagnosis_counts = {
    str(k): int(v)
    for k, v in
    analysis["diagnosis_group_standardized"]
    .value_counts()
    .items()
}

if observed_diagnosis_counts != expected_diagnosis_counts:

    raise RuntimeError(
        "Frozen diagnosis-count mismatch."
    )

print("\nPASS — authoritative N=85, K=6 partition and diagnosis bookkeeping verified.")


# ==================================================================================================
# 6. FREEZE GLOBAL INFERENTIAL SETTINGS
# ==================================================================================================

N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20261009

OMNIBUS_ALPHA = 0.05
FDR_Q = 0.05

# No pairwise post-hoc testing is authorized in Stage4C.
ALLOW_POSTHOC = False


# ==================================================================================================
# 7. DEFINE PRE-SPECIFIED ENDPOINT FAMILIES
# ==================================================================================================

# ----------------------------------------------------------------------------------
# FAMILY A
# Single primary clinical characterization endpoint.
# ----------------------------------------------------------------------------------

PRIMARY_CLINICAL = [

    {
        "endpoint":
            "diagnosis_group_standardized",

        "analysis_population":
            "all_authoritative_subjects",

        "variable_type":
            "categorical",

        "test":
            "permutation_Pearson_chi_square",

        "effect_size":
            "Cramers_V",

        "multiplicity_family":
            "PRIMARY_CLINICAL_SINGLE_ENDPOINT",

        "role":
            "held-out clinical characterization; not model selection and not external validation",
    }
]


# ----------------------------------------------------------------------------------
# FAMILY B
# Held-out whole-cohort behavioral / demographic characterization.
# ----------------------------------------------------------------------------------

TRANSDIAGNOSTIC_ENDPOINTS = [

    {
        "endpoint": "age",
        "variable_type": "continuous",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "gender",
        "variable_type": "categorical",
        "test": "permutation_Pearson_chi_square",
        "effect_size": "Cramers_V",
    },

    {
        "endpoint": "education",
        "variable_type": "ordinal_or_continuous",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "avg_rt_oddb_CP",
        "variable_type": "continuous",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "n_oddb_CP",
        "variable_type": "count",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "n_oddb_FP",
        "variable_type": "count",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "n_oddb_CN",
        "variable_type": "count",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },

    {
        "endpoint": "n_oddb_FN",
        "variable_type": "count",
        "test": "permutation_Kruskal_Wallis",
        "effect_size": "epsilon_squared",
    },
]

for d in TRANSDIAGNOSTIC_ENDPOINTS:

    d["analysis_population"] = "all_authoritative_subjects"
    d["multiplicity_family"] = "TRANSDIAGNOSTIC_HELDOUT_FAMILY"
    d["role"] = "held-out behavioral/demographic characterization"


# ----------------------------------------------------------------------------------
# FAMILY C
# Diagnosis-specific symptom analyses.
#
# We do NOT pool these variables across diagnoses because Stage4A showed
# structural diagnosis-dependent availability.
# ----------------------------------------------------------------------------------

DIAGNOSIS_SPECIFIC_ENDPOINTS = [

    {
        "endpoint":
            "ADHD_pre_Att_leading",

        "diagnosis_subset":
            "ADHD_ADD",

        "variable_type":
            "continuous_or_ordinal",

        "test":
            "permutation_Kruskal_Wallis",

        "effect_size":
            "epsilon_squared",

        "multiplicity_family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "role":
            "secondary exploratory within-diagnosis characterization",
    },

    {
        "endpoint":
            "ADHD_pre_Hyp_leading",

        "diagnosis_subset":
            "ADHD_ADD",

        "variable_type":
            "continuous_or_ordinal",

        "test":
            "permutation_Kruskal_Wallis",

        "effect_size":
            "epsilon_squared",

        "multiplicity_family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "role":
            "secondary exploratory within-diagnosis characterization",
    },

    {
        "endpoint":
            "YBOCS_pre",

        "diagnosis_subset":
            "OCD",

        "variable_type":
            "continuous_or_ordinal",

        "test":
            "permutation_Kruskal_Wallis",

        "effect_size":
            "epsilon_squared",

        "multiplicity_family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "role":
            "secondary exploratory within-diagnosis characterization",
    },

    {
        "endpoint":
            "BDI_pre",

        "diagnosis_subset":
            "MDD",

        "variable_type":
            "continuous_or_ordinal",

        "test":
            "permutation_Kruskal_Wallis",

        "effect_size":
            "epsilon_squared",

        "multiplicity_family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "role":
            "secondary exploratory within-diagnosis characterization",
    },
]


# ----------------------------------------------------------------------------------
# DESCRIPTIVE ONLY
# ----------------------------------------------------------------------------------

DESCRIPTIVE_ONLY = [

    {
        "endpoint":
            "avg_rt_oddb_FP",

        "reason":
            (
                "Stage4A availability only 35/85 and strongly uneven across clusters; "
                "false-positive reaction time may be undefined when no false-positive response occurs."
            ),
    },

    {
        "endpoint":
            "neoFFI_q1 ... neoFFI_q60",

        "reason":
            (
                "NEO-FFI is a MODEL INPUT in Experiment 3 and therefore cannot be "
                "treated as independent held-out validation."
            ),
    },
]


# ==================================================================================================
# 8. FEASIBILITY RULES — FIXED BEFORE TESTING VALUES
# ==================================================================================================

FEASIBILITY_RULES = {

    "primary_clinical":
        (
            "Full authoritative cohort. At least two outcome categories overall. "
            "Permutation chi-square used regardless of asymptotic expected-cell counts."
        ),

    "transdiagnostic_numeric":
        (
            "At least 80% overall availability AND at least 5 observed subjects "
            "in every authoritative cluster."
        ),

    "transdiagnostic_categorical":
        (
            "At least 80% overall availability, at least two categories overall, "
            "and at least 5 observed subjects in every authoritative cluster."
        ),

    "diagnosis_specific_symptom":
        (
            "Restrict to the prespecified diagnosis subgroup. "
            "Require at least 3 represented clusters, and every represented cluster "
            "must contain at least 3 observed scores. Otherwise descriptive only."
        ),
}


# ==================================================================================================
# 9. FEASIBILITY AUDIT — COUNTS ONLY, NO TEST STATISTICS
# ==================================================================================================

feasibility_rows = []


# --------------------------------------------------------------------------------------------------
# 9A. PRIMARY DIAGNOSIS ENDPOINT
# --------------------------------------------------------------------------------------------------

endpoint = "diagnosis_group_standardized"

s = analysis[endpoint]

cluster_counts = (
    analysis.groupby("EEG_PHENO_cluster")[endpoint]
    .apply(lambda x: int(x.notna().sum()))
    .to_dict()
)

feasible = (
    s.notna().sum() == 85
    and s.nunique(dropna=True) >= 2
)

feasibility_rows.append({

    "family":
        "PRIMARY_CLINICAL_SINGLE_ENDPOINT",

    "endpoint":
        endpoint,

    "population":
        "ALL_N85",

    "available_N":
        int(s.notna().sum()),

    "available_percent":
        float(100 * s.notna().mean()),

    "represented_clusters":
        int(
            sum(v > 0 for v in cluster_counts.values())
        ),

    "minimum_observed_per_represented_cluster":
        int(
            min(v for v in cluster_counts.values() if v > 0)
        ),

    "cluster_observed_counts":
        json.dumps(
            {
                str(k): int(v)
                for k, v in cluster_counts.items()
            },
            sort_keys=True
        ),

    "planned_test":
        "permutation_Pearson_chi_square",

    "effect_size":
        "Cramers_V",

    "feasible":
        bool(feasible),

    "decision":
        "INFERENTIAL" if feasible else "DESCRIPTIVE_ONLY",
})


# --------------------------------------------------------------------------------------------------
# 9B. TRANSDIAGNOSTIC HELD-OUT ENDPOINTS
# --------------------------------------------------------------------------------------------------

for spec in TRANSDIAGNOSTIC_ENDPOINTS:

    endpoint = spec["endpoint"]

    s = analysis[endpoint]

    cluster_counts = (
        analysis.groupby("EEG_PHENO_cluster")[endpoint]
        .apply(lambda x: int(x.notna().sum()))
        .to_dict()
    )

    availability = float(s.notna().mean())

    minimum_cluster_n = min(
        cluster_counts.values()
    )

    if spec["variable_type"] == "categorical":

        feasible = (
            availability >= 0.80
            and minimum_cluster_n >= 5
            and s.nunique(dropna=True) >= 2
        )

    else:

        feasible = (
            availability >= 0.80
            and minimum_cluster_n >= 5
        )

    feasibility_rows.append({

        "family":
            "TRANSDIAGNOSTIC_HELDOUT_FAMILY",

        "endpoint":
            endpoint,

        "population":
            "ALL_N85",

        "available_N":
            int(s.notna().sum()),

        "available_percent":
            float(100 * availability),

        "represented_clusters":
            int(
                sum(v > 0 for v in cluster_counts.values())
            ),

        "minimum_observed_per_represented_cluster":
            int(
                min(v for v in cluster_counts.values() if v > 0)
            ),

        "cluster_observed_counts":
            json.dumps(
                {
                    str(k): int(v)
                    for k, v in cluster_counts.items()
                },
                sort_keys=True
            ),

        "planned_test":
            spec["test"],

        "effect_size":
            spec["effect_size"],

        "feasible":
            bool(feasible),

        "decision":
            "INFERENTIAL" if feasible else "DESCRIPTIVE_ONLY",
    })


# --------------------------------------------------------------------------------------------------
# 9C. DIAGNOSIS-SPECIFIC SYMPTOM ENDPOINTS
# --------------------------------------------------------------------------------------------------

for spec in DIAGNOSIS_SPECIFIC_ENDPOINTS:

    endpoint = spec["endpoint"]
    diagnosis = spec["diagnosis_subset"]

    subset = analysis[
        analysis["diagnosis_group_standardized"] == diagnosis
    ].copy()

    subset = subset[
        subset[endpoint].notna()
    ].copy()

    cluster_counts_series = (
        subset["EEG_PHENO_cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_counts = {
        int(k): int(v)
        for k, v in cluster_counts_series.items()
    }

    represented_clusters = len(
        cluster_counts
    )

    if represented_clusters > 0:

        min_represented_n = min(
            cluster_counts.values()
        )

    else:

        min_represented_n = 0

    feasible = (
        represented_clusters >= 3
        and min_represented_n >= 3
    )

    feasibility_rows.append({

        "family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "endpoint":
            endpoint,

        "population":
            f"{diagnosis}_ONLY",

        "available_N":
            int(len(subset)),

        "available_percent":
            float(
                100
                * len(subset)
                / max(
                    1,
                    int(
                        (
                            analysis["diagnosis_group_standardized"]
                            == diagnosis
                        ).sum()
                    )
                )
            ),

        "represented_clusters":
            int(represented_clusters),

        "minimum_observed_per_represented_cluster":
            int(min_represented_n),

        "cluster_observed_counts":
            json.dumps(
                {
                    str(k): int(v)
                    for k, v in cluster_counts.items()
                },
                sort_keys=True
            ),

        "planned_test":
            spec["test"],

        "effect_size":
            spec["effect_size"],

        "feasible":
            bool(feasible),

        "decision":
            "INFERENTIAL_SECONDARY"
            if feasible
            else "DESCRIPTIVE_ONLY",
    })


feasibility = pd.DataFrame(
    feasibility_rows
)


# ==================================================================================================
# 10. MULTIPLICITY PLAN
# ==================================================================================================

multiplicity = pd.DataFrame([

    {
        "family":
            "PRIMARY_CLINICAL_SINGLE_ENDPOINT",

        "endpoints":
            "diagnosis_group_standardized",

        "correction":
            "none_single_prespecified_endpoint",

        "threshold":
            0.05,

        "interpretation":
            "held-out clinical characterization; not external validation",
    },

    {
        "family":
            "TRANSDIAGNOSTIC_HELDOUT_FAMILY",

        "endpoints":
            (
                "age; gender; education; avg_rt_oddb_CP; "
                "n_oddb_CP; n_oddb_FP; n_oddb_CN; n_oddb_FN"
            ),

        "correction":
            "Benjamini-Hochberg FDR",

        "threshold":
            FDR_Q,

        "interpretation":
            "held-out behavioral/demographic characterization",
    },

    {
        "family":
            "DIAGNOSIS_SPECIFIC_SYMPTOM_FAMILY",

        "endpoints":
            (
                "ADHD_pre_Att_leading within ADHD_ADD; "
                "ADHD_pre_Hyp_leading within ADHD_ADD; "
                "YBOCS_pre within OCD; "
                "BDI_pre within MDD"
            ),

        "correction":
            "Benjamini-Hochberg FDR among feasible prespecified tests",

        "threshold":
            FDR_Q,

        "interpretation":
            "secondary exploratory within-diagnosis characterization",
    },
])


# ==================================================================================================
# 11. COMPLETE FROZEN DESIGN
# ==================================================================================================

design = {

    "stage":
        "STAGE4B_STATISTICAL_CHARACTERIZATION_DESIGN",

    "authoritative_state": {

        "N":
            85,

        "K":
            6,

        "clusters":
            [0, 1, 2, 3, 4, 5],

        "cluster_sizes":
            {
                str(k): int(v)
                for k, v in expected_cluster_sizes.items()
            },
    },

    "global_settings": {

        "permutations":
            N_PERMUTATIONS,

        "permutation_seed":
            PERMUTATION_SEED,

        "omnibus_alpha":
            OMNIBUS_ALPHA,

        "FDR_q":
            FDR_Q,

        "posthoc_testing_authorized":
            ALLOW_POSTHOC,
    },

    "primary_clinical_family":
        PRIMARY_CLINICAL,

    "transdiagnostic_heldout_family":
        TRANSDIAGNOSTIC_ENDPOINTS,

    "diagnosis_specific_symptom_family":
        DIAGNOSIS_SPECIFIC_ENDPOINTS,

    "descriptive_only":
        DESCRIPTIVE_ONLY,

    "feasibility_rules":
        FEASIBILITY_RULES,

    "important_interpretation_rules": [

        "Diagnosis was not used in model construction or K selection.",

        "A diagnosis association is held-out clinical characterization, not external validation.",

        "Age, gender, education and oddball variables were not used in model construction.",

        "NEO-FFI was used as a phenotype input and therefore is model-informed rather than independent validation.",

        "ADHD and YBOCS scales are structurally diagnosis-specific in this cohort and must not be pooled across all 85 subjects.",

        "BDI_pre will be tested only within the MDD subgroup because whole-cohort availability is diagnosis-dependent.",

        "avg_rt_oddb_FP is descriptive only because of low and uneven availability.",

        "No cluster will be clinically named from one significant variable.",

        "No model parameter, K value, UMAP setting or authoritative cluster assignment may be changed after clinical results are seen.",

        "No post-treatment variable is authorized.",
    ],
}


# ==================================================================================================
# 12. SAVE STAGE4B FILES
# ==================================================================================================

FEASIBILITY_FILE = (
    CHAR_DIR
    / "STAGE4B_ENDPOINT_FEASIBILITY.csv"
)

MULTIPLICITY_FILE = (
    CHAR_DIR
    / "STAGE4B_MULTIPLICITY_PLAN.csv"
)

DESIGN_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4B_STATISTICAL_CHARACTERIZATION_DESIGN_LOCK.json"
)

REPORT_FILE = (
    CHAR_DIR
    / "STAGE4B_STATISTICAL_CHARACTERIZATION_DESIGN_REPORT.txt"
)


feasibility.to_csv(
    FEASIBILITY_FILE,
    index=False
)

multiplicity.to_csv(
    MULTIPLICITY_FILE,
    index=False
)

with open(
    DESIGN_LOCK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        design,
        f,
        indent=2
    )


# ==================================================================================================
# 13. TEXT REPORT
# ==================================================================================================

report_lines = []

report_lines.append(
    "TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3"
)

report_lines.append(
    "STAGE 4B — STATISTICAL CHARACTERIZATION DESIGN + FEASIBILITY LOCK"
)

report_lines.append(
    "=" * 100
)

report_lines.append("")
report_lines.append("AUTHORITATIVE STATE")
report_lines.append("N = 85")
report_lines.append("K = 6")
report_lines.append(
    "Cluster sizes = [27, 15, 18, 10, 9, 6]"
)

report_lines.append("")
report_lines.append("GLOBAL INFERENTIAL SETTINGS")
report_lines.append(
    f"Permutation iterations = {N_PERMUTATIONS}"
)
report_lines.append(
    f"Permutation seed = {PERMUTATION_SEED}"
)
report_lines.append(
    f"Nominal alpha = {OMNIBUS_ALPHA}"
)
report_lines.append(
    f"BH-FDR q = {FDR_Q}"
)
report_lines.append(
    "Post-hoc testing = NOT AUTHORIZED"
)

report_lines.append("")
report_lines.append("FEASIBILITY")
report_lines.append(
    feasibility.to_string(index=False)
)

report_lines.append("")
report_lines.append("MULTIPLICITY PLAN")
report_lines.append(
    multiplicity.to_string(index=False)
)

report_lines.append("")
report_lines.append(
    "NO CLINICAL TEST STATISTIC OR P-VALUE WAS CALCULATED IN STAGE4B."
)

report_lines.append(
    "K REMAINS FROZEN AT 6."
)

report_lines.append(
    "CLUSTERS REMAIN NUMERIC 0-5."
)

report_lines.append(
    "POST-TREATMENT VARIABLES REMAIN SEALED."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ==================================================================================================
# 14. OUTPUT LOCK
# ==================================================================================================

OUTPUT_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4B_STATISTICAL_CHARACTERIZATION_OUTPUT_LOCK.json"
)

output_lock = {

    "stage":
        "STAGE4B",

    "status":
        "PASS",

    "stage4a_input_hashes":
        EXPECTED_STAGE4A_HASHES,

    "design_lock": {
        "path":
            str(DESIGN_LOCK_FILE),

        "sha256":
            sha256_file(DESIGN_LOCK_FILE),
    },

    "feasibility": {
        "path":
            str(FEASIBILITY_FILE),

        "sha256":
            sha256_file(FEASIBILITY_FILE),
    },

    "multiplicity_plan": {
        "path":
            str(MULTIPLICITY_FILE),

        "sha256":
            sha256_file(MULTIPLICITY_FILE),
    },

    "report": {
        "path":
            str(REPORT_FILE),

        "sha256":
            sha256_file(REPORT_FILE),
    },

    "clinical_tests_run":
        False,

    "p_values_computed":
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
# 15. PRINT RESULTS
# ==================================================================================================

print("\n" + "=" * 120)
print("ENDPOINT FEASIBILITY — COUNTS ONLY")
print("=" * 120)

print(
    feasibility.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("MULTIPLICITY PLAN")
print("=" * 120)

print(
    multiplicity.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("DESCRIPTIVE-ONLY VARIABLES")
print("=" * 120)

for d in DESCRIPTIVE_ONLY:

    print()
    print("Endpoint:", d["endpoint"])
    print("Reason  :", d["reason"])


print("\n" + "=" * 120)
print("STAGE4B HASHES")
print("=" * 120)

print("Design lock:")
print(
    sha256_file(
        DESIGN_LOCK_FILE
    )
)

print("\nEndpoint feasibility:")
print(
    sha256_file(
        FEASIBILITY_FILE
    )
)

print("\nMultiplicity plan:")
print(
    sha256_file(
        MULTIPLICITY_FILE
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


print("\n" + "=" * 120)
print("FINAL STATUS")
print("=" * 120)

print("""
STAGE 4B PASS IF NO ERROR APPEARED ABOVE.

NO CLINICAL P-VALUE WAS COMPUTED.
NO CLINICAL TEST STATISTIC WAS COMPUTED.
NO POST-HOC TEST WAS PERFORMED.

K REMAINS FROZEN AT 6.
CLUSTERS REMAIN NUMERIC 0-5.
NEO-FFI REMAINS A MODEL INPUT.
POST-TREATMENT VARIABLES REMAIN SEALED.

NEXT PIPELINE STAGE:
Stage 4C performs the prespecified held-out association tests.
Only endpoints meeting the frozen feasibility criteria are included.
""")