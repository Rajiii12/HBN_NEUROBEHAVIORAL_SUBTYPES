# ==================================================================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# STAGE 4A — HELD-OUT CHARACTERIZATION AVAILABILITY + PRE-ANALYSIS AUDIT
# CORRECTED R1
# ==================================================================================================
#
# FIX:
#   The earlier cell stopped because the Stage0B-R2 baseline file contains both
#   'subject' and 'TDBRAIN_ID'. We do NOT choose between them heuristically.
#
#   Frozen linking strategy in this R1:
#       Experiment-3 baseline       -> subject_norm
#       Experiment-3 partition      -> subject_norm
#       Original metadata workbook -> TDBRAIN_ID, normalized and cross-checked
#
#   We additionally verify that:
#       baseline subject == baseline subject_norm == baseline TDBRAIN_ID
#       partition subject == partition subject_norm
#
# IMPORTANT:
#   - NO clinical hypothesis test
#   - NO p-values
#   - NO change to K
#   - NO cluster renaming
#   - NO model retraining
#   - NO post-treatment variables
#   - NO inference of diagnosis from indication/formal_status
# ==================================================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
import re
from datetime import datetime, timezone

print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 4A-R1 — HELD-OUT CHARACTERIZATION AVAILABILITY + PRE-ANALYSIS AUDIT")
print("=" * 120)


# ==================================================================================================
# 1. EXACT VERIFIED PATHS FROM THE PREVIOUS READ-ONLY AUDITS
# ==================================================================================================

ROOT = Path(
    "/content/drive/MyDrive/"
    "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION"
)

META_XLSX = Path(
    "/content/drive/MyDrive/TDBRAIN_ADHD_EXPERIMENT1/"
    "00_LOCK/TDBRAIN_participants_V3.1.xlsx"
)

BASELINE_FILE = (
    ROOT
    / "00_METADATA_AUDIT"
    / "STAGE0B_R2_BASELINE_SESSION1_SELECTION.csv"
)

PARTITION_FILE = (
    ROOT
    / "01_LOCK"
    / "STAGE3B_AUTHORITATIVE_PRIMARY_PARTITION.csv"
)

STAGE3E_LOCK_FILE = (
    ROOT
    / "01_LOCK"
    / "STAGE3E_RESAMPLING_OUTPUT_LOCK.json"
)

META_SHEET = "TDBRAIN-V3.1"


# ==================================================================================================
# 2. FROZEN HASHES
# ==================================================================================================

EXPECTED_HASHES = {

    "metadata":
        "86cd876ea89e69e4bbaf3c9582aea9a02a07b9d3ead2fe2b9d1cc45423def3a9",

    "baseline":
        "88011cbdaab202bb2a2309468cdff171af84eabecc4b09b0fe0dc305e5632258",

    "partition":
        "f11a53d12e6676c7175622edef55984686363af051f4158b899fc321df50a782",

    "stage3e_lock":
        "9be12df8c2efbd4a64574a7e3b58d494bbf335d48fe7cca39d16d7e175188d70",
}


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
# 3. VERIFY EVERY FROZEN INPUT
# ==================================================================================================

print("\n" + "=" * 120)
print("FROZEN INPUT VERIFICATION")
print("=" * 120)

INPUTS = {
    "metadata": META_XLSX,
    "baseline": BASELINE_FILE,
    "partition": PARTITION_FILE,
    "stage3e_lock": STAGE3E_LOCK_FILE,
}

for name, path in INPUTS.items():

    if not path.exists():

        raise RuntimeError(
            f"Required frozen input missing:\n{name}\n{path}"
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
    print("     SHA256:", actual)


# ==================================================================================================
# 4. READ HEADERS ONLY
# ==================================================================================================

baseline_header = pd.read_csv(
    BASELINE_FILE,
    nrows=0
)

partition_header = pd.read_csv(
    PARTITION_FILE,
    nrows=0
)

metadata_header = pd.read_excel(
    META_XLSX,
    sheet_name=META_SHEET,
    nrows=0
)

print("\n" + "=" * 120)
print("SCHEMA VERIFICATION")
print("=" * 120)

print("\nBASELINE:")
print(list(baseline_header.columns))

print("\nPARTITION:")
print(list(partition_header.columns))

print("\nMETADATA COLUMN COUNT:")
print(len(metadata_header.columns))


# ==================================================================================================
# 5. REQUIRE THE EXACT IDENTIFIER FIELDS OBSERVED IN THE PREVIOUS RUN
# ==================================================================================================

BASELINE_REQUIRED = [
    "subject",
    "subject_norm",
    "diagnosis_group_standardized",
    "TDBRAIN_ID",
    "sessID_numeric",
]

PARTITION_REQUIRED = [
    "subject",
    "subject_norm",
    "EEG_PHENO_cluster",
    "K",
    "UMAP_seed",
]

for col in BASELINE_REQUIRED:

    if col not in baseline_header.columns:

        raise RuntimeError(
            f"Missing required frozen baseline column: {col}"
        )

for col in PARTITION_REQUIRED:

    if col not in partition_header.columns:

        raise RuntimeError(
            f"Missing required frozen partition column: {col}"
        )


print("\nPASS — exact frozen Experiment-3 identifier columns verified.")
print("Primary Experiment-3 linkage key = subject_norm")
print("Metadata linkage source = TDBRAIN_ID")
print("Cross-field identity agreement will be tested before any characterization.")


# ==================================================================================================
# 6. VERIFY AUTHORIZED METADATA VARIABLES EXIST
# ==================================================================================================

AUTHORIZED_METADATA_COLUMNS = [

    "TDBRAIN_ID",
    "sessID",

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

missing_metadata_columns = [

    c for c in AUTHORIZED_METADATA_COLUMNS
    if c not in metadata_header.columns
]

if missing_metadata_columns:

    raise RuntimeError(
        "Required authorized metadata variables missing:\n"
        + "\n".join(missing_metadata_columns)
    )

print("\nPASS — all authorized baseline metadata variables exist.")


# ==================================================================================================
# 7. CREATE / OVERWRITE STAGE4A-R1 DESIGN LOCK BEFORE READING CLINICAL VALUES
# ==================================================================================================

LOCK_DIR = ROOT / "01_LOCK"
CHAR_DIR = ROOT / "08_CHARACTERIZATION"

LOCK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHAR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DESIGN_LOCK_PATH = (
    LOCK_DIR
    / "STAGE4A_R1_CHARACTERIZATION_PREANALYSIS_LOCK.json"
)

design = {

    "stage":
        "STAGE4A_R1",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "purpose":
        "Held-out characterization availability and pre-analysis audit only.",

    "authoritative_N":
        85,

    "authoritative_K":
        6,

    "primary_linkage_key":
        "subject_norm",

    "metadata_linkage_field":
        "TDBRAIN_ID",

    "identity_validation_rule":
        (
            "Require canonical agreement among Stage0B-R2 subject, "
            "Stage0B-R2 subject_norm, Stage0B-R2 TDBRAIN_ID, "
            "Stage3B subject, Stage3B subject_norm, and metadata TDBRAIN_ID."
        ),

    "baseline_session_rule":
        "sessID == 1",

    "diagnosis_source":
        str(BASELINE_FILE),

    "diagnosis_variable":
        "diagnosis_group_standardized",

    "authorized_baseline_symptom_variables": [
        "ADHD_pre_Att_leading",
        "ADHD_pre_Hyp_leading",
        "YBOCS_pre",
        "BDI_pre",
    ],

    "authorized_oddball_variables": [
        "avg_rt_oddb_CP",
        "avg_rt_oddb_FP",
        "n_oddb_CP",
        "n_oddb_FP",
        "n_oddb_CN",
        "n_oddb_FN",
    ],

    "authorized_demographic_context": [
        "age",
        "gender",
        "education",
    ],

    "model_input_not_independent_validation":
        "neoFFI_q1 ... neoFFI_q60",

    "sealed_post_treatment_variables": [
        "Responder",
        "Remitter",
        "BDI_post",
        "ADHD_post_Att_leading",
        "ADHD_post_Hyp_leading",
        "YBOCS_post",
    ],

    "clinical_hypothesis_tests_authorized":
        False,

    "p_values_authorized":
        False,

    "cluster_renaming_authorized":
        False,

    "model_retuning_authorized":
        False,

    "K_change_authorized":
        False,
}

with open(
    DESIGN_LOCK_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        design,
        f,
        indent=2
    )

DESIGN_HASH = sha256_file(
    DESIGN_LOCK_PATH
)

print("\n" + "=" * 120)
print("STAGE4A-R1 DESIGN LOCKED BEFORE CLINICAL VALUE AUDIT")
print("=" * 120)

print(DESIGN_LOCK_PATH)
print("SHA256 =", DESIGN_HASH)


# ==================================================================================================
# 8. CANONICAL SUBJECT-ID FUNCTION
# ==================================================================================================

def canonical_subject_id(x):

    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    # Excel numeric conversion
    s = re.sub(
        r"\.0$",
        "",
        s
    )

    # Remove sub prefix if present
    s = re.sub(
        r"^sub[-_ ]?",
        "",
        s,
        flags=re.IGNORECASE
    )

    # TDBRAIN identifiers are expected to resolve to digits.
    if not re.fullmatch(r"\d+", s):

        raise ValueError(
            f"Unexpected subject identifier format: {x}"
        )

    return "sub-" + s


# ==================================================================================================
# 9. LOAD FROZEN BASELINE + PARTITION
# ==================================================================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

partition = pd.read_csv(
    PARTITION_FILE
)


# ==================================================================================================
# 10. AUDIT BASELINE IDENTIFIER AGREEMENT
# ==================================================================================================

baseline["key_subject"] = (
    baseline["subject"]
    .apply(canonical_subject_id)
)

baseline["key_subject_norm"] = (
    baseline["subject_norm"]
    .apply(canonical_subject_id)
)

baseline["key_TDBRAIN_ID"] = (
    baseline["TDBRAIN_ID"]
    .apply(canonical_subject_id)
)

baseline_identity_ok = (

    (baseline["key_subject"] == baseline["key_subject_norm"])
    &
    (baseline["key_subject_norm"] == baseline["key_TDBRAIN_ID"])
)

print("\n" + "=" * 120)
print("BASELINE IDENTIFIER CONSISTENCY")
print("=" * 120)

print(
    "Rows:",
    len(baseline)
)

print(
    "Rows with complete identifier agreement:",
    int(baseline_identity_ok.sum())
)

print(
    "Rows with identifier disagreement:",
    int((~baseline_identity_ok).sum())
)

if not baseline_identity_ok.all():

    print(
        baseline.loc[
            ~baseline_identity_ok,
            [
                "subject",
                "subject_norm",
                "TDBRAIN_ID",
                "key_subject",
                "key_subject_norm",
                "key_TDBRAIN_ID",
            ]
        ]
    )

    raise RuntimeError(
        "Baseline subject identifiers disagree. STOP."
    )

print("PASS — baseline subject identifiers agree exactly.")


# ==================================================================================================
# 11. AUDIT PARTITION IDENTIFIER AGREEMENT
# ==================================================================================================

partition["key_subject"] = (
    partition["subject"]
    .apply(canonical_subject_id)
)

partition["key_subject_norm"] = (
    partition["subject_norm"]
    .apply(canonical_subject_id)
)

partition_identity_ok = (
    partition["key_subject"]
    ==
    partition["key_subject_norm"]
)

print("\n" + "=" * 120)
print("AUTHORITATIVE PARTITION IDENTIFIER CONSISTENCY")
print("=" * 120)

print(
    "Rows:",
    len(partition)
)

print(
    "Rows with identifier agreement:",
    int(partition_identity_ok.sum())
)

print(
    "Rows with identifier disagreement:",
    int((~partition_identity_ok).sum())
)

if not partition_identity_ok.all():

    print(
        partition.loc[
            ~partition_identity_ok,
            [
                "subject",
                "subject_norm",
                "key_subject",
                "key_subject_norm",
            ]
        ]
    )

    raise RuntimeError(
        "Partition subject identifiers disagree. STOP."
    )

print("PASS — partition subject identifiers agree exactly.")


# ==================================================================================================
# 12. VERIFY AUTHORITATIVE PARTITION
# ==================================================================================================

if len(partition) != 85:

    raise RuntimeError(
        f"Partition N={len(partition)}, expected 85."
    )

if partition["key_subject_norm"].duplicated().any():

    raise RuntimeError(
        "Duplicate subjects in authoritative partition."
    )

partition["EEG_PHENO_cluster"] = pd.to_numeric(
    partition["EEG_PHENO_cluster"],
    errors="raise"
).astype(int)

partition["K"] = pd.to_numeric(
    partition["K"],
    errors="raise"
).astype(int)

partition["UMAP_seed"] = pd.to_numeric(
    partition["UMAP_seed"],
    errors="raise"
).astype(int)

if set(partition["EEG_PHENO_cluster"].unique()) != {0, 1, 2, 3, 4, 5}:

    raise RuntimeError(
        "Authoritative cluster IDs are not exactly 0-5."
    )

if not (partition["K"] == 6).all():

    raise RuntimeError(
        "Partition contains K other than frozen K=6."
    )

if not (partition["UMAP_seed"] == 42).all():

    raise RuntimeError(
        "Partition contains a UMAP seed other than frozen seed42."
    )

cluster_sizes = (
    partition["EEG_PHENO_cluster"]
    .value_counts()
    .sort_index()
)

expected_cluster_sizes = pd.Series(
    {
        0: 27,
        1: 15,
        2: 18,
        3: 10,
        4: 9,
        5: 6,
    }
)

if not cluster_sizes.equals(expected_cluster_sizes):

    raise RuntimeError(
        "Authoritative cluster-size mismatch.\n"
        f"Observed:\n{cluster_sizes}\n"
        f"Expected:\n{expected_cluster_sizes}"
    )

print("\n" + "=" * 120)
print("AUTHORITATIVE PARTITION VERIFIED")
print("=" * 120)

print("N = 85")
print("K = 6")
print("UMAP seed = 42")
print(cluster_sizes)


# ==================================================================================================
# 13. LOAD ONLY AUTHORIZED METADATA VARIABLES
# ==================================================================================================

metadata = pd.read_excel(
    META_XLSX,
    sheet_name=META_SHEET,
    usecols=AUTHORIZED_METADATA_COLUMNS
)

metadata["metadata_subject_key"] = (
    metadata["TDBRAIN_ID"]
    .apply(canonical_subject_id)
)

metadata["sessID_numeric"] = pd.to_numeric(
    metadata["sessID"],
    errors="coerce"
)

metadata_baseline = metadata[
    metadata["sessID_numeric"] == 1
].copy()


# ==================================================================================================
# 14. VERIFY EXACTLY ONE sessID=1 ROW FOR EACH AUTHORITATIVE SUBJECT
# ==================================================================================================

authoritative_subjects = set(
    partition["key_subject_norm"]
)

meta85 = metadata_baseline[
    metadata_baseline["metadata_subject_key"].isin(
        authoritative_subjects
    )
].copy()

missing_metadata_subjects = sorted(
    authoritative_subjects
    -
    set(meta85["metadata_subject_key"])
)

metadata_counts = (
    meta85["metadata_subject_key"]
    .value_counts()
)

duplicate_metadata_subjects = metadata_counts[
    metadata_counts != 1
]

print("\n" + "=" * 120)
print("BASELINE METADATA SUBJECT COVERAGE")
print("=" * 120)

print("Authoritative subjects:", len(authoritative_subjects))
print("Matched sessID=1 rows:", len(meta85))
print("Missing subjects:", len(missing_metadata_subjects))
print("Non-unique baseline subjects:", len(duplicate_metadata_subjects))

if missing_metadata_subjects:

    print("\nMissing:")
    print("\n".join(missing_metadata_subjects))

    raise RuntimeError(
        "Some authoritative subjects have no sessID=1 metadata row."
    )

if len(duplicate_metadata_subjects) > 0:

    print(duplicate_metadata_subjects)

    raise RuntimeError(
        "Expected one metadata sessID=1 row per authoritative subject."
    )

if len(meta85) != 85:

    raise RuntimeError(
        f"Matched baseline metadata N={len(meta85)}, expected 85."
    )

print("PASS — exactly one baseline metadata row exists for every authoritative subject.")


# ==================================================================================================
# 15. EXTRACT FROZEN STANDARDIZED DIAGNOSIS
# ==================================================================================================

baseline_diag = baseline[
    [
        "key_subject_norm",
        "diagnosis_group_standardized",
    ]
].copy()

baseline_diag = baseline_diag.rename(
    columns={
        "key_subject_norm":
            "subject_key"
    }
)

if baseline_diag["subject_key"].duplicated().any():

    raise RuntimeError(
        "Duplicate subject IDs in frozen Stage0B-R2 diagnosis source."
    )


# ==================================================================================================
# 16. PREPARE METADATA FOR MERGE
# ==================================================================================================

meta85 = meta85.rename(
    columns={
        "metadata_subject_key":
            "subject_key"
    }
)

meta85 = meta85.drop(
    columns=[
        "TDBRAIN_ID",
        "sessID",
        "sessID_numeric",
    ]
)


# ==================================================================================================
# 17. BUILD STAGE4A AUDIT TABLE
# ==================================================================================================

partition_for_merge = partition[
    [
        "key_subject_norm",
        "EEG_PHENO_cluster",
    ]
].copy()

partition_for_merge = partition_for_merge.rename(
    columns={
        "key_subject_norm":
            "subject_key"
    }
)

analysis = (
    partition_for_merge
    .merge(
        baseline_diag,
        on="subject_key",
        how="left",
        validate="one_to_one"
    )
    .merge(
        meta85,
        on="subject_key",
        how="left",
        validate="one_to_one"
    )
)

if len(analysis) != 85:

    raise RuntimeError(
        f"Final audit table N={len(analysis)}, expected 85."
    )

if analysis["diagnosis_group_standardized"].isna().any():

    raise RuntimeError(
        "Missing standardized diagnosis in authoritative cohort."
    )


# ==================================================================================================
# 18. VERIFY FROZEN DIAGNOSIS COUNTS
# ==================================================================================================

diagnosis_counts = (
    analysis["diagnosis_group_standardized"]
    .value_counts()
)

print("\n" + "=" * 120)
print("FROZEN DIAGNOSIS BOOKKEEPING")
print("=" * 120)

print(diagnosis_counts)

expected_diagnosis_counts = {
    "MDD": 31,
    "ADHD_ADD": 28,
    "OCD": 26,
}

actual_diagnosis_counts = {
    str(k): int(v)
    for k, v in diagnosis_counts.items()
}

if actual_diagnosis_counts != expected_diagnosis_counts:

    raise RuntimeError(
        "Diagnosis-count mismatch.\n"
        f"Expected: {expected_diagnosis_counts}\n"
        f"Observed: {actual_diagnosis_counts}"
    )

print("PASS — frozen diagnosis counts exactly reproduced.")


# ==================================================================================================
# 19. VARIABLE FAMILIES — AVAILABILITY ONLY
# ==================================================================================================

VARIABLE_GROUPS = {

    "baseline_symptom": [
        "ADHD_pre_Att_leading",
        "ADHD_pre_Hyp_leading",
        "YBOCS_pre",
        "BDI_pre",
    ],

    "oddball_behavior": [
        "avg_rt_oddb_CP",
        "avg_rt_oddb_FP",
        "n_oddb_CP",
        "n_oddb_FP",
        "n_oddb_CN",
        "n_oddb_FN",
    ],

    "demographic_context": [
        "age",
        "gender",
        "education",
    ],
}


# ==================================================================================================
# 20. OVERALL VARIABLE AVAILABILITY
# ==================================================================================================

availability_rows = []

for family, variables in VARIABLE_GROUPS.items():

    for variable in variables:

        s = analysis[variable]

        availability_rows.append({

            "family":
                family,

            "variable":
                variable,

            "total_N":
                int(len(analysis)),

            "observed_N":
                int(s.notna().sum()),

            "missing_N":
                int(s.isna().sum()),

            "observed_percent":
                float(100 * s.notna().mean()),

            "unique_nonmissing":
                int(s.nunique(dropna=True)),

            "dtype":
                str(s.dtype),
        })

availability = pd.DataFrame(
    availability_rows
)


# ==================================================================================================
# 21. DIAGNOSIS-SPECIFIC AVAILABILITY
# ==================================================================================================

diagnosis_availability_rows = []

for diagnosis, group in analysis.groupby(
    "diagnosis_group_standardized",
    sort=True
):

    for family, variables in VARIABLE_GROUPS.items():

        for variable in variables:

            s = group[variable]

            diagnosis_availability_rows.append({

                "diagnosis":
                    diagnosis,

                "diagnosis_N":
                    int(len(group)),

                "family":
                    family,

                "variable":
                    variable,

                "observed_N":
                    int(s.notna().sum()),

                "missing_N":
                    int(s.isna().sum()),

                "observed_percent":
                    float(100 * s.notna().mean()),
            })

diagnosis_availability = pd.DataFrame(
    diagnosis_availability_rows
)


# ==================================================================================================
# 22. CLUSTER-SPECIFIC AVAILABILITY — NO VALUES / NO TESTS
# ==================================================================================================

cluster_availability_rows = []

for cluster_id, group in analysis.groupby(
    "EEG_PHENO_cluster",
    sort=True
):

    for family, variables in VARIABLE_GROUPS.items():

        for variable in variables:

            s = group[variable]

            cluster_availability_rows.append({

                "cluster":
                    int(cluster_id),

                "cluster_N":
                    int(len(group)),

                "family":
                    family,

                "variable":
                    variable,

                "observed_N":
                    int(s.notna().sum()),

                "missing_N":
                    int(s.isna().sum()),

                "observed_percent":
                    float(100 * s.notna().mean()),
            })

cluster_availability = pd.DataFrame(
    cluster_availability_rows
)


# ==================================================================================================
# 23. CLUSTER × DIAGNOSIS COUNTS — DESCRIPTIVE FEASIBILITY ONLY
# ==================================================================================================

cluster_diagnosis = pd.crosstab(
    analysis["EEG_PHENO_cluster"],
    analysis["diagnosis_group_standardized"],
    margins=True
)


# ==================================================================================================
# 24. SAVE STAGE4A-R1 AUDIT OUTPUTS
# ==================================================================================================

ANALYSIS_TABLE = (
    CHAR_DIR
    / "STAGE4A_R1_BASELINE_HELDOUT_AUDIT_TABLE.csv"
)

AVAILABILITY_FILE = (
    CHAR_DIR
    / "STAGE4A_R1_VARIABLE_AVAILABILITY.csv"
)

DIAGNOSIS_AVAILABILITY_FILE = (
    CHAR_DIR
    / "STAGE4A_R1_DIAGNOSIS_SPECIFIC_AVAILABILITY.csv"
)

CLUSTER_AVAILABILITY_FILE = (
    CHAR_DIR
    / "STAGE4A_R1_CLUSTER_VARIABLE_AVAILABILITY.csv"
)

CLUSTER_DIAGNOSIS_FILE = (
    CHAR_DIR
    / "STAGE4A_R1_CLUSTER_DIAGNOSIS_COUNTS.csv"
)


analysis.to_csv(
    ANALYSIS_TABLE,
    index=False
)

availability.to_csv(
    AVAILABILITY_FILE,
    index=False
)

diagnosis_availability.to_csv(
    DIAGNOSIS_AVAILABILITY_FILE,
    index=False
)

cluster_availability.to_csv(
    CLUSTER_AVAILABILITY_FILE,
    index=False
)

cluster_diagnosis.to_csv(
    CLUSTER_DIAGNOSIS_FILE
)


# ==================================================================================================
# 25. SUMMARY
# ==================================================================================================

SUMMARY_FILE = (
    CHAR_DIR
    / "STAGE4A_R1_AVAILABILITY_AUDIT_SUMMARY.json"
)

summary = {

    "stage":
        "STAGE4A_R1",

    "status":
        "PASS",

    "N":
        85,

    "K":
        6,

    "primary_linkage_key":
        "subject_norm",

    "baseline_identifier_consistency":
        True,

    "partition_identifier_consistency":
        True,

    "all_authoritative_subjects_have_sessID1_metadata":
        True,

    "cluster_sizes": {
        str(k): int(v)
        for k, v in cluster_sizes.items()
    },

    "diagnosis_counts":
        actual_diagnosis_counts,

    "clinical_hypothesis_tests_run":
        False,

    "p_values_computed":
        False,

    "cluster_names_assigned":
        False,

    "K_retuned":
        False,

    "post_treatment_variables_used":
        False,

    "NEO_FFI_role":
        "MODEL INPUT — not independent validation",

    "design_lock_sha256":
        DESIGN_HASH,

    "outputs": {

        "analysis_table_sha256":
            sha256_file(ANALYSIS_TABLE),

        "availability_sha256":
            sha256_file(AVAILABILITY_FILE),

        "diagnosis_availability_sha256":
            sha256_file(DIAGNOSIS_AVAILABILITY_FILE),

        "cluster_availability_sha256":
            sha256_file(CLUSTER_AVAILABILITY_FILE),

        "cluster_diagnosis_sha256":
            sha256_file(CLUSTER_DIAGNOSIS_FILE),
    },
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
# 26. OUTPUT LOCK
# ==================================================================================================

OUTPUT_LOCK_FILE = (
    LOCK_DIR
    / "STAGE4A_R1_AVAILABILITY_AUDIT_LOCK.json"
)

output_lock = {

    "stage":
        "STAGE4A_R1",

    "status":
        "PASS",

    "design_lock": {
        "path":
            str(DESIGN_LOCK_PATH),

        "sha256":
            DESIGN_HASH,
    },

    "analysis_table": {
        "path":
            str(ANALYSIS_TABLE),

        "sha256":
            sha256_file(ANALYSIS_TABLE),
    },

    "variable_availability": {
        "path":
            str(AVAILABILITY_FILE),

        "sha256":
            sha256_file(AVAILABILITY_FILE),
    },

    "diagnosis_specific_availability": {
        "path":
            str(DIAGNOSIS_AVAILABILITY_FILE),

        "sha256":
            sha256_file(DIAGNOSIS_AVAILABILITY_FILE),
    },

    "cluster_variable_availability": {
        "path":
            str(CLUSTER_AVAILABILITY_FILE),

        "sha256":
            sha256_file(CLUSTER_AVAILABILITY_FILE),
    },

    "cluster_diagnosis_counts": {
        "path":
            str(CLUSTER_DIAGNOSIS_FILE),

        "sha256":
            sha256_file(CLUSTER_DIAGNOSIS_FILE),
    },

    "summary": {
        "path":
            str(SUMMARY_FILE),

        "sha256":
            SUMMARY_HASH,
    },

    "clinical_hypothesis_tests_run":
        False,

    "p_values_computed":
        False,

    "post_treatment_variables_used":
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
# 27. PRINT RESULTS
# ==================================================================================================

print("\n" + "=" * 120)
print("OVERALL BASELINE / HELD-OUT VARIABLE AVAILABILITY")
print("=" * 120)

print(
    availability.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("DIAGNOSIS-SPECIFIC AVAILABILITY")
print("=" * 120)

print(
    diagnosis_availability.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("CLUSTER × DIAGNOSIS COUNTS — DESCRIPTIVE ONLY")
print("=" * 120)

print(
    cluster_diagnosis.to_string()
)


print("\n" + "=" * 120)
print("CLUSTER-SPECIFIC VARIABLE COVERAGE — MISSINGNESS ONLY")
print("=" * 120)

print(
    cluster_availability.to_string(
        index=False
    )
)


# ==================================================================================================
# 28. HASH REPORT
# ==================================================================================================

print("\n" + "=" * 120)
print("STAGE4A-R1 HASHES")
print("=" * 120)

print("Design lock:")
print(DESIGN_HASH)

print("\nAnalysis table:")
print(sha256_file(ANALYSIS_TABLE))

print("\nOverall availability:")
print(sha256_file(AVAILABILITY_FILE))

print("\nDiagnosis-specific availability:")
print(sha256_file(DIAGNOSIS_AVAILABILITY_FILE))

print("\nCluster-specific availability:")
print(sha256_file(CLUSTER_AVAILABILITY_FILE))

print("\nCluster × diagnosis counts:")
print(sha256_file(CLUSTER_DIAGNOSIS_FILE))

print("\nSummary:")
print(SUMMARY_HASH)

print("\nOutput lock:")
print(OUTPUT_LOCK_HASH)


# ==================================================================================================
# 29. FINAL STATUS
# ==================================================================================================

print("\n" + "=" * 120)
print("FINAL STATUS")
print("=" * 120)

print("""
STAGE 4A-R1 PASS IF NO ERROR APPEARED ABOVE.

THIS STAGE DID NOT:
- compute any clinical p-value,
- test any cluster association statistically,
- use clinical variables for model selection,
- change K,
- modify the authoritative partition,
- assign clinical cluster names,
- use post-treatment variables.

K REMAINS FROZEN AT 6.
CLUSTERS REMAIN NUMERIC 0-5.
NEO-FFI REMAINS A MODEL INPUT.

NEXT PIPELINE STAGE:
Stage 4B verifies endpoint coverage and freezes the statistical analysis plan.
Inferential testing follows the design checks.
""")