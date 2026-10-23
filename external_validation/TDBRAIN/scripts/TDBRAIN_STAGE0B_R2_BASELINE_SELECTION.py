# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 0B-R2
# BASELINE NEO-FFI EXTRACTION + PRIMARY FUSION COHORT LOCK
#
# RESOLUTION OF PREVIOUS ISSUE
# ----------------------------
# TDBRAIN contains repeated sessions for some subjects.
#
# Verified session field:
#       sessID
#
# Baseline rule:
#       sessID == 1
#
# IMPORTANT
# ---------
# - NO averaging across sessions
# - NO follow-up NEO values
# - NO imputation
# - NO diagnosis used as model input
# - NO ADHD/YBOCS/BDI used as model input
# - NO Experiment-2 cluster labels used
# - NO graph construction
# - NO clustering
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. ROOTS
# ============================================================

DRIVE = Path(
    "/content/drive/MyDrive"
)

EXP1 = (
    DRIVE
    / "TDBRAIN_ADHD_EXPERIMENT1"
)

EXP2 = (
    DRIVE
    / "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT2"
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


for folder in [
    AUDIT,
    LOCK,
    INPUTS
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 0B-R2 — BASELINE NEO-FFI + PRIMARY FUSION COHORT LOCK")
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


def normalize_id(x):

    if pd.isna(x):

        return None

    s = str(
        x
    ).strip()

    if s.lower() in {
        "",
        "nan",
        "none"
    }:

        return None

    if s.startswith(
        "sub-"
    ):

        s = s[
            4:
        ]

    if s.endswith(
        ".0"
    ):

        s = s[
            :-2
        ]

    return s.strip()


# ============================================================
# 3. AUTHORITATIVE FILES
# ============================================================

META = (
    EXP1
    / "00_LOCK"
    / "TDBRAIN_participants_V3.1.xlsx"
)

SHEET = "TDBRAIN-V3.1"


EXP2_FINAL_SUBJECTS = (
    EXP2
    / "16_CLINICAL_CHARACTERIZATION"
    / "STAGE8C_FINAL_SUBJECT_LABELS_AND_BASELINE_PROFILES.csv"
)


STAGE0A_AUDIT = (
    AUDIT
    / "STAGE0A_PHENOTYPE_AVAILABILITY_AUDIT.csv"
)

STAGE0A_CANDIDATES = (
    AUDIT
    / "STAGE0A_PHENOTYPE_MANUAL_REVIEW_CANDIDATES.csv"
)

STAGE0A_LOCK = (
    LOCK
    / "STAGE0A_PHENOTYPE_AVAILABILITY_AUDIT_LOCK.json"
)


R1_REPEATED = (
    AUDIT
    / "STAGE0B_R1_REPEATED_SUBJECT_ROWS.csv"
)

R1_CANDIDATES = (
    AUDIT
    / "STAGE0B_R1_SESSION_COLUMN_CANDIDATES.csv"
)

R1_DIFFERENCES = (
    AUDIT
    / "STAGE0B_R1_REPEATED_ROW_CONTEXT_DIFFERENCES.csv"
)

R1_SUMMARY = (
    AUDIT
    / "STAGE0B_R1_BASELINE_SESSION_RESOLUTION_SUMMARY.json"
)


EQUAL8_ROOT = (
    EXP2
    / "15_EQUAL_WINDOW_SENSITIVITY"
)

EQUAL8_EEG128 = (
    EQUAL8_ROOT
    / "EQUAL8_SUBJECT_FEATURE128.npy"
)

EQUAL8_LABEL_TABLE = (
    EQUAL8_ROOT
    / "EQUAL8_K3_SUBJECT_LABELS.csv"
)


for path in [

    META,
    EXP2_FINAL_SUBJECTS,

    STAGE0A_AUDIT,
    STAGE0A_CANDIDATES,
    STAGE0A_LOCK,

    R1_REPEATED,
    R1_CANDIDATES,
    R1_DIFFERENCES,
    R1_SUMMARY,

    EQUAL8_EEG128,
    EQUAL8_LABEL_TABLE

]:

    if not path.exists():

        raise FileNotFoundError(
            str(
                path
            )
        )


# ============================================================
# 4. VERIFY COMPLETED AUDITS / FROZEN EEG INPUT
# ============================================================

EXPECTED_HASHES = {

    STAGE0A_AUDIT:
        "83dc848a63f6f9dcc005e165cd291eb610efb45d609cdd8ae72eb70a9ae1b4fb",

    STAGE0A_CANDIDATES:
        "3f07d613c1d3b336e942ec5252a2cfbb4a7eeddf954e2d6c7f640cfa46de87a6",

    STAGE0A_LOCK:
        "b960cd27b5a617478534a2c583a39a1495dca4fc450f3db0c9f83a9060e6f4b2",

    R1_REPEATED:
        "a69339d4e30ec33805986cbec792d449d50ac0b3fb34a277c55af3c089e86d4a",

    R1_CANDIDATES:
        "422959d733dd84d3bd8907652a5e4a5d9e3b587d1a83f4312bead8a7242e376d",

    R1_DIFFERENCES:
        "74db0352c99cb8e2f43bad5af218b8311316aa5665fa1748190b7cff1bf37743",

    R1_SUMMARY:
        "d1cbd15d827e3a015439085e23c9764a74e57679196c6a78b700c058c0747c37",

    EQUAL8_EEG128:
        "cb8909876944dc70465668880a6166c48ad470b3a7771c27cc3c00c10c23ba3d",

    EQUAL8_LABEL_TABLE:
        "ff8a508d2a428eb3560aae87781eec2540fb3ad21e9b9fc45894f259e5a732fb"
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
            "Frozen input changed. STOP."
        )


print(
    "\nALL AUTHORITATIVE INPUT HASHES PASS."
)


# ============================================================
# 5. VERIFY R1 SESSION RESOLUTION
# ============================================================

with open(
    R1_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    r1 = json.load(
        f
    )


valid_session_columns = r1[
    "valid_session_columns_with_unique_baseline_for_all_subjects"
]


assert valid_session_columns == [
    "sessID"
]


assert r1[
    "source_subjects"
] == 86


assert r1[
    "metadata_rows"
] == 91


assert r1[
    "repeated_subjects"
] == 5


print(
    "\nVerified baseline session field: sessID"
)

print(
    "Frozen baseline rule: sessID == 1"
)


# ============================================================
# 6. DEFINE PHENOTYPE FEATURES
# ============================================================

NEO_COLUMNS = [

    f"neoFFI_q{i}"

    for i in range(
        1,
        61
    )
]


assert len(
    NEO_COLUMNS
) == 60


print(
    "\nPhenotype feature family:"
)

print(
    "NEO-FFI item-level responses"
)

print(
    "Features:",
    len(
        NEO_COLUMNS
    )
)


# ============================================================
# 7. LOAD SOURCE N=86 COHORT
#
# IMPORTANT:
# Do not load Experiment-2 cluster labels.
# ============================================================

cohort = pd.read_csv(

    EXP2_FINAL_SUBJECTS,

    usecols=[
        "subject",
        "diagnosis_group_standardized"
    ]
)


assert len(
    cohort
) == 86


assert cohort[
    "subject"
].nunique() == 86


cohort[
    "subject_norm"
] = cohort[
    "subject"
].map(
    normalize_id
)


assert cohort[
    "subject_norm"
].notna().all()


assert cohort[
    "subject_norm"
].nunique() == 86


source_subjects = set(
    cohort[
        "subject_norm"
    ]
)


print(
    "\nFrozen source cohort:"
)

print(
    "N =",
    len(
        cohort
    )
)


print(

    cohort[
        "diagnosis_group_standardized"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 8. LOAD ONLY ID + SESSION + NEO60
# ============================================================

header = pd.read_excel(

    META,

    sheet_name=SHEET,

    nrows=0
)


required_columns = {

    "TDBRAIN_ID",
    "sessID",
    *NEO_COLUMNS
}


missing_columns = (

    required_columns

    -

    set(
        header.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Required metadata columns missing: "
        +
        str(
            sorted(
                missing_columns
            )
        )
    )


meta = pd.read_excel(

    META,

    sheet_name=SHEET,

    usecols=[
        "TDBRAIN_ID",
        "sessID",
        *NEO_COLUMNS
    ]
)


meta[
    "subject_norm"
] = meta[
    "TDBRAIN_ID"
].map(
    normalize_id
)


meta = meta[

    meta[
        "subject_norm"
    ].isin(
        source_subjects
    )

].copy()


meta[
    "_source_excel_row"
] = meta.index + 2


print(
    "\nMetadata rows for N=86:",
    len(
        meta
    )
)


assert len(
    meta
) == 91


# ============================================================
# 9. PARSE SESSION FIELD
# ============================================================

meta[
    "sessID_numeric"
] = pd.to_numeric(

    meta[
        "sessID"
    ],

    errors="coerce"
)


if meta[
    "sessID_numeric"
].isna().any():

    bad = meta.loc[

        meta[
            "sessID_numeric"
        ].isna(),

        [
            "TDBRAIN_ID",
            "sessID"
        ]
    ]

    print(
        bad.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Unparseable sessID detected. STOP."
    )


meta[
    "sessID_numeric"
] = meta[
    "sessID_numeric"
].astype(
    int
)


# ============================================================
# 10. SELECT BASELINE SESSION ONLY
#
# FROZEN RULE:
#       sessID == 1
# ============================================================

baseline = meta[

    meta[
        "sessID_numeric"
    ]
    ==
    1

].copy()


counts = (

    baseline[
        "subject_norm"
    ]
    .value_counts()
)


if len(
    counts
) != 86:

    raise RuntimeError(
        "Not all 86 subjects have a baseline sessID=1 row."
    )


if not (
    counts
    ==
    1
).all():

    raise RuntimeError(
        "One or more subjects have multiple sessID=1 rows."
    )


assert len(
    baseline
) == 86


assert baseline[
    "subject_norm"
].nunique() == 86


print(
    "\n" + "=" * 120
)

print(
    "BASELINE SESSION SELECTION"
)

print(
    "=" * 120
)


print(
    "Baseline rule: sessID == 1"
)

print(
    "Baseline rows:",
    len(
        baseline
    ),
    "/ 86"
)

print(
    "Unique baseline subjects:",
    baseline[
        "subject_norm"
    ].nunique(),
    "/ 86"
)

print(
    "Follow-up rows used: 0"
)


# ============================================================
# 11. SAVE BASELINE ROW SELECTION AUDIT
# ============================================================

BASELINE_SELECTION_FILE = (
    AUDIT
    / "STAGE0B_R2_BASELINE_SESSION1_SELECTION.csv"
)


baseline_selection = (

    cohort[
        [
            "subject",
            "subject_norm",
            "diagnosis_group_standardized"
        ]
    ]

    .merge(

        baseline[
            [
                "subject_norm",
                "TDBRAIN_ID",
                "sessID_numeric",
                "_source_excel_row"
            ]
        ],

        on="subject_norm",

        how="left",

        validate="one_to_one"
    )
)


baseline_selection.to_csv(
    BASELINE_SELECTION_FILE,
    index=False
)


# ============================================================
# 12. BASELINE NEO-FFI NUMERIC VALIDATION
# ============================================================

for variable in NEO_COLUMNS:

    baseline[
        variable
    ] = pd.to_numeric(

        baseline[
            variable
        ],

        errors="coerce"
    )


item_rows = []


for variable in NEO_COLUMNS:

    x = baseline[
        variable
    ]

    observed = x.dropna()


    item_rows.append({

        "variable":
            variable,

        "observed_N":
            int(
                observed.shape[
                    0
                ]
            ),

        "missing_N":
            int(
                x.isna().sum()
            ),

        "minimum":
            (
                float(
                    observed.min()
                )
                if len(
                    observed
                ) > 0
                else np.nan
            ),

        "maximum":
            (
                float(
                    observed.max()
                )
                if len(
                    observed
                ) > 0
                else np.nan
            ),

        "mean":
            (
                float(
                    observed.mean()
                )
                if len(
                    observed
                ) > 0
                else np.nan
            ),

        "SD_population":
            (
                float(
                    observed.std(
                        ddof=0
                    )
                )
                if len(
                    observed
                ) > 0
                else np.nan
            ),

        "unique_values":
            int(
                observed.nunique()
            ),

        "zero_variance":
            bool(
                observed.nunique()
                <=
                1
            )
    })


item_audit = pd.DataFrame(
    item_rows
)


ITEM_AUDIT_FILE = (
    AUDIT
    / "STAGE0B_R2_BASELINE_NEOFFI60_ITEM_AUDIT.csv"
)


item_audit.to_csv(
    ITEM_AUDIT_FILE,
    index=False
)


zero_variance = item_audit.loc[

    item_audit[
        "zero_variance"
    ],

    "variable"

].tolist()


print(
    "\nBaseline NEO item observed-N range:",
    int(
        item_audit[
            "observed_N"
        ].min()
    ),
    "to",
    int(
        item_audit[
            "observed_N"
        ].max()
    )
)


print(
    "Zero-variance items:",
    zero_variance
)


if zero_variance:

    raise RuntimeError(
        "Zero-variance baseline NEO item detected. STOP."
    )


# ============================================================
# 13. SUBJECT COMPLETENESS
#
# PRIMARY RULE:
# Complete on all 60 baseline NEO items.
#
# NO IMPUTATION.
# ============================================================

baseline[
    "NEO_observed_items"
] = baseline[
    NEO_COLUMNS
].notna().sum(
    axis=1
)


baseline[
    "NEO_missing_items"
] = (

    60

    -

    baseline[
        "NEO_observed_items"
    ]
)


baseline[
    "NEO_complete60"
] = (

    baseline[
        "NEO_observed_items"
    ]
    ==
    60
)


completeness = (

    cohort[
        [
            "subject",
            "subject_norm",
            "diagnosis_group_standardized"
        ]
    ]

    .merge(

        baseline[
            [
                "subject_norm",
                "sessID_numeric",
                "NEO_observed_items",
                "NEO_missing_items",
                "NEO_complete60"
            ]
        ],

        on="subject_norm",

        how="left",

        validate="one_to_one"
    )
)


COMPLETENESS_FILE = (
    AUDIT
    / "STAGE0B_R2_BASELINE_NEOFFI60_SUBJECT_COMPLETENESS.csv"
)


completeness.to_csv(
    COMPLETENESS_FILE,
    index=False
)


complete = completeness[

    completeness[
        "NEO_complete60"
    ]

].copy()


N_COMPLETE = len(
    complete
)


excluded = completeness[

    ~completeness[
        "NEO_complete60"
    ]

].copy()


print(
    "\n" + "=" * 120
)

print(
    "PRIMARY COMPLETE-CASE PHENOTYPE COHORT"
)

print(
    "=" * 120
)


print(
    "Complete baseline NEO-FFI:",
    N_COMPLETE,
    "/ 86"
)


print(
    "\nComplete-case diagnosis counts:"
)

print(

    complete[
        "diagnosis_group_standardized"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nExcluded for incomplete baseline NEO-FFI:",
    len(
        excluded
    )
)


if len(
    excluded
) > 0:

    print(

        excluded[
            [
                "subject",
                "diagnosis_group_standardized",
                "NEO_observed_items",
                "NEO_missing_items"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 14. CREATE DETERMINISTIC EXPERIMENT-3 SUBJECT ORDER
#
# Numeric subject sort is NOT assumed.
# We use the exact Stage7G EEG subject order as the
# authoritative reference for EEG128 rows.
# ============================================================

stage7g_subject_table = pd.read_csv(

    EQUAL8_LABEL_TABLE,

    usecols=[
        "subject"
    ],

    dtype={
        "subject": str
    }
)


assert len(
    stage7g_subject_table
) == 86


assert stage7g_subject_table[
    "subject"
].nunique() == 86


stage7g_subject_table[
    "subject_norm"
] = stage7g_subject_table[
    "subject"
].map(
    normalize_id
)


assert stage7g_subject_table[
    "subject_norm"
].nunique() == 86


complete_set = set(
    complete[
        "subject_norm"
    ]
)


primary_order = stage7g_subject_table[

    stage7g_subject_table[
        "subject_norm"
    ].isin(
        complete_set
    )

].copy()


assert len(
    primary_order
) == N_COMPLETE


assert primary_order[
    "subject_norm"
].nunique() == N_COMPLETE


primary_order[
    "experiment3_row"
] = np.arange(
    N_COMPLETE
)


# Add original subject label from N=86 cohort.
subject_name_map = dict(

    zip(
        cohort[
            "subject_norm"
        ],
        cohort[
            "subject"
        ]
    )
)


primary_order[
    "subject"
] = primary_order[
    "subject_norm"
].map(
    subject_name_map
)


# Save only identity/order.
# Diagnosis is intentionally NOT included in model input file.

SUBJECT_ORDER_FILE = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)


primary_order[
    [
        "experiment3_row",
        "subject",
        "subject_norm"
    ]
].to_csv(
    SUBJECT_ORDER_FILE,
    index=False
)


# ============================================================
# 15. CREATE BASELINE PHENOTYPE MATRIX
#
# Exact same subject order as Experiment-3 row order.
# Raw item responses only.
# Standardization happens later.
# ============================================================

baseline_indexed = (
    baseline
    .set_index(
        "subject_norm"
    )
)


X_pheno = np.vstack(

    [

        baseline_indexed.loc[
            sid,
            NEO_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )

        for sid in primary_order[
            "subject_norm"
        ]

    ]
)


assert X_pheno.shape == (
    N_COMPLETE,
    60
)


assert np.isfinite(
    X_pheno
).all()


assert np.all(

    np.std(
        X_pheno,
        axis=0,
        ddof=0
    )

    >
    0
)


PHENO_NPY = (
    INPUTS
    / "PRIMARY_BASELINE_NEOFFI60_RAW.npy"
)


np.save(
    PHENO_NPY,
    X_pheno.astype(
        np.float32
    )
)


PHENO_CSV = (
    INPUTS
    / "PRIMARY_BASELINE_NEOFFI60_RAW.csv"
)


pheno_csv = primary_order[
    [
        "experiment3_row",
        "subject",
        "subject_norm"
    ]
].copy()


for j, variable in enumerate(
    NEO_COLUMNS
):

    pheno_csv[
        variable
    ] = X_pheno[
        :,
        j
    ]


pheno_csv.to_csv(
    PHENO_CSV,
    index=False
)


# ============================================================
# 16. ALIGN EXACT STAGE7G EQUAL-WINDOW EEG128
#
# Stage7G constructed SUBJECT128 in the same ordered
# 'subjects' sequence used to write EQUAL8_K3_SUBJECT_LABELS.
#
# We use only the subject column from that table.
# Cluster labels are NOT loaded.
# ============================================================

X_eeg86 = np.load(
    EQUAL8_EEG128
)


assert X_eeg86.shape == (
    86,
    128
)


assert np.isfinite(
    X_eeg86
).all()


stage7g_row_map = {

    sid:
        i

    for i, sid in enumerate(

        stage7g_subject_table[
            "subject_norm"
        ].tolist()
    )
}


eeg_row_indices = [

    stage7g_row_map[
        sid
    ]

    for sid in primary_order[
        "subject_norm"
    ]
]


X_eeg = X_eeg86[
    eeg_row_indices
].astype(
    np.float32
)


assert X_eeg.shape == (
    N_COMPLETE,
    128
)


assert np.isfinite(
    X_eeg
).all()


EEG_NPY = (
    INPUTS
    / "PRIMARY_EQUAL8_EEG128_RAW.npy"
)


np.save(
    EEG_NPY,
    X_eeg
)


# ============================================================
# 17. SAVE COMPLETE-CASE COHORT AUDIT
#
# Diagnosis retained here ONLY as held-out audit information.
# This file is NOT a model-input matrix.
# ============================================================

PRIMARY_COHORT_AUDIT_FILE = (
    AUDIT
    / "STAGE0B_R2_PRIMARY_FUSION_COHORT_AUDIT.csv"
)


primary_cohort_audit = (

    primary_order[
        [
            "experiment3_row",
            "subject",
            "subject_norm"
        ]
    ]

    .merge(

        cohort[
            [
                "subject_norm",
                "diagnosis_group_standardized"
            ]
        ],

        on="subject_norm",

        how="left",

        validate="one_to_one"
    )
)


primary_cohort_audit.to_csv(
    PRIMARY_COHORT_AUDIT_FILE,
    index=False
)


# ============================================================
# 18. MODEL / HELD-OUT POLICY
# ============================================================

HELD_OUT = [

    "diagnosis",

    "ADHD_pre_Att_leading",

    "ADHD_pre_Hyp_leading",

    "YBOCS_pre",

    "BDI_pre",

    "oddball behavioral measures"
]


FORBIDDEN_MODEL_SELECTION = [

    "diagnosis",

    "ADHD_pre_Att_leading",

    "ADHD_pre_Hyp_leading",

    "YBOCS_pre",

    "BDI_pre",

    "all post-treatment variables",

    "Experiment2 EEG cluster labels"
]


# ============================================================
# 19. FREEZE EXPERIMENT-3 DESIGN
#
# This is frozen BEFORE fused graph construction.
#
# alpha=0.6 is treated as a PRESPECIFIED TRANSFER
# from the original EPG-SD configuration.
#
# It will NOT be retuned using TDBRAIN clinical outcomes.
# ============================================================

DESIGN_LOCK = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_DESIGN_LOCK.json"
)


design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE0B_R2_PRIMARY_FUSION_DESIGN_LOCK",

    "status":
        "FROZEN_BEFORE_GRAPH_CONSTRUCTION",

    "source_cohort_N":
        86,

    "baseline_session_rule":
        {

            "field":
                "sessID",

            "value":
                1,

            "reason":
                (
                    "Use phenotype measurements from the same "
                    "baseline session as the primary baseline EEG."
                ),

            "followup_rows_used":
                False,

            "repeated_session_values_averaged":
                False
        },

    "phenotype_complete_case_rule":
        {

            "features":
                60,

            "required_nonmissing":
                60,

            "imputation":
                False,

            "primary_fusion_N":
                int(
                    N_COMPLETE
                )
        },

    "primary_EEG_view":
        {

            "source":
                "Experiment2 Stage7G equal-window EEG representation",

            "dimension":
                128,

            "restEO_windows":
                8,

            "restEC_windows":
                8,

            "total_windows":
                16,

            "clean_window_quantity_equalized":
                True
        },

    "primary_phenotype_view":
        {

            "source":
                "baseline session-1 NEO-FFI",

            "representation":
                "60 item-level responses",

            "dimension":
                60,

            "official_BigFive_trait_scores_computed":
                False,

            "reverse_scoring_guessed":
                False,

            "imputation":
                False
        },

    "standardization":
        {

            "EEG":
                "feature-wise z-score across Experiment3 subjects",

            "phenotype":
                "item-wise z-score across Experiment3 subjects"
        },

    "similarity_views":
        {

            "EEG":
                "cosine_similarity",

            "phenotype":
                "cosine_similarity"
        },

    "fusion":
        {

            "equation":
                "A_fused = alpha*A_EEG + (1-alpha)*A_PHENOTYPE",

            "alpha":
                0.6,

            "EEG_weight":
                0.6,

            "phenotype_weight":
                0.4,

            "status":
                "prespecified_transfer_from_EPGSD",

            "clinical_outcomes_used_to_choose_alpha":
                False,

            "TDBRAIN_alpha_retuning_permitted":
                False
        },

    "future_node_features":
        {

            "EEG":
                128,

            "phenotype":
                60,

            "concatenated_dimension":
                188
        },

    "graph_neighborhood_candidates":
        [
            4,
            6,
            8,
            10,
            12
        ],

    "graph_neighborhood_selection_rule":
        (
            "smallest candidate k producing one connected "
            "component with zero isolated nodes"
        ),

    "VGAE":
        {

            "input_dimension":
                188,

            "hidden_dimension":
                64,

            "latent_dimension":
                16,

            "activation":
                "ReLU",

            "dropout":
                0.0,

            "decoder":
                "inner_product_sigmoid",

            "optimizer":
                "Adam",

            "learning_rate":
                0.001,

            "epochs":
                500,

            "downstream_embedding":
                "posterior_mean_mu",

            "seed":
                42
        },

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

            "primary_seed":
                42
        },

    "spectral_clustering":
        {

            "affinity":
                "exact_4NN_on_UMAP_embedding",

            "candidate_K":
                [
                    2,
                    3,
                    4,
                    5,
                    6
                ],

            "selection_inputs":
                "internal_geometry_and_EEG_phenotype_stability_only",

            "diagnosis_used_for_K_selection":
                False,

            "clinical_scores_used_for_K_selection":
                False
        },

    "held_out_characterization":
        HELD_OUT,

    "forbidden_for_model_selection":
        FORBIDDEN_MODEL_SELECTION,

    "Experiment2_partition_modified":
        False,

    "graph_constructed":
        False,

    "clustering_run":
        False,

    "input_hashes":
        {

            "metadata":
                sha256_file(
                    META
                ),

            "Experiment2_final_subject_table":
                sha256_file(
                    EXP2_FINAL_SUBJECTS
                ),

            "Stage0A_lock":
                sha256_file(
                    STAGE0A_LOCK
                ),

            "Stage0B_R1_summary":
                sha256_file(
                    R1_SUMMARY
                ),

            "equal8_EEG128":
                sha256_file(
                    EQUAL8_EEG128
                ),

            "equal8_subject_order_source":
                sha256_file(
                    EQUAL8_LABEL_TABLE
                )
        },

    "output_hashes":
        {

            "baseline_selection":
                sha256_file(
                    BASELINE_SELECTION_FILE
                ),

            "baseline_item_audit":
                sha256_file(
                    ITEM_AUDIT_FILE
                ),

            "baseline_subject_completeness":
                sha256_file(
                    COMPLETENESS_FILE
                ),

            "primary_subject_order":
                sha256_file(
                    SUBJECT_ORDER_FILE
                ),

            "phenotype_raw_npy":
                sha256_file(
                    PHENO_NPY
                ),

            "phenotype_raw_csv":
                sha256_file(
                    PHENO_CSV
                ),

            "EEG_raw_npy":
                sha256_file(
                    EEG_NPY
                ),

            "primary_cohort_audit":
                sha256_file(
                    PRIMARY_COHORT_AUDIT_FILE
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
            "Existing Stage0B-R2 design lock differs. STOP."
        )

    print(
        "\nExisting Stage0B-R2 design lock verified."
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
        "\nStage0B-R2 fusion design lock created."
    )


# ============================================================
# 20. SUMMARY
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE0B_R2_BASELINE_PHENOTYPE_COHORT_SUMMARY.json"
)


summary = {

    "source_N":
        86,

    "baseline_session_field":
        "sessID",

    "baseline_session_value":
        1,

    "baseline_rows":
        86,

    "followup_rows_used":
        0,

    "NEOFFI_items":
        60,

    "primary_fusion_N":
        int(
            N_COMPLETE
        ),

    "excluded_for_incomplete_baseline_NEO":
        int(
            len(
                excluded
            )
        ),

    "complete_case_diagnosis_counts":
        {

            str(k):
                int(v)

            for k, v in

            complete[
                "diagnosis_group_standardized"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

    "EEG_dimension":
        128,

    "phenotype_dimension":
        60,

    "future_node_feature_dimension":
        188,

    "fusion_alpha":
        0.6,

    "fusion_alpha_status":
        "prespecified_transfer_from_EPGSD",

    "diagnosis_used_as_model_input":
        False,

    "clinical_scores_used_as_model_input":
        False,

    "Experiment2_labels_used":
        False,

    "post_treatment_used":
        False,

    "graph_constructed":
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
# 21. FINAL INTEGRITY CHECK
# ============================================================

assert np.load(
    PHENO_NPY
).shape == (
    N_COMPLETE,
    60
)


assert np.load(
    EEG_NPY
).shape == (
    N_COMPLETE,
    128
)


order_check = pd.read_csv(
    SUBJECT_ORDER_FILE
)


assert len(
    order_check
) == N_COMPLETE


assert np.array_equal(

    order_check[
        "experiment3_row"
    ].to_numpy(),

    np.arange(
        N_COMPLETE
    )
)


# ============================================================
# 22. PRINT FINAL RESULT
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "EXPERIMENT 3 STAGE 0B-R2 RESULT"
)

print(
    "=" * 120
)


print(
    "Source cohort:",
    86
)


print(
    "Baseline session rule:"
)

print(
    "sessID = 1"
)


print(
    "\nBaseline rows selected:",
    86
)


print(
    "Follow-up rows used:",
    0
)


print(
    "\nComplete baseline NEO-FFI subjects:",
    N_COMPLETE
)


print(
    "Excluded for incomplete baseline NEO:",
    len(
        excluded
    )
)


print(
    "\nPrimary fusion cohort diagnosis counts:"
)

print(

    complete[
        "diagnosis_group_standardized"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nPRIMARY MODEL INPUTS"
)

print(
    "EEG:",
    X_eeg.shape
)

print(
    "Phenotype:",
    X_pheno.shape
)


print(
    "\nFuture concatenated node feature dimension:",
    188
)


print(
    "\nFusion alpha:",
    0.6
)

print(
    "alpha status: PRESPECIFIED TRANSFER FROM EPG-SD"
)


print(
    "\nDiagnosis used as model input: NO"
)

print(
    "ADHD scores used as model input: NO"
)

print(
    "YBOCS used as model input: NO"
)

print(
    "BDI used as model input: NO"
)

print(
    "Experiment2 clusters used: NO"
)

print(
    "Post-treatment variables used: NO"
)


print(
    "\nHASHES"
)

print(
    "Baseline selection:",
    sha256_file(
        BASELINE_SELECTION_FILE
    )
)

print(
    "Item audit:",
    sha256_file(
        ITEM_AUDIT_FILE
    )
)

print(
    "Completeness:",
    sha256_file(
        COMPLETENESS_FILE
    )
)

print(
    "Subject order:",
    sha256_file(
        SUBJECT_ORDER_FILE
    )
)

print(
    "Phenotype NPY:",
    sha256_file(
        PHENO_NPY
    )
)

print(
    "Phenotype CSV:",
    sha256_file(
        PHENO_CSV
    )
)

print(
    "EEG NPY:",
    sha256_file(
        EEG_NPY
    )
)

print(
    "Cohort audit:",
    sha256_file(
        PRIMARY_COHORT_AUDIT_FILE
    )
)

print(
    "Summary:",
    sha256_file(
        SUMMARY_FILE
    )
)

print(
    "Design lock:",
    sha256_file(
        DESIGN_LOCK
    )
)


print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 0B-R2 PASS — "
    "BASELINE NEO-FFI PHENOTYPE AND PRIMARY FUSION COHORT FROZEN"
)

print(
    "=" * 120
)


print(
    "\nNO NEO RESPONSES WERE AVERAGED ACROSS SESSIONS."
)

print(
    "NO IMPUTATION WAS PERFORMED."
)

print(
    "NO FUSED GRAPH HAS BEEN CREATED."
)

print(
    "NO CLUSTERING HAS BEEN RUN."
)

print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 1A — STANDARDIZE THE TWO VIEWS AND "
    "CONSTRUCT THE PRESPECIFIED alpha=0.6 FUSED SUBJECT GRAPH."
)