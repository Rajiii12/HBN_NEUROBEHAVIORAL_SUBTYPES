# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 0B — PHENOTYPE FEATURE + COHORT + FUSION DESIGN LOCK
#
# PRIMARY PHENOTYPE VIEW
# ----------------------
# 60 item-level NEO-FFI responses:
#   neoFFI_q1 ... neoFFI_q60
#
# IMPORTANT:
#   - raw item responses are used
#   - NO unofficial Big-Five scoring
#   - NO guessed reverse coding
#   - NO imputation
#   - complete-case NEO-FFI subjects only
#
# PRIMARY EEG VIEW
# ----------------
# Stage7G equal-window EEG subject representation.
#
# This avoids the known retained-clean-window-count
# imbalance discovered in Experiment 2.
#
# THIS STAGE DOES NOT:
#   build fused graph
#   run VGAE
#   run UMAP
#   cluster
#   use diagnosis for feature construction
#   use ADHD/YBOCS/BDI
#   use Experiment2 cluster labels
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
print("STAGE 0B — PHENOTYPE FEATURE + COHORT + FUSION DESIGN LOCK")
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

    if pd.isna(
        x
    ):

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
# 3. AUTHORITATIVE INPUTS
# ============================================================

META = (
    EXP1
    / "00_LOCK"
    / "TDBRAIN_participants_V3.1.xlsx"
)

SHEET = (
    "TDBRAIN-V3.1"
)


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


# Experiment-2 equal-window representation.

EQUAL8_ROOT = (
    EXP2
    / "15_EQUAL_WINDOW_SENSITIVITY"
)

EQUAL8_EEG128 = (
    EQUAL8_ROOT
    / "EQUAL8_SUBJECT_FEATURE128.npy"
)

EQUAL8_WINDOW_SELECTION = (
    EQUAL8_ROOT
    / "EQUAL8_WINDOW_SELECTION.csv"
)


for path in [

    META,
    EXP2_FINAL_SUBJECTS,

    STAGE0A_AUDIT,
    STAGE0A_CANDIDATES,
    STAGE0A_LOCK,

    EQUAL8_EEG128,
    EQUAL8_WINDOW_SELECTION

]:

    if not path.exists():

        raise FileNotFoundError(
            str(
                path
            )
        )


# ============================================================
# 4. VERIFY COMPLETED STAGE0A
# ============================================================

EXPECTED_STAGE0A_AUDIT_SHA = (
    "83dc848a63f6f9dcc005e165cd291eb610efb45d609cdd8ae72eb70a9ae1b4fb"
)

EXPECTED_STAGE0A_CANDIDATE_SHA = (
    "3f07d613c1d3b336e942ec5252a2cfbb4a7eeddf954e2d6c7f640cfa46de87a6"
)

EXPECTED_STAGE0A_LOCK_SHA = (
    "b960cd27b5a617478534a2c583a39a1495dca4fc450f3db0c9f83a9060e6f4b2"
)


EXPECTED_EQUAL8_EEG_SHA = (
    "cb8909876944dc70465668880a6166c48ad470b3a7771c27cc3c00c10c23ba3d"
)

EXPECTED_EQUAL8_SELECTION_SHA = (
    "314f0d1185795af8ca7328ec9c0c44cce55b75540f530aa695dc8b53758deca0"
)


checks = [

    (
        STAGE0A_AUDIT,
        EXPECTED_STAGE0A_AUDIT_SHA
    ),

    (
        STAGE0A_CANDIDATES,
        EXPECTED_STAGE0A_CANDIDATE_SHA
    ),

    (
        STAGE0A_LOCK,
        EXPECTED_STAGE0A_LOCK_SHA
    ),

    (
        EQUAL8_EEG128,
        EXPECTED_EQUAL8_EEG_SHA
    ),

    (
        EQUAL8_WINDOW_SELECTION,
        EXPECTED_EQUAL8_SELECTION_SHA
    )
]


print(
    "\n" + "=" * 120
)

print(
    "FROZEN INPUT HASH VERIFICATION"
)

print(
    "=" * 120
)


for path, expected in checks:

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


print(
    "\nALL AUTHORITATIVE INPUT HASHES PASS."
)


# ============================================================
# 5. PROSPECTIVELY DEFINE PRIMARY PHENOTYPE FEATURES
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


audit = pd.read_csv(
    STAGE0A_AUDIT
)


audit_variables = set(
    audit[
        "variable"
    ].astype(str)
)


missing_from_audit = [

    c
    for c in NEO_COLUMNS

    if c not in audit_variables
]


if missing_from_audit:

    raise RuntimeError(
        "NEO-FFI items missing from Stage0A audit: "
        +
        str(
            missing_from_audit
        )
    )


neo_audit = (

    audit[
        audit[
            "variable"
        ].isin(
            NEO_COLUMNS
        )
    ]

    .set_index(
        "variable"
    )

    .loc[
        NEO_COLUMNS
    ]

    .reset_index()
)


assert len(
    neo_audit
) == 60


print(
    "\nPrimary phenotype feature family:"
)

print(
    "NEO-FFI item-level response vector"
)

print(
    "Number of features:",
    60
)


print(
    "\nStage0A NEO-FFI availability:"
)

print(
    "Overall observed range:",
    int(
        neo_audit[
            "overall_observed"
        ].min()
    ),
    "to",
    int(
        neo_audit[
            "overall_observed"
        ].max()
    )
)


print(
    "Minimum diagnosis-group coverage:",
    f"{neo_audit['minimum_group_percent'].min():.2f}%"
)


# ============================================================
# 6. LOAD FROZEN N=86 SUBJECT COHORT
#
# IMPORTANT:
# EEG_cluster is NOT loaded.
# ============================================================

cohort_header = pd.read_csv(
    EXP2_FINAL_SUBJECTS,
    nrows=0
)


required_cohort_columns = {

    "subject",
    "diagnosis_group_standardized"
}


if not required_cohort_columns.issubset(
    set(
        cohort_header.columns
    )
):

    raise RuntimeError(
        "Required Experiment2 cohort fields absent."
    )


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


print(
    "\nFrozen source cohort:",
    len(
        cohort
    )
)


print(
    cohort[
        "diagnosis_group_standardized"
    ].value_counts()
)


# ============================================================
# 7. LOAD ONLY TDBRAIN_ID + 60 NEO-FFI ITEMS
#
# No diagnosis, clinical score, age, sex, post-treatment,
# oddball or lifestyle value is loaded from metadata here.
# ============================================================

metadata_header = pd.read_excel(

    META,

    sheet_name=SHEET,

    nrows=0
)


required_meta = {

    "TDBRAIN_ID",
    *NEO_COLUMNS
}


missing_meta = (

    required_meta

    -

    set(
        metadata_header.columns
    )
)


if missing_meta:

    raise RuntimeError(
        "Required metadata fields absent: "
        +
        str(
            sorted(
                missing_meta
            )
        )
    )


meta = pd.read_excel(

    META,

    sheet_name=SHEET,

    usecols=[
        "TDBRAIN_ID",
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
        set(
            cohort[
                "subject_norm"
            ]
        )
    )

].copy()


print(
    "\nMetadata rows for source N=86:",
    len(
        meta
    )
)


# ============================================================
# 8. CONSOLIDATE REPEATED METADATA ROWS
#
# Rule:
# For each subject and item:
#
#   0 nonmissing values -> missing
#   1 unique value      -> retain
#   >1 unique values    -> STOP
#
# We do NOT average repeated questionnaire responses.
# ============================================================

subject_rows = []

conflict_rows = []


for sid in cohort[
    "subject_norm"
]:

    g = meta[
        meta[
            "subject_norm"
        ]
        ==
        sid
    ]


    row = {
        "subject_norm":
            sid
    }


    for variable in NEO_COLUMNS:

        numeric = pd.to_numeric(

            g[
                variable
            ],

            errors="coerce"
        )


        vals = np.unique(

            numeric[
                numeric.notna()
            ].to_numpy(
                dtype=float
            )
        )


        if len(
            vals
        ) > 1:

            conflict_rows.append({

                "subject_norm":
                    sid,

                "variable":
                    variable,

                "values":
                    "|".join(
                        map(
                            str,
                            vals.tolist()
                        )
                    )
            })


            row[
                variable
            ] = np.nan


        elif len(
            vals
        ) == 1:

            row[
                variable
            ] = float(
                vals[
                    0
                ]
            )


        else:

            row[
                variable
            ] = np.nan


    subject_rows.append(
        row
    )


neo86 = pd.DataFrame(
    subject_rows
)


assert len(
    neo86
) == 86


assert neo86[
    "subject_norm"
].nunique() == 86


conflict_df = pd.DataFrame(
    conflict_rows
)


CONFLICT_FILE = (
    AUDIT
    / "STAGE0B_NEOFFI_REPEATED_VALUE_CONFLICTS.csv"
)


conflict_df.to_csv(
    CONFLICT_FILE,
    index=False
)


print(
    "\nRepeated-value conflicts:",
    len(
        conflict_df
    )
)


if len(
    conflict_df
) > 0:

    print(
        conflict_df.head(
            50
        ).to_string(
            index=False
        )
    )


    raise RuntimeError(
        "STOP. Repeated NEO-FFI responses conflict for "
        "one or more subjects. Do not arbitrarily average them."
    )


# ============================================================
# 9. ITEM-LEVEL DATA AUDIT
# ============================================================

item_rows = []


for variable in NEO_COLUMNS:

    x = pd.to_numeric(

        neo86[
            variable
        ],

        errors="coerce"
    )


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

        "SD":
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
    / "STAGE0B_NEOFFI60_ITEM_AUDIT.csv"
)


item_audit.to_csv(
    ITEM_AUDIT_FILE,
    index=False
)


zero_variance_items = item_audit.loc[

    item_audit[
        "zero_variance"
    ],

    "variable"

].tolist()


print(
    "\nZero-variance NEO items:",
    zero_variance_items
)


if zero_variance_items:

    raise RuntimeError(
        "One or more NEO-FFI items have zero variance. STOP."
    )


# ============================================================
# 10. SUBJECT-LEVEL COMPLETENESS
# ============================================================

neo86[
    "NEO_observed_items"
] = neo86[
    NEO_COLUMNS
].notna().sum(
    axis=1
)


neo86[
    "NEO_missing_items"
] = (

    60

    -

    neo86[
        "NEO_observed_items"
    ]
)


neo86[
    "NEO_complete60"
] = (

    neo86[
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

        neo86[
            [
                "subject_norm",
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
    / "STAGE0B_NEOFFI60_SUBJECT_COMPLETENESS.csv"
)


completeness.to_csv(
    COMPLETENESS_FILE,
    index=False
)


complete_subjects = completeness[

    completeness[
        "NEO_complete60"
    ]

].copy()


N_COMPLETE = len(
    complete_subjects
)


print(
    "\n" + "=" * 120
)

print(
    "NEO-FFI COMPLETE-CASE COHORT"
)

print(
    "=" * 120
)


print(
    "Complete all 60 items:",
    N_COMPLETE,
    "/ 86"
)


print(
    "\nBy diagnosis:"
)


print(

    complete_subjects[
        "diagnosis_group_standardized"
    ]
    .value_counts()
    .to_string()
)


missing_subjects = completeness[

    ~completeness[
        "NEO_complete60"
    ]

]


print(
    "\nSubjects excluded by complete-case phenotype rule:",
    len(
        missing_subjects
    )
)


if len(
    missing_subjects
) > 0:

    print(

        missing_subjects[
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
# 11. CREATE RAW COMPLETE-CASE NEO MATRIX
#
# NO imputation.
# NO standardization yet.
# ============================================================

complete_ids = complete_subjects[
    "subject_norm"
].tolist()


neo_complete = (

    complete_subjects[
        [
            "subject",
            "subject_norm",
            "diagnosis_group_standardized"
        ]
    ]

    .merge(

        neo86[
            [
                "subject_norm",
                *NEO_COLUMNS
            ]
        ],

        on="subject_norm",

        how="left",

        validate="one_to_one"
    )
)


X_pheno_raw = (

    neo_complete[
        NEO_COLUMNS
    ]

    .to_numpy(
        dtype=np.float64
    )
)


assert X_pheno_raw.shape == (
    N_COMPLETE,
    60
)


assert np.isfinite(
    X_pheno_raw
).all()


assert np.all(

    np.std(
        X_pheno_raw,
        axis=0,
        ddof=0
    )

    >
    0
)


PHENO_RAW_NPY = (
    INPUTS
    / "PRIMARY_NEOFFI60_RAW.npy"
)


PHENO_RAW_CSV = (
    INPUTS
    / "PRIMARY_NEOFFI60_RAW.csv"
)


np.save(
    PHENO_RAW_NPY,
    X_pheno_raw.astype(
        np.float32
    )
)


neo_complete.to_csv(
    PHENO_RAW_CSV,
    index=False
)


# ============================================================
# 12. ALIGN STAGE7G EQUAL-WINDOW EEG128
#
# NO cluster labels are loaded.
# ============================================================

selection = pd.read_csv(

    EQUAL8_WINDOW_SELECTION,

    dtype={
        "subject": str
    }
)


selection_subjects = sorted(

    selection[
        "subject"
    ]
    .astype(str)
    .unique()
)


assert len(
    selection_subjects
) == 86


selection_subject_norm = [

    normalize_id(
        x
    )

    for x in selection_subjects
]


assert len(
    set(
        selection_subject_norm
    )
) == 86


eeg86 = np.load(
    EQUAL8_EEG128
)


assert eeg86.shape == (
    86,
    128
)


assert np.isfinite(
    eeg86
).all()


eeg_index = {

    sid:
        i

    for i, sid in enumerate(
        selection_subject_norm
    )
}


missing_from_equal8 = [

    sid

    for sid in complete_ids

    if sid not in eeg_index
]


if missing_from_equal8:

    raise RuntimeError(
        "Complete phenotype subjects missing from "
        "equal-window EEG representation: "
        +
        str(
            missing_from_equal8
        )
    )


eeg_rows = [

    eeg_index[
        sid
    ]

    for sid in complete_ids
]


X_eeg_raw = eeg86[
    eeg_rows
].astype(
    np.float32
)


assert X_eeg_raw.shape == (
    N_COMPLETE,
    128
)


assert np.isfinite(
    X_eeg_raw
).all()


EEG_RAW_NPY = (
    INPUTS
    / "PRIMARY_EQUAL8_EEG128_RAW.npy"
)


np.save(
    EEG_RAW_NPY,
    X_eeg_raw
)


# ============================================================
# 13. SAVE AUTHORITATIVE EXPERIMENT3 SUBJECT ORDER
# ============================================================

subject_order = complete_subjects[
    [
        "subject",
        "subject_norm",
        "diagnosis_group_standardized"
    ]
].copy()


subject_order[
    "experiment3_row"
] = np.arange(
    N_COMPLETE
)


SUBJECT_ORDER_FILE = (
    LOCK
    / "STAGE0B_PRIMARY_FUSION_COHORT.csv"
)


subject_order.to_csv(
    SUBJECT_ORDER_FILE,
    index=False
)


# ============================================================
# 14. PRIMARY / HELD-OUT VARIABLE POLICY
# ============================================================

PRIMARY_PHENOTYPE = [

    "neoFFI_q1 ... neoFFI_q60"
]


EXCLUDED_FROM_PRIMARY_PHENOTYPE = [

    "diagnosis",

    "age",

    "sex/gender",

    "education",

    "cigarette",

    "coffee",

    "beer",

    "drugs",

    "meal",

    "sleep",

    "well",

    "oddball performance variables",

    "ADHD_pre_Att_leading",

    "ADHD_pre_Hyp_leading",

    "YBOCS_pre",

    "BDI_pre",

    "all post-treatment variables"
]


HELD_OUT_CHARACTERIZATION = [

    "diagnosis",

    "ADHD_pre_Att_leading",

    "ADHD_pre_Hyp_leading",

    "YBOCS_pre",

    "BDI_pre",

    "oddball behavioral measures"
]


CONFOUND_AUDIT_VARIABLES = [

    "age",

    "sex/gender",

    "education",

    "sleep",

    "well",

    "cigarette",

    "coffee",

    "beer",

    "drugs",

    "meal"
]


# ============================================================
# 15. PROSPECTIVELY FREEZE FUSION DESIGN
#
# BEFORE any graph construction or clustering.
# ============================================================

DESIGN_LOCK = (
    LOCK
    / "STAGE0B_PRIMARY_FUSION_DESIGN_LOCK.json"
)


design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE0B_PRIMARY_FUSION_DESIGN_LOCK",

    "status":
        "FROZEN_BEFORE_GRAPH_CONSTRUCTION",

    "source_cohort_N":
        86,

    "primary_fusion_cohort_rule":
        "complete_case_on_all_60_NEOFFI_items_no_imputation",

    "primary_fusion_cohort_N":
        N_COMPLETE,

    "primary_EEG_view":
        {

            "source":
                "Experiment2 Stage7G equal-window representation",

            "features":
                128,

            "restEO_clean_windows_per_subject":
                8,

            "restEC_clean_windows_per_subject":
                8,

            "technical_reason":
                (
                    "Uses equal-window representation to avoid "
                    "known clean-window-count imbalance from the "
                    "Experiment2 primary analysis."
                )
        },

    "primary_phenotype_view":
        {

            "source":
                "TDBRAIN item-level NEO-FFI",

            "features":
                60,

            "variables":
                NEO_COLUMNS,

            "official_trait_scoring_applied":
                False,

            "reverse_coding_guessed":
                False,

            "representation_description":
                "raw item-response phenotype vector",

            "imputation":
                False
        },

    "pre_graph_standardization":
        {

            "EEG":
                "z-score feature-wise within Experiment3 cohort",

            "phenotype":
                "z-score item-wise within Experiment3 cohort"
        },

    "similarity":
        {

            "EEG":
                "cosine",

            "phenotype":
                "cosine"
        },

    "fusion":
        {

            "formula":
                "A_fused = alpha*A_EEG + (1-alpha)*A_PHENOTYPE",

            "alpha":
                0.6,

            "interpretation":
                "60_percent_EEG_40_percent_phenotype",

            "alpha_selected_using_clinical_outcomes":
                False,

            "alpha_reselection_permitted":
                False
        },

    "graph_neighborhood":
        {

            "candidate_k":
                [
                    4,
                    6,
                    8,
                    10,
                    12
                ],

            "selection_rule":
                (
                    "smallest k producing a connected graph "
                    "with zero isolated nodes"
                ),

            "clinical_variables_used":
                False
        },

    "VGAE":
        {

            "node_feature_dimension":
                188,

            "node_features":
                "standardized_EEG128_concatenated_with_standardized_NEO60",

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
                1e-3,

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

    "clustering":
        {

            "method":
                "spectral_clustering_on_exact_4NN_UMAP_affinity",

            "candidate_K":
                [
                    2,
                    3,
                    4,
                    5,
                    6
                ],

            "K_selection":
                "EEG_phenotype_internal_criteria_and_stability_only",

            "minimum_cluster_size":
                5,

            "maximum_balance_ratio":
                5.0,

            "clinical_variables_used_for_K_selection":
                False
        },

    "held_out_characterization":
        HELD_OUT_CHARACTERIZATION,

    "confound_audit":
        CONFOUND_AUDIT_VARIABLES,

    "forbidden_for_model_selection":
        [

            "diagnosis",

            "ADHD_pre_Att_leading",

            "ADHD_pre_Hyp_leading",

            "YBOCS_pre",

            "BDI_pre",

            "post-treatment variables",

            "Experiment2 cluster labels"
        ],

    "experiment2_primary_partition_modified":
        False,

    "input_hashes":
        {

            "Stage0A_audit":
                sha256_file(
                    STAGE0A_AUDIT
                ),

            "Stage0A_candidates":
                sha256_file(
                    STAGE0A_CANDIDATES
                ),

            "Stage0A_lock":
                sha256_file(
                    STAGE0A_LOCK
                ),

            "TDBRAIN_metadata":
                sha256_file(
                    META
                ),

            "Experiment2_final_subject_table":
                sha256_file(
                    EXP2_FINAL_SUBJECTS
                ),

            "equal8_EEG128":
                sha256_file(
                    EQUAL8_EEG128
                ),

            "equal8_window_selection":
                sha256_file(
                    EQUAL8_WINDOW_SELECTION
                )
        },

    "output_hashes":
        {

            "phenotype_raw_npy":
                sha256_file(
                    PHENO_RAW_NPY
                ),

            "phenotype_raw_csv":
                sha256_file(
                    PHENO_RAW_CSV
                ),

            "EEG_raw_npy":
                sha256_file(
                    EEG_RAW_NPY
                ),

            "subject_order":
                sha256_file(
                    SUBJECT_ORDER_FILE
                ),

            "item_audit":
                sha256_file(
                    ITEM_AUDIT_FILE
                ),

            "subject_completeness":
                sha256_file(
                    COMPLETENESS_FILE
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
            "Existing Stage0B design lock differs. STOP."
        )


    print(
        "\nExisting Stage0B design lock verified."
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
        "\nStage0B fusion design lock created BEFORE graph construction."
    )


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

SUMMARY_FILE = (
    AUDIT
    / "STAGE0B_PHENOTYPE_AND_FUSION_DESIGN_SUMMARY.json"
)


summary = {

    "source_N":
        86,

    "NEOFFI_items":
        60,

    "complete_case_N":
        N_COMPLETE,

    "complete_case_diagnosis_counts":
        complete_subjects[
            "diagnosis_group_standardized"
        ].value_counts().to_dict(),

    "repeated_value_conflicts":
        len(
            conflict_df
        ),

    "zero_variance_items":
        zero_variance_items,

    "EEG_source":
        "Stage7G_equal_window_EEG128",

    "EEG_dimension":
        128,

    "phenotype_dimension":
        60,

    "future_node_feature_dimension":
        188,

    "fusion_alpha":
        0.6,

    "diagnosis_used_as_input":
        False,

    "clinical_scores_used_as_input":
        False,

    "Experiment2_cluster_labels_used":
        False,

    "post_treatment_used":
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
# 17. PRINT FINAL AUDIT
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "EXPERIMENT 3 PRIMARY INPUT DESIGN"
)

print(
    "=" * 120
)


print(
    "Source cohort:",
    86
)

print(
    "Complete-case NEO-FFI cohort:",
    N_COMPLETE
)


print(
    "\nComplete-case diagnosis counts:"
)

print(

    complete_subjects[
        "diagnosis_group_standardized"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nEEG view:"
)

print(
    "Stage7G equal-window EEG128"
)


print(
    "\nPhenotype view:"
)

print(
    "NEO-FFI item-level 60-dimensional vector"
)


print(
    "\nFuture fused VGAE node features:"
)

print(
    "EEG128 + NEO60 = 188 dimensions"
)


print(
    "\nFusion:"
)

print(
    "alpha = 0.6"
)

print(
    "EEG contribution = 60%"
)

print(
    "Phenotype contribution = 40%"
)


print(
    "\nPrimary phenotype includes diagnosis: NO"
)

print(
    "Primary phenotype includes ADHD scores: NO"
)

print(
    "Primary phenotype includes YBOCS: NO"
)

print(
    "Primary phenotype includes BDI: NO"
)

print(
    "Experiment2 cluster labels used: NO"
)

print(
    "Post-treatment variables used: NO"
)


print(
    "\nHeld out for later characterization:"
)

for x in HELD_OUT_CHARACTERIZATION:

    print(
        " -",
        x
    )


print(
    "\nHASHES"
)

print(
    "NEO raw matrix:",
    sha256_file(
        PHENO_RAW_NPY
    )
)

print(
    "EEG raw matrix:",
    sha256_file(
        EEG_RAW_NPY
    )
)

print(
    "Subject order:",
    sha256_file(
        SUBJECT_ORDER_FILE
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
    "FINAL STATUS: EXPERIMENT 3 STAGE 0B PASS — "
    "PHENOTYPE FEATURE SET, PRIMARY COHORT RULE, "
    "AND FUSION DESIGN FROZEN"
)

print(
    "=" * 120
)


print(
    "\nNO FUSED GRAPH HAS BEEN CREATED YET."
)

print(
    "NO CLUSTERING HAS BEEN RUN."
)

print(
    "NEXT: STAGE 1A — STANDARDIZE EEG/NEO VIEWS "
    "AND BUILD THE FROZEN alpha=0.6 FUSED GRAPH."
)