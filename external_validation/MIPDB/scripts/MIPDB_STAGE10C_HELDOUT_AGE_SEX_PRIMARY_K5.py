#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

from scipy.stats import (
    f_oneway,
    kruskal,
    chi2_contingency,
)


# ======================================================================
# PATHS
# ======================================================================

DRIVE = Path(
    "/content/drive/MyDrive/FULL_23PHD1241_BACKUP_2026_07_04"
)

EXT = (
    DRIVE
    / "EPGSD_EXTERNAL_VALIDATION_MIPDB_20261002"
    / "EXTERNAL_VALIDATION_MIPDB"
)

AUDIT = EXT / "00_AUDIT"

PRIMARY = (
    EXT
    / "08_CLUSTERING"
    / "EEG_ONLY"
    / "PRIMARY_K5"
)

ASSIGN_FILE = (
    PRIMARY
    / "MIPDB_PRIMARY_K5_SUBTYPE_ASSIGNMENTS.csv"
)

PRIMARY_METRICS = (
    PRIMARY
    / "MIPDB_PRIMARY_K5_METRICS.json"
)

AGE_MATCH = (
    AUDIT
    / "MIPDB_PRIMARY_AGE_MATCHED_SUBJECTS.csv"
)

PUBLIC = (
    EXT
    / "02_METADATA"
    / "MIPDB_ORIGINAL_DOCUMENTATION"
    / "MIPDB_PublicFile.csv"
)

OUT = (
    EXT
    / "09_RESULTS"
    / "EEG_ONLY"
    / "PRIMARY_K5_HELDOUT"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_N = 91
EXPECTED_K = 5


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def norm_id(x):

    s = str(x).strip()

    return re.sub(
        r"^sub-",
        "",
        s,
        flags=re.I
    )


def eta_omega_squared(values, groups):

    values = np.asarray(
        values,
        dtype=float
    )

    groups = np.asarray(
        groups
    )

    unique = np.unique(
        groups
    )

    grand_mean = np.mean(
        values
    )

    ss_between = 0.0

    ss_within = 0.0

    for g in unique:

        x = values[
            groups == g
        ]

        mean_g = np.mean(
            x
        )

        ss_between += (
            len(x)
            * (
                mean_g
                - grand_mean
            ) ** 2
        )

        ss_within += np.sum(
            (
                x
                - mean_g
            ) ** 2
        )


    ss_total = (
        ss_between
        + ss_within
    )


    df_between = (
        len(unique)
        - 1
    )

    df_within = (
        len(values)
        - len(unique)
    )


    ms_within = (
        ss_within
        / df_within
    )


    eta2 = (
        ss_between
        / ss_total
        if ss_total > 0
        else np.nan
    )


    omega2 = (
        (
            ss_between
            - df_between
            * ms_within
        )
        /
        (
            ss_total
            + ms_within
        )
        if (
            ss_total
            + ms_within
        ) > 0
        else np.nan
    )


    return {
        "ss_between":
            float(ss_between),

        "ss_within":
            float(ss_within),

        "ss_total":
            float(ss_total),

        "df_between":
            int(df_between),

        "df_within":
            int(df_within),

        "eta_squared":
            float(eta2),

        "omega_squared":
            float(omega2),
    }


# ======================================================================
# START
# ======================================================================

print("=" * 78)
print("MIPDB STAGE 10C — PRIMARY K=5 HELD-OUT AGE / SEX")
print("POST-CLUSTER EXTERNAL CHARACTERIZATION")
print("AGE AND SEX WERE NOT USED FOR MODEL FITTING OR CLUSTERING")
print("=" * 78)


# ======================================================================
# 1. VERIFY LOCKED PRIMARY K=5
# ======================================================================

print()
print("=" * 78)
print("1. VERIFY PRIMARY K=5 RESULT")
print("=" * 78)


for p in [
    ASSIGN_FILE,
    PRIMARY_METRICS,
    AGE_MATCH,
    PUBLIC,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


primary = json.loads(
    PRIMARY_METRICS.read_text(
        encoding="utf-8"
    )
)


if primary.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Primary clustering is not PASS."
    )


if primary.get(
    "spectral_k"
) != 5:

    raise RuntimeError(
        "Primary clustering is not K=5."
    )


assign = pd.read_csv(
    ASSIGN_FILE
)


if len(assign) != EXPECTED_N:

    raise RuntimeError(
        f"Expected {EXPECTED_N} assignments, "
        f"found {len(assign)}"
    )


if assign[
    "subject"
].duplicated().any():

    raise RuntimeError(
        "Duplicate subjects in assignment file."
    )


if assign[
    "subtype"
].nunique() != EXPECTED_K:

    raise RuntimeError(
        "Primary assignment does not have five subtypes."
    )


print(
    "SUBJECTS =",
    len(assign)
)

print(
    "PRIMARY_K =",
    assign[
        "subtype"
    ].nunique()
)

print(
    "CLUSTER_SIZES =",
    assign[
        "subtype"
    ].value_counts().sort_index().to_dict()
)


# ======================================================================
# 2. LOAD / CROSS-CHECK METADATA
# ======================================================================

print()
print("=" * 78)
print("2. VERIFY AGE / SEX METADATA")
print("=" * 78)


am = pd.read_csv(
    AGE_MATCH
)


pub = pd.read_csv(
    PUBLIC
)


assign["_id"] = (
    assign[
        "subject"
    ].map(
        norm_id
    )
)


am["_id"] = (
    am[
        "participant_id"
    ].map(
        norm_id
    )
)


pub["_id"] = (
    pub[
        "ID"
    ].map(
        norm_id
    )
)


final_ids = set(
    assign["_id"]
)


am = am[
    am["_id"].isin(
        final_ids
    )
].copy()


pub = pub[
    pub["_id"].isin(
        final_ids
    )
].copy()


if len(am) != EXPECTED_N:

    raise RuntimeError(
        f"Age-matched metadata overlap = {len(am)}, "
        f"expected {EXPECTED_N}"
    )


if len(pub) != EXPECTED_N:

    raise RuntimeError(
        f"Public metadata overlap = {len(pub)}, "
        f"expected {EXPECTED_N}"
    )


cross = am[
    [
        "_id",
        "age",
        "sex",
    ]
].merge(

    pub[
        [
            "_id",
            "Age",
            "Sex",
        ]
    ],

    on="_id",

    how="inner"
)


age_a = pd.to_numeric(
    cross["age"],
    errors="coerce"
)


age_b = pd.to_numeric(
    cross["Age"],
    errors="coerce"
)


age_match_numeric = np.isclose(
    age_a.to_numpy(),
    age_b.to_numpy(),
    equal_nan=True
)


sex_match = (
    pd.to_numeric(
        cross["sex"],
        errors="coerce"
    ).to_numpy()
    ==
    pd.to_numeric(
        cross["Sex"],
        errors="coerce"
    ).to_numpy()
)


print(
    "AGE_NUMERIC_MATCH =",
    int(
        age_match_numeric.sum()
    ),
    "/",
    EXPECTED_N
)

print(
    "SEX_CODE_MATCH =",
    int(
        sex_match.sum()
    ),
    "/",
    EXPECTED_N
)


if not age_match_numeric.all():

    bad = cross.loc[
        ~age_match_numeric,
        [
            "_id",
            "age",
            "Age",
        ]
    ]

    print(
        bad.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Age values differ numerically "
        "between metadata sources."
    )


if not sex_match.all():

    raise RuntimeError(
        "Sex values differ "
        "between metadata sources."
    )


print(
    "METADATA_CROSSCHECK=PASS"
)


# ======================================================================
# 3. MERGE HELD-OUT VARIABLES WITH PRIMARY LABELS
# ======================================================================

held = assign.merge(

    am[
        [
            "_id",
            "age",
            "sex",
            "adult_handedness",
            "child_handedness",
        ]
    ],

    on="_id",

    how="left",

    validate="one_to_one"
)


held["age"] = pd.to_numeric(
    held["age"],
    errors="coerce"
)


held["sex"] = pd.to_numeric(
    held["sex"],
    errors="coerce"
)


print()
print("=" * 78)
print("3. FINAL HELD-OUT DATA COVERAGE")
print("=" * 78)

print(
    "AGE_NON_NULL =",
    int(
        held[
            "age"
        ].notna().sum()
    )
)

print(
    "SEX_NON_NULL =",
    int(
        held[
            "sex"
        ].notna().sum()
    )
)

print(
    "SEX_CODES =",
    sorted(
        held[
            "sex"
        ].dropna().unique().tolist()
    )
)


if held[
    "age"
].notna().sum() != EXPECTED_N:

    raise RuntimeError(
        "Age is not complete for all 91 subjects."
    )


if held[
    "sex"
].notna().sum() != EXPECTED_N:

    raise RuntimeError(
        "Sex is not complete for all 91 subjects."
    )


# ======================================================================
# 4. AGE — SAME HELD-OUT METHODS AS HBN
# ======================================================================

print()
print("=" * 78)
print("4. HELD-OUT AGE ASSOCIATION")
print("=" * 78)


age_groups = [

    held.loc[
        held[
            "subtype"
        ] == k,
        "age"
    ].to_numpy(
        dtype=float
    )

    for k in range(
        1,
        EXPECTED_K + 1
    )
]


anova = f_oneway(
    *age_groups
)


kw = kruskal(
    *age_groups
)


effect = eta_omega_squared(

    held[
        "age"
    ].to_numpy(
        dtype=float
    ),

    held[
        "subtype"
    ].to_numpy()
)


age_desc_rows = []


for k in range(
    1,
    EXPECTED_K + 1
):

    x = held.loc[
        held[
            "subtype"
        ] == k,
        "age"
    ].to_numpy(
        dtype=float
    )


    age_desc_rows.append({

        "subtype":
            k,

        "n":
            len(x),

        "mean_age":
            float(
                np.mean(x)
            ),

        "std_age":
            float(
                np.std(
                    x,
                    ddof=1
                )
            )
            if len(x) > 1
            else np.nan,

        "median_age":
            float(
                np.median(x)
            ),

        "min_age":
            float(
                np.min(x)
            ),

        "max_age":
            float(
                np.max(x)
            ),
    })


age_desc = pd.DataFrame(
    age_desc_rows
)


print(
    age_desc.to_string(
        index=False
    )
)

print()
print(
    "AGE_ANOVA_F =",
    float(
        anova.statistic
    )
)

print(
    "AGE_ANOVA_P =",
    float(
        anova.pvalue
    )
)

print(
    "AGE_KRUSKAL_H =",
    float(
        kw.statistic
    )
)

print(
    "AGE_KRUSKAL_P =",
    float(
        kw.pvalue
    )
)

print(
    "AGE_ETA_SQUARED =",
    effect[
        "eta_squared"
    ]
)

print(
    "AGE_OMEGA_SQUARED =",
    effect[
        "omega_squared"
    ]
)


# ======================================================================
# 5. SEX — PEARSON CHI-SQUARE + CRAMER'S V
# ======================================================================

print()
print("=" * 78)
print("5. HELD-OUT SEX ASSOCIATION")
print("=" * 78)


table = pd.crosstab(

    held[
        "subtype"
    ],

    held[
        "sex"
    ]
)


chi2, p_chi, dof, expected = (
    chi2_contingency(
        table.to_numpy(),
        correction=False
    )
)


n = int(
    table.to_numpy().sum()
)


r, c = table.shape


den = min(
    r - 1,
    c - 1
)


cramers_v = (
    float(
        np.sqrt(
            chi2
            /
            (
                n
                * den
            )
        )
    )
    if den > 0
    else np.nan
)


print(
    "SEX_CONTINGENCY_TABLE:"
)

print(
    table.to_string()
)

print()

print(
    "SEX_CHI2 =",
    float(chi2)
)

print(
    "SEX_DF =",
    int(dof)
)

print(
    "SEX_P =",
    float(p_chi)
)

print(
    "SEX_CRAMERS_V =",
    cramers_v
)

print(
    "SEX_MIN_EXPECTED_COUNT =",
    float(
        expected.min()
    )
)

print(
    "SEX_EXPECTED_LT5 =",
    int(
        np.sum(
            expected < 5
        )
    )
)


# ======================================================================
# 6. HANDEDNESS — DOCUMENTED DECISION
# ======================================================================

print()
print("=" * 78)
print("6. HANDEDNESS DECISION")
print("=" * 78)


child_n = int(
    held[
        "child_handedness"
    ].notna().sum()
)


adult_n = int(
    held[
        "adult_handedness"
    ].notna().sum()
)


print(
    "CHILD_HANDEDNESS_NON_NULL =",
    child_n
)

print(
    "ADULT_HANDEDNESS_NON_NULL =",
    adult_n
)

print(
    "HANDEDNESS_TEST_PERFORMED=NO"
)

print(
    "REASON="
    "MIPDB 1-5 child-handedness coding was not "
    "defined in the audited documentation, and adult "
    "handedness is available for only 3 final subjects."
)


# ======================================================================
# 7. SAVE OUTPUTS
# ======================================================================

held_file = (
    OUT
    / "MIPDB_PRIMARY_K5_HELDOUT_SUBJECT_DATA.csv"
)


age_file = (
    OUT
    / "MIPDB_PRIMARY_K5_AGE_BY_SUBTYPE.csv"
)


sex_file = (
    OUT
    / "MIPDB_PRIMARY_K5_SEX_CONTINGENCY.csv"
)


summary_file = (
    OUT
    / "MIPDB_PRIMARY_K5_HELDOUT_SUMMARY.json"
)


held.to_csv(
    held_file,
    index=False
)


age_desc.to_csv(
    age_file,
    index=False
)


table.to_csv(
    sex_file
)


summary = {

    "validation_branch":
        "MIPDB_EEG_ONLY_EXTERNAL_COMPONENT_VALIDATION",

    "analysis":
        "primary_K5_heldout_age_sex",

    "subjects":
        EXPECTED_N,

    "primary_k":
        EXPECTED_K,

    "age": {

        "n":
            EXPECTED_N,

        "anova_F":
            float(
                anova.statistic
            ),

        "anova_p":
            float(
                anova.pvalue
            ),

        "kruskal_H":
            float(
                kw.statistic
            ),

        "kruskal_p":
            float(
                kw.pvalue
            ),

        "eta_squared":
            effect[
                "eta_squared"
            ],

        "omega_squared":
            effect[
                "omega_squared"
            ],
    },

    "sex": {

        "n":
            n,

        "codes":
            sorted(
                held[
                    "sex"
                ].unique().tolist()
            ),

        "code_labels_interpreted":
            False,

        "chi_square":
            float(chi2),

        "degrees_of_freedom":
            int(dof),

        "p_value":
            float(p_chi),

        "cramers_v":
            cramers_v,

        "minimum_expected_count":
            float(
                expected.min()
            ),

        "expected_cells_below_5":
            int(
                np.sum(
                    expected < 5
                )
            ),
    },

    "handedness": {

        "test_performed":
            False,

        "child_non_null":
            child_n,

        "adult_non_null":
            adult_n,

        "reason":
            (
                "Child handedness coding 1-5 was "
                "not defined in the audited documentation; "
                "adult handedness was available for only "
                "3 final subjects."
            ),
    },

    "age_used_for_model_selection":
        False,

    "sex_used_for_model_selection":
        False,

    "heldout_analysis_post_clustering":
        True,

    "status":
        "PASS",
}


summary_file.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ======================================================================
# 8. LOCK
# ======================================================================

lock_file = (
    AUDIT
    / "MIPDB_STAGE10C_PRIMARY_K5_HELDOUT_LOCK.txt"
)


lock_file.write_text(
f"""======================================================================
EPG-SD MIPDB EXTERNAL VALIDATION — STAGE 10C HELD-OUT LOCK
======================================================================

PRIMARY_K=5
SUBJECTS=91

AGE_AVAILABLE=91/91
SEX_AVAILABLE=91/91

AGE_METADATA_NUMERIC_CROSSCHECK=PASS
SEX_METADATA_CROSSCHECK=PASS

AGE_ANOVA_F={float(anova.statistic)}
AGE_ANOVA_P={float(anova.pvalue)}
AGE_KRUSKAL_H={float(kw.statistic)}
AGE_KRUSKAL_P={float(kw.pvalue)}
AGE_ETA_SQUARED={effect["eta_squared"]}
AGE_OMEGA_SQUARED={effect["omega_squared"]}

SEX_CHI2={float(chi2)}
SEX_DF={int(dof)}
SEX_P={float(p_chi)}
SEX_CRAMERS_V={cramers_v}
SEX_MIN_EXPECTED_COUNT={float(expected.min())}
SEX_EXPECTED_CELLS_LT5={int(np.sum(expected < 5))}

SEX_CODES_INTERPRETED_AS_MALE_FEMALE=NO

CHILD_HANDEDNESS_AVAILABLE={child_n}/91
ADULT_HANDEDNESS_AVAILABLE={adult_n}/91
HANDEDNESS_TEST_PERFORMED=NO

AGE_USED_FOR_CLUSTER_SELECTION=NO
SEX_USED_FOR_CLUSTER_SELECTION=NO
HELDOUT_ANALYSIS_PERFORMED_AFTER_PRIMARY_LOCK=YES

SUMMARY_FILE=
{summary_file}

STAGE10C_STATUS=PASS
======================================================================
""",
    encoding="utf-8"
)


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 78)
print("STAGE 10C COMPLETE")
print("=" * 78)

print(
    "AGE_N=91"
)

print(
    "AGE_ANOVA_P=",
    float(
        anova.pvalue
    )
)

print(
    "AGE_KRUSKAL_P=",
    float(
        kw.pvalue
    )
)

print(
    "AGE_ETA_SQUARED=",
    effect[
        "eta_squared"
    ]
)

print(
    "AGE_OMEGA_SQUARED=",
    effect[
        "omega_squared"
    ]
)

print()

print(
    "SEX_N=91"
)

print(
    "SEX_CHI2=",
    float(chi2)
)

print(
    "SEX_DF=",
    int(dof)
)

print(
    "SEX_P=",
    float(p_chi)
)

print(
    "SEX_CRAMERS_V=",
    cramers_v
)

print()

print(
    "HANDEDNESS_TEST_PERFORMED=NO"
)

print(
    "AGE_USED_FOR_CLUSTER_SELECTION=NO"
)

print(
    "SEX_USED_FOR_CLUSTER_SELECTION=NO"
)

print(
    "STAGE10C_STATUS=PASS"
)

print()
print(
    "OUTPUT_DIR =",
    OUT
)

print(
    "LOCK_FILE =",
    lock_file
)

print("=" * 78)

