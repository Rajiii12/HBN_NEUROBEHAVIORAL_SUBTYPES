# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
#
# STAGE 0B-R1 — BASELINE SESSION RESOLUTION AUDIT
#
# PURPOSE
# -------
# Resolve repeated metadata rows scientifically.
#
# We need the NEO-FFI responses belonging to the SAME
# baseline session used for the primary EEG:
#
#       session 1 / baseline
#
# THIS STAGE:
#   DOES NOT average NEO responses
#   DOES NOT build phenotype features
#   DOES NOT build graph
#   DOES NOT cluster
#   DOES NOT use Experiment2 cluster labels
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import hashlib


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

AUDIT.mkdir(
    parents=True,
    exist_ok=True
)

LOCK.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 0B-R1 — BASELINE SESSION RESOLUTION AUDIT")
print("=" * 120)


# ============================================================
# 2. INPUTS
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


for p in [
    META,
    EXP2_FINAL_SUBJECTS
]:

    if not p.exists():

        raise FileNotFoundError(
            str(p)
        )


# ============================================================
# 3. HELPERS
# ============================================================

def normalize_id(x):

    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.lower() in {
        "",
        "nan",
        "none"
    }:
        return None

    if s.startswith("sub-"):
        s = s[4:]

    if s.endswith(".0"):
        s = s[:-2]

    return s


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


def parse_session_number(value, subject_id=None):

    if pd.isna(value):
        return None

    s = str(value).strip()

    if not s:
        return None


    # --------------------------------------------------------
    # Explicit baseline words
    # --------------------------------------------------------

    if re.fullmatch(
        r"(?i)baseline",
        s
    ):
        return 1


    # --------------------------------------------------------
    # ses-1 / ses1 / session-1 / session 1
    # --------------------------------------------------------

    patterns = [

        r"(?i)^ses[-_ ]?0*([1-9]\d*)$",

        r"(?i)^session[-_ ]?0*([1-9]\d*)$",

        r"(?i)^visit[-_ ]?0*([1-9]\d*)$",

        r"(?i)^wave[-_ ]?0*([1-9]\d*)$"
    ]


    for pattern in patterns:

        m = re.match(
            pattern,
            s
        )

        if m:

            return int(
                m.group(1)
            )


    # --------------------------------------------------------
    # Published TDBRAIN convention:
    # participantID-sessionnumber
    #
    # Example:
    # 88012461-1
    # --------------------------------------------------------

    if subject_id is not None:

        sid = str(
            subject_id
        ).strip()

        m = re.match(
            rf"^{re.escape(sid)}[-_](\d+)$",
            s
        )

        if m:

            return int(
                m.group(1)
            )


    # --------------------------------------------------------
    # Bare integer session values
    # --------------------------------------------------------

    try:

        x = float(s)

        if (
            np.isfinite(x)
            and
            x.is_integer()
            and
            1 <= int(x) <= 20
        ):

            return int(x)

    except Exception:

        pass


    return None


# ============================================================
# 4. FROZEN N=86 SUBJECTS
#
# Load only subject + diagnosis.
# DO NOT load Experiment2 cluster labels.
# ============================================================

cohort = pd.read_csv(

    EXP2_FINAL_SUBJECTS,

    usecols=[
        "subject",
        "diagnosis_group_standardized"
    ]
)


assert len(cohort) == 86

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


subject_set = set(
    cohort[
        "subject_norm"
    ]
)


print(
    "\nFrozen source subjects:",
    len(subject_set)
)


# ============================================================
# 5. LOAD COMPLETE METADATA
#
# We need all columns ONLY to identify the session field.
# No values are used for clustering here.
# ============================================================

meta = pd.read_excel(
    META,
    sheet_name=SHEET
)


if "TDBRAIN_ID" not in meta.columns:

    raise RuntimeError(
        "TDBRAIN_ID not found."
    )


meta[
    "subject_norm"
] = meta[
    "TDBRAIN_ID"
].map(
    normalize_id
)


m86 = meta[

    meta[
        "subject_norm"
    ].isin(
        subject_set
    )

].copy()


m86[
    "_excel_row"
] = m86.index + 2


print(
    "Metadata rows:",
    len(m86)
)


# ============================================================
# 6. IDENTIFY REPEATED SUBJECTS
# ============================================================

row_counts = (

    m86[
        "subject_norm"
    ]
    .value_counts()
)


duplicates = row_counts[
    row_counts > 1
]


print(
    "\nRepeated subjects:",
    len(duplicates)
)

print(
    "Extra metadata rows:",
    int(
        len(m86)
        -
        m86[
            "subject_norm"
        ].nunique()
    )
)


print(
    "\nRepeated subject row counts:"
)

print(
    duplicates.to_string()
)


duplicate_ids = set(
    duplicates.index
)


dup = m86[

    m86[
        "subject_norm"
    ].isin(
        duplicate_ids
    )

].copy()


# ============================================================
# 7. FIND LIKELY SESSION COLUMNS BY COLUMN NAME
# ============================================================

name_tokens = [

    "session",
    "sess",
    "visit",
    "wave",
    "measurement",
    "measure",
    "recording"
]


name_candidate_columns = [

    c

    for c in meta.columns

    if any(
        token in str(c).lower()
        for token in name_tokens
    )
]


print(
    "\n" + "=" * 120
)

print(
    "SESSION-LIKE COLUMNS BY COLUMN NAME"
)

print(
    "=" * 120
)


print(
    name_candidate_columns
)


# ============================================================
# 8. FIND SESSION COLUMNS BY VALUE PATTERN
#
# A candidate is useful if its values can be interpreted as
# session numbers for the duplicated subjects.
# ============================================================

candidate_rows = []


for column in meta.columns:

    if column in {
        "TDBRAIN_ID",
        "subject_norm"
    }:
        continue


    parsed = []


    for _, row in dup.iterrows():

        session_number = parse_session_number(

            row[
                column
            ],

            subject_id=row[
                "subject_norm"
            ]
        )


        parsed.append(
            session_number
        )


    parsed_nonmissing = [

        x

        for x in parsed

        if x is not None
    ]


    coverage = (

        len(
            parsed_nonmissing
        )

        /
        len(
            dup
        )

        if len(
            dup
        ) > 0

        else 0
    )


    unique_sessions = sorted(
        set(
            parsed_nonmissing
        )
    )


    if (
        coverage >= 0.50
        and
        len(unique_sessions) >= 1
    ):

        candidate_rows.append({

            "column":
                column,

            "duplicate_row_parse_coverage":
                coverage,

            "parsed_session_numbers":
                "|".join(
                    map(
                        str,
                        unique_sessions
                    )
                ),

            "column_name_session_like":
                any(
                    token
                    in
                    str(column).lower()

                    for token in name_tokens
                )
        })


candidate_table = pd.DataFrame(
    candidate_rows
)


if len(candidate_table) > 0:

    candidate_table = candidate_table.sort_values(

        [
            "column_name_session_like",
            "duplicate_row_parse_coverage"
        ],

        ascending=[
            False,
            False
        ]
    )


print(
    "\n" + "=" * 120
)

print(
    "SESSION CANDIDATES BY VALUE PATTERN"
)

print(
    "=" * 120
)


if len(candidate_table) == 0:

    print(
        "No session-number column was automatically identified."
    )

else:

    print(

        candidate_table.to_string(

            index=False,

            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 9. SHOW NON-NEO COLUMNS THAT DIFFER ACROSS REPEATED ROWS
#
# This helps identify session metadata without printing
# all 60 questionnaire values.
# ============================================================

NEO_COLUMNS = [

    f"neoFFI_q{i}"

    for i in range(
        1,
        61
    )
]


difference_rows = []


for column in m86.columns:

    if column in (
        NEO_COLUMNS
        +
        [
            "subject_norm",
            "_excel_row"
        ]
    ):
        continue


    differing_subjects = []


    for sid in duplicate_ids:

        g = dup[
            dup[
                "subject_norm"
            ]
            ==
            sid
        ]


        vals = (

            g[
                column
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )


        if len(vals) > 1:

            differing_subjects.append(
                sid
            )


    if differing_subjects:

        difference_rows.append({

            "column":
                column,

            "n_duplicate_subjects_with_different_values":
                len(
                    differing_subjects
                ),

            "subjects":
                "|".join(
                    map(
                        str,
                        sorted(
                            differing_subjects
                        )
                    )
                )
        })


difference_table = pd.DataFrame(
    difference_rows
)


print(
    "\n" + "=" * 120
)

print(
    "NON-NEO FIELDS THAT CHANGE ACROSS REPEATED SUBJECT ROWS"
)

print(
    "=" * 120
)


if len(difference_table) == 0:

    print(
        "No changing non-NEO columns found."
    )

else:

    print(
        difference_table.to_string(
            index=False
        )
    )


# ============================================================
# 10. DETAILED DUPLICATE-ROW SESSION AUDIT
#
# Show only session/context fields plus NEO completeness.
# ============================================================

context_columns = []


for c in meta.columns:

    lower = str(c).lower()

    if (
        any(
            token in lower

            for token in [
                "session",
                "sess",
                "visit",
                "wave",
                "measurement",
                "date",
                "age"
            ]
        )

        and

        c not in NEO_COLUMNS
    ):

        context_columns.append(
            c
        )


context_columns = list(
    dict.fromkeys(
        context_columns
    )
)


dup[
    "NEO_observed_items"
] = dup[
    NEO_COLUMNS
].notna().sum(
    axis=1
)


display_columns = [

    "_excel_row",
    "TDBRAIN_ID",
    "subject_norm",
    *context_columns,
    "NEO_observed_items"
]


display_columns = [

    c

    for c in display_columns

    if c in dup.columns
]


print(
    "\n" + "=" * 120
)

print(
    "DETAILED REPEATED-SUBJECT ROWS"
)

print(
    "=" * 120
)


print(

    dup[
        display_columns
    ]

    .sort_values(
        [
            "subject_norm",
            "_excel_row"
        ]
    )

    .to_string(
        index=False
    )
)


# ============================================================
# 11. AUTOMATIC BASELINE-ROW FEASIBILITY CHECK
#
# We DO NOT select a baseline row unless a candidate column
# produces exactly ONE session-1 row for EVERY source subject.
# ============================================================

valid_session_columns = []


for column in candidate_table[
    "column"
].tolist() if len(candidate_table) else []:

    temp = m86[
        [
            "subject_norm",
            column,
            "_excel_row"
        ]
    ].copy()


    temp[
        "_parsed_session"
    ] = [

        parse_session_number(
            value,
            subject_id=sid
        )

        for value, sid in zip(
            temp[
                column
            ],
            temp[
                "subject_norm"
            ]
        )
    ]


    baseline = temp[
        temp[
            "_parsed_session"
        ]
        ==
        1
    ]


    baseline_counts = (

        baseline[
            "subject_norm"
        ]
        .value_counts()
    )


    all_subjects_have_one = (

        len(
            baseline_counts
        )
        ==
        86

        and

        (
            baseline_counts
            ==
            1
        ).all()
    )


    if all_subjects_have_one:

        valid_session_columns.append(
            column
        )


print(
    "\n" + "=" * 120
)

print(
    "BASELINE SESSION RESOLUTION"
)

print(
    "=" * 120
)


print(
    "Columns yielding exactly one session-1 row "
    "for all 86 subjects:"
)

print(
    valid_session_columns
)


# ============================================================
# 12. SAVE AUDIT
# ============================================================

DUPLICATE_FILE = (
    AUDIT
    / "STAGE0B_R1_REPEATED_SUBJECT_ROWS.csv"
)

CANDIDATE_FILE = (
    AUDIT
    / "STAGE0B_R1_SESSION_COLUMN_CANDIDATES.csv"
)

DIFFERENCE_FILE = (
    AUDIT
    / "STAGE0B_R1_REPEATED_ROW_CONTEXT_DIFFERENCES.csv"
)

SUMMARY_FILE = (
    AUDIT
    / "STAGE0B_R1_BASELINE_SESSION_RESOLUTION_SUMMARY.json"
)


dup[
    display_columns
].to_csv(
    DUPLICATE_FILE,
    index=False
)


candidate_table.to_csv(
    CANDIDATE_FILE,
    index=False
)


difference_table.to_csv(
    DIFFERENCE_FILE,
    index=False
)


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE0B_R1_BASELINE_SESSION_RESOLUTION_AUDIT",

    "source_subjects":
        86,

    "metadata_rows":
        int(
            len(
                m86
            )
        ),

    "repeated_subjects":
        int(
            len(
                duplicates
            )
        ),

    "extra_rows":
        int(
            len(
                m86
            )
            -
            86
        ),

    "valid_session_columns_with_unique_baseline_for_all_subjects":
        valid_session_columns,

    "baseline_row_selected":
        False,

    "NEOFFI_values_averaged":
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
# 13. PRINT HASHES / STATUS
# ============================================================

print(
    "\nHASHES"
)

print(
    "Repeated rows:",
    sha256_file(
        DUPLICATE_FILE
    )
)

print(
    "Session candidates:",
    sha256_file(
        CANDIDATE_FILE
    )
)

print(
    "Context differences:",
    sha256_file(
        DIFFERENCE_FILE
    )
)

print(
    "Summary:",
    sha256_file(
        SUMMARY_FILE
    )
)


print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: STAGE 0B-R1 AUDIT COMPLETE"
)

print(
    "=" * 120
)


if len(
    valid_session_columns
) == 1:

    print(
        "\nUNIQUE SESSION FIELD FOUND:"
    )

    print(
        valid_session_columns[
            0
        ]
    )

    print(
        "\nA unique baseline/session-1 row exists "
        "for all 86 subjects."
    )

    print(
        "NEXT: freeze that baseline-row rule."
    )


elif len(
    valid_session_columns
) > 1:

    print(
        "\nMULTIPLE POSSIBLE SESSION FIELDS FOUND."
    )

    print(
        "Do not choose one yet."
    )


else:

    print(
        "\nNO AUTOMATICALLY VERIFIED BASELINE FIELD FOUND."
    )

    print(
        "Do not average or select the first row."
    )


print(
    "\nNO NEO-FFI VALUES WERE AVERAGED."
)

print(
    "NO PHENOTYPE MATRIX WAS CREATED."
)

print(
    "NO FUSED GRAPH OR CLUSTERING WAS RUN."
)