#!/usr/bin/env python3
"""
Reproduce held-out age, EHQ, and sex associations for K=4, 5, and 6
from saved subtype labels.

The script does not rerun EEG preprocessing, representation learning,
graph construction, VGAE training, UMAP, or spectral clustering.
It verifies saved K=4/5/6 partitions, aligns them with subject-level
metadata, reproduces the continuous age/EHQ analyses, and evaluates
the categorical sex-subtype association using Pearson's chi-square
test and Cramer's V.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EXPECTED = {
    4: {"n": 430, "min": 48, "balance": 3.50},
    5: {"n": 430, "min": 48, "balance": 3.52},
    6: {"n": 430, "min": 48, "balance": 2.52},
}

BAD_PATH_TOKENS = (
    "baseline", "ablation", "hyperparam", "seed_robust", "random_seed",
    "comment6", "phenotype_similarity", "fresh_cosine", "benchmark",
    "eeg_only", "phenotype_only", "deepcluster", "kmeans", "gmm",
)
GOOD_PATH_TOKENS = (
    "k_selection", "k-selection", "multicriteria", "multi_criteria",
    "number_of_clusters", "number_of_subtypes", "k2_10", "k2-10",
    "elbow", "final_forced_k5", "graphk12_alpha0p6", "umap30",
)

ID_ALIASES = {
    "subject", "subjectid", "subject_id", "participant", "participantid",
    "participant_id", "subjectkey", "src_subject_id", "id", "subid",
}
VAR_ALIASES = {
    "age": {"age", "ageyears", "age_years", "age_yrs", "interview_age", "interviewage"},
    "ehq": {"ehq", "ehqtotal", "ehq_total", "ehqscore", "ehq_score", "handedness", "handednessscore", "handedness_score"},
    "p_factor": {"pfactor", "p_factor", "p-factor", "generalpsychopathology", "general_psychopathology"},
    "attention": {"attention", "attn", "attentionfactor", "attention_factor"},
    "internalizing": {"internalizing", "internalising", "internalizingfactor", "internalizing_factor"},
    "externalizing": {"externalizing", "externalising", "externalizingfactor", "externalizing_factor"},
    "sex": {"sex", "biological_sex"},
}
LABEL_TOKENS = ("label", "labels", "cluster", "clusters", "subtype", "assignment")


def norm_col(x: object) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def norm_id(x: object) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"^sub[-_]", "", s)
    return s


def is_numeric_label_vector(a: np.ndarray) -> bool:
    if a.ndim != 1 or len(a) != 430:
        return False
    try:
        b = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy()
    except Exception:
        return False
    if np.isnan(b).any():
        return False
    return np.allclose(b, np.round(b))


def label_signature(labels: np.ndarray) -> Tuple[int, int, float, Tuple[int, ...]]:
    vals, counts = np.unique(labels.astype(int), return_counts=True)
    k = len(vals)
    mn, mx = int(counts.min()), int(counts.max())
    bal = mx / mn
    return k, mn, bal, tuple(sorted(map(int, counts)))


def signature_matches(labels: np.ndarray, k: int) -> bool:
    kk, mn, bal, _ = label_signature(labels)
    exp = EXPECTED[k]
    return kk == k and len(labels) == exp["n"] and mn == exp["min"] and round(bal, 2) == exp["balance"]


def score_path(path: Path) -> int:
    s = str(path).lower()
    score = 0
    for t in GOOD_PATH_TOKENS:
        if t in s:
            score += 15
    for t in BAD_PATH_TOKENS:
        if t in s:
            score -= 25
    if "result" in s:
        score += 2
    return score


def detect_id_column(columns: Sequence[object]) -> Optional[str]:
    normed = {norm_col(c): str(c) for c in columns}
    for alias in ID_ALIASES:
        if alias in normed:
            return normed[alias]
    for c in columns:
        nc = norm_col(c)
        if "subject" in nc or "participant" in nc:
            return str(c)
    return None


def find_sidecar_ids(directory: Path, n: int = 430) -> Optional[np.ndarray]:
    candidates: List[Tuple[int, Path, np.ndarray]] = []
    search_dirs = [directory, directory.parent, directory.parent.parent]
    seen: set[Path] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            name = p.name.lower()
            if not any(t in name for t in ("subject", "participant", "ids", "order")):
                continue
            try:
                arr: Optional[np.ndarray] = None
                if p.suffix.lower() == ".npy":
                    x = np.load(p, allow_pickle=True)
                    if x.ndim == 1 and len(x) == n:
                        arr = x.astype(str)
                elif p.suffix.lower() == ".csv":
                    df = pd.read_csv(p)
                    id_col = detect_id_column(df.columns)
                    if id_col and len(df) == n:
                        arr = df[id_col].astype(str).to_numpy()
                    elif df.shape == (n, 1):
                        arr = df.iloc[:, 0].astype(str).to_numpy()
                elif p.suffix.lower() in (".txt", ".tsv"):
                    sep = "\t" if p.suffix.lower() == ".tsv" else None
                    df = pd.read_csv(p, sep=sep, header=None, engine="python")
                    if df.shape[0] == n:
                        arr = df.iloc[:, 0].astype(str).to_numpy()
                if arr is not None and len(set(map(norm_id, arr))) == n:
                    sc = score_path(p) + (10 if p.parent == directory else 0)
                    candidates.append((sc, p, arr))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return candidates[0][2]


@dataclass
class LabelCandidate:
    k: int
    path: Path
    source: str
    labels: np.ndarray
    ids: Optional[np.ndarray]
    score: int
    signature: Tuple[int, int, float, Tuple[int, ...]]


def scan_label_candidates(root: Path) -> Dict[int, List[LabelCandidate]]:
    out: Dict[int, List[LabelCandidate]] = {4: [], 5: [], 6: []}
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Avoid git internals and huge raw/preprocessed arrays.
        dirnames[:] = [x for x in dirnames if x not in {".git", "raw", "preprocessed", "__pycache__", ".ipynb_checkpoints"}]
        for fn in filenames:
            p = d / fn
            suf = p.suffix.lower()
            if suf not in {".csv", ".npy", ".npz"}:
                continue
            try:
                if p.stat().st_size > 200_000_000:
                    continue
            except OSError:
                continue
            base_score = score_path(p)
            try:
                if suf == ".csv":
                    df = pd.read_csv(p)
                    if len(df) != 430:
                        continue
                    id_col = detect_id_column(df.columns)
                    ids = df[id_col].astype(str).to_numpy() if id_col else None
                    for col in df.columns:
                        nc = norm_col(col)
                        if not any(tok in nc for tok in LABEL_TOKENS):
                            continue
                        a = pd.to_numeric(df[col], errors="coerce").to_numpy()
                        if np.isnan(a).any() or not is_numeric_label_vector(a):
                            continue
                        a = np.round(a).astype(int)
                        k = len(np.unique(a))
                        if k in out and signature_matches(a, k):
                            bonus = 10 if re.search(fr"(^|[^0-9])k[_-]?{k}([^0-9]|$)", nc) else 0
                            out[k].append(LabelCandidate(k, p, f"csv:{col}", a, ids, base_score + bonus, label_signature(a)))
                elif suf == ".npy":
                    a = np.load(p, allow_pickle=True)
                    if is_numeric_label_vector(np.asarray(a)):
                        a = np.round(np.asarray(a)).astype(int)
                        k = len(np.unique(a))
                        if k in out and signature_matches(a, k):
                            ids = find_sidecar_ids(p.parent)
                            bonus = 10 if re.search(fr"(^|[^0-9])k[_-]?{k}([^0-9]|$)", p.stem.lower()) else 0
                            out[k].append(LabelCandidate(k, p, "npy", a, ids, base_score + bonus, label_signature(a)))
                else:  # npz
                    z = np.load(p, allow_pickle=True)
                    id_arr = None
                    for key in z.files:
                        if any(t in norm_col(key) for t in ("subject", "participant", "ids")):
                            x = np.asarray(z[key])
                            if x.ndim == 1 and len(x) == 430:
                                id_arr = x.astype(str)
                    for key in z.files:
                        a = np.asarray(z[key])
                        if is_numeric_label_vector(a):
                            a = np.round(a).astype(int)
                            k = len(np.unique(a))
                            if k in out and signature_matches(a, k):
                                bonus = 10 if re.search(fr"(^|[^0-9])k[_-]?{k}([^0-9]|$)", norm_col(key)) else 0
                                out[k].append(LabelCandidate(k, p, f"npz:{key}", a, id_arr, base_score + bonus, label_signature(a)))
            except Exception:
                continue
    return out


def arrays_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    # Exact labels may differ only by permutation. Compare pairwise co-membership.
    if len(a) != len(b):
        return False
    return np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])


def choose_candidate(cands: List[LabelCandidate], k: int) -> LabelCandidate:
    if not cands:
        raise RuntimeError(
            f"No saved K={k} label vector matched Table 9 signature: "
            f"N=430, min={EXPECTED[k]['min']}, balance={EXPECTED[k]['balance']:.2f}."
        )
    cands = sorted(cands, key=lambda c: (c.score, str(c.path), c.source), reverse=True)
    top_score = cands[0].score
    top = [c for c in cands if c.score == top_score]
    # Duplicate copies of the same partition are safe.
    if all(arrays_equivalent(top[0].labels, c.labels) for c in top[1:]):
        return top[0]
    msg = [f"Ambiguous saved K={k} label candidates with equal top score. Refusing to guess:"]
    for c in top:
        msg.append(f"  score={c.score} {c.path} [{c.source}] sizes={c.signature[3]}")
    raise RuntimeError("\n".join(msg))


def detect_variable_columns(columns: Sequence[object]) -> Dict[str, str]:
    norm_to_orig = {norm_col(c): str(c) for c in columns}
    found: Dict[str, str] = {}
    for var, aliases in VAR_ALIASES.items():
        for alias in aliases:
            na = norm_col(alias)
            if na in norm_to_orig:
                found[var] = norm_to_orig[na]
                break
        if var in found:
            continue
        # conservative contains matching for long aliases only
        for nc, orig in norm_to_orig.items():
            if var == "age" and (nc == "age" or nc.endswith("_age") or nc.startswith("age_")):
                found[var] = orig; break
            if var == "ehq" and ("ehq" in nc or "handedness" in nc):
                found[var] = orig; break
            if var == "p_factor" and ("pfactor" in nc or "p_factor" in nc):
                found[var] = orig; break
            if var in {"attention", "internalizing", "externalizing"} and var in nc:
                found[var] = orig; break
    return found


@dataclass
class ClinicalCandidate:
    path: Path
    df: pd.DataFrame
    id_col: str
    vars: Dict[str, str]
    score: int


def scan_clinical_candidates(root: Path) -> List[ClinicalCandidate]:
    out: List[ClinicalCandidate] = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = [x for x in dirnames if x not in {".git", "raw", "preprocessed", "__pycache__", ".ipynb_checkpoints"}]
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            p = d / fn
            try:
                if p.stat().st_size > 100_000_000:
                    continue
                df = pd.read_csv(p)
            except Exception:
                continue
            if not (400 <= len(df) <= 5000):
                continue
            id_col = detect_id_column(df.columns)
            if not id_col:
                continue
            vars_found = detect_variable_columns(df.columns)
            if "age" not in vars_found or "ehq" not in vars_found:
                continue
            score = score_path(p)
            score += 10 * len(vars_found)
            s = str(p).lower()
            if "phenotype" in s or "clinical" in s:
                score += 10
            if len(df) == 430:
                score += 10
            out.append(ClinicalCandidate(p, df, id_col, vars_found, score))
    return sorted(out, key=lambda c: (c.score, str(c.path)), reverse=True)


def choose_clinical(cands: List[ClinicalCandidate]) -> ClinicalCandidate:
    if not cands:
        raise RuntimeError("No subject-level CSV containing subject ID, Age, and EHQ was found under the project root.")
    top_score = cands[0].score
    top = [c for c in cands if c.score == top_score]
    if len(top) == 1:
        return top[0]
    # Accept duplicate paths only if their normalized subject IDs and age/EHQ values are identical.
    ref = top[0]
    ref_df = ref.df[[ref.id_col, ref.vars["age"], ref.vars["ehq"]]].copy()
    ref_df[ref.id_col] = ref_df[ref.id_col].map(norm_id)
    ref_df = ref_df.sort_values(ref.id_col).reset_index(drop=True)
    for c in top[1:]:
        x = c.df[[c.id_col, c.vars["age"], c.vars["ehq"]]].copy()
        x[c.id_col] = x[c.id_col].map(norm_id)
        x = x.sort_values(c.id_col).reset_index(drop=True)
        if x.shape != ref_df.shape or not np.array_equal(x.iloc[:, 0].to_numpy(), ref_df.iloc[:, 0].to_numpy()):
            break
        if not np.allclose(pd.to_numeric(x.iloc[:, 1], errors="coerce"), pd.to_numeric(ref_df.iloc[:, 1], errors="coerce"), equal_nan=True):
            break
        if not np.allclose(pd.to_numeric(x.iloc[:, 2], errors="coerce"), pd.to_numeric(ref_df.iloc[:, 2], errors="coerce"), equal_nan=True):
            break
    else:
        return ref
    msg = ["Ambiguous subject-level Age/EHQ CSV candidates. Refusing to guess:"]
    for c in top:
        msg.append(f"  score={c.score} {c.path} columns={c.vars}")
    raise RuntimeError("\n".join(msg))




def detect_numeric_label_column(df: pd.DataFrame, exclude: Sequence[str] = ()) -> Optional[str]:
    """Find a 430-row integer clustering column such as subtype/cluster/label."""
    excluded = {str(x) for x in exclude}
    ranked: List[Tuple[int, str]] = []
    for col in df.columns:
        if str(col) in excluded:
            continue
        nc = norm_col(col)
        if not any(tok in nc for tok in LABEL_TOKENS):
            continue
        a = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if len(a) != 430 or np.isnan(a).any() or not np.allclose(a, np.round(a)):
            continue
        k = len(np.unique(np.round(a).astype(int)))
        if not (2 <= k <= 10):
            continue
        score = 0
        if nc == "subtype": score += 30
        if nc == "cluster": score += 20
        if nc == "label": score += 15
        if "final" in nc: score += 5
        ranked.append((score, str(col)))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def is_sequential_row_index(ids: Sequence[object], n: int = 430) -> bool:
    """True when an apparent subject-ID column is only 0..N-1 row numbering."""
    vals = np.asarray([norm_id(x) for x in ids], dtype=object)
    expected = np.asarray([str(i) for i in range(n)], dtype=object)
    return len(vals) == n and np.array_equal(vals, expected)

def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def one_way_effects(values: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(values)
    y = values[mask].astype(float)
    g = labels[mask].astype(int)
    groups = [y[g == lab] for lab in sorted(np.unique(g))]
    groups = [x for x in groups if len(x) > 0]
    n = len(y)
    k = len(groups)
    if n <= k or k < 2:
        raise ValueError("Insufficient observations/groups for one-way analysis.")
    grand = float(np.mean(y))
    ss_between = float(sum(len(x) * (float(np.mean(x)) - grand) ** 2 for x in groups))
    ss_within = float(sum(np.sum((x - np.mean(x)) ** 2) for x in groups))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total > 0 else float("nan")
    f_stat, anova_p = stats.f_oneway(*groups)
    h_stat, kruskal_p = stats.kruskal(*groups)
    return {
        "n": n,
        "groups": k,
        "anova_f": float(f_stat),
        "anova_p_raw": float(anova_p),
        "kruskal_h": float(h_stat),
        "kruskal_p_raw": float(kruskal_p),
        "eta_squared": float(eta2),
        "omega_squared": float(omega2),
    }


def format_p(p: float) -> str:
    if p < 0.0001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def categorical_sex_effects(values: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Pearson chi-square association and Cramer's V for sex versus subtype."""
    df = pd.DataFrame({
        "sex": values,
        "subtype": labels.astype(int),
    })

    df["sex"] = df["sex"].astype(str).str.strip()
    df = df[
        df["sex"].notna()
        & (~df["sex"].str.lower().isin(["", "nan", "none", "na"]))
    ]

    table = pd.crosstab(df["subtype"], df["sex"])

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Sex-subtype contingency table must have at least 2 rows and 2 columns.")

    chi2, p_raw, dof, expected = stats.chi2_contingency(
        table.to_numpy(),
        correction=False,
    )

    n = int(table.to_numpy().sum())
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = math.sqrt(float(chi2) / denom) if denom > 0 else float("nan")

    return {
        "n": n,
        "chi_square": float(chi2),
        "df": int(dof),
        "p_raw": float(p_raw),
        "cramers_v": float(cramers_v),
        "min_expected_count": float(np.min(expected)),
        "n_expected_lt5": int(np.sum(expected < 5)),
    }


def ensure_ids(candidate: LabelCandidate, fallback_ids: Optional[np.ndarray]) -> np.ndarray:
    ids = candidate.ids if candidate.ids is not None else fallback_ids
    if ids is None:
        raise RuntimeError(
            f"Saved labels at {candidate.path} have no subject IDs, and no canonical subject-order file was found nearby."
        )
    ids = np.asarray(ids).astype(str)
    if len(ids) != 430:
        raise RuntimeError(f"Subject ID vector for {candidate.path} has {len(ids)} rows; expected 430.")
    normed = np.array([norm_id(x) for x in ids], dtype=object)
    if len(set(normed.tolist())) != 430:
        raise RuntimeError(f"Subject IDs for {candidate.path} are not 430 unique values.")
    return normed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise SystemExit(f"Project root does not exist: {root}")

    print("=" * 88)
    print("K=4/5/6 HELD-OUT ROBUSTNESS FROM SAVED LABELS ONLY")
    print("=" * 88)
    print(f"Project root: {root}")
    print(f"Output dir : {outdir}")
    print("No preprocessing, CNN-AE, PCA, graph, VGAE, UMAP, or clustering will be rerun.\n")

    label_cands = scan_label_candidates(root)
    selected: Dict[int, LabelCandidate] = {}
    for k in (4, 5, 6):
        print(f"Saved K={k} candidates matching Table 9 signature:")
        for c in sorted(label_cands[k], key=lambda x: (x.score, str(x.path)), reverse=True)[:20]:
            print(f"  score={c.score:4d} sizes={c.signature[3]}  {c.path} [{c.source}]")
        selected[k] = choose_candidate(label_cands[k], k)
        print(f"SELECTED K={k}: {selected[k].path} [{selected[k].source}]\n")

    clinical_cands = scan_clinical_candidates(root)
    print("Subject-level clinical candidates containing Age and EHQ:")
    for c in clinical_cands[:20]:
        print(f"  score={c.score:4d} rows={len(c.df):4d} vars={sorted(c.vars)}  {c.path}")
    clinical = choose_clinical(clinical_cands)
    print(f"SELECTED CLINICAL FILE: {clinical.path}")
    print(f"SUBJECT ID COLUMN     : {clinical.id_col}")
    print(f"VARIABLE COLUMNS      : {clinical.vars}\n")

    cdf = clinical.df.copy()
    cdf["__id_norm__"] = cdf[clinical.id_col].map(norm_id)
    # If multiple rows per subject exist, only accept them when values are identical; otherwise stop.
    if cdf["__id_norm__"].duplicated().any():
        cols_check = [clinical.vars[v] for v in clinical.vars]
        bad = []
        for sid, g in cdf.groupby("__id_norm__"):
            for col in cols_check:
                vals = pd.to_numeric(g[col], errors="coerce").dropna().unique()
                if len(vals) > 1:
                    bad.append((sid, col, vals[:5]))
        if bad:
            raise RuntimeError(f"Clinical file has conflicting duplicate subject rows, e.g. {bad[:5]}")
        cdf = cdf.drop_duplicates("__id_norm__", keep="first")
    cdf = cdf.set_index("__id_norm__", drop=False)

    # Establish a common subject order. Some labels_K*.csv files store 0..429
    # in a column named "subject"; those are row indices, not HBN subject IDs.
    # In that case, row-order alignment is accepted only after the saved K=5
    # partition is verified against the subtype column in the selected clinical file.
    aligned_labels: Dict[int, np.ndarray] = {}
    aligned_ids_by_k: Dict[int, np.ndarray] = {}
    alignment_mode = "subject_id"

    embedded_ids = [selected[k].ids for k in (4, 5, 6)]
    pseudo_index_mode = all(x is not None and is_sequential_row_index(x) for x in embedded_ids)

    if pseudo_index_mode:
        alignment_mode = "verified_row_order"
        if len(cdf) != 430:
            raise RuntimeError(
                f"Row-order alignment requires exactly 430 clinical rows; found {len(cdf)}."
            )
        clinical_label_col = detect_numeric_label_column(
            clinical.df,
            exclude=[clinical.id_col, *clinical.vars.values()],
        )
        if clinical_label_col is None:
            raise RuntimeError(
                "The saved K-selection label files use row indices 0..429, but the selected "
                "clinical file has no numeric subtype/cluster column for verifying row order."
            )
        clinical_k5 = pd.to_numeric(clinical.df[clinical_label_col], errors="coerce").to_numpy()
        if np.isnan(clinical_k5).any():
            raise RuntimeError(f"Clinical verification column {clinical_label_col!r} contains missing labels.")
        clinical_k5 = np.round(clinical_k5).astype(int)
        if len(np.unique(clinical_k5)) != 5:
            raise RuntimeError(
                f"Clinical verification column {clinical_label_col!r} does not contain a K=5 partition."
            )
        if not arrays_equivalent(selected[5].labels.astype(int), clinical_k5):
            raise RuntimeError(
                "Row-order verification failed: saved labels_K5.csv is not partition-equivalent "
                f"to clinical column {clinical_label_col!r}. Refusing positional alignment."
            )

        actual_ids = clinical.df[clinical.id_col].map(norm_id).to_numpy(dtype=object)
        if len(set(actual_ids.tolist())) != 430:
            raise RuntimeError("Clinical subject IDs are not 430 unique values.")
        for k in (4, 5, 6):
            if not is_sequential_row_index(selected[k].ids):
                raise RuntimeError(f"K={k} does not share the verified 0..429 row-index convention.")
            aligned_ids_by_k[k] = actual_ids.copy()
            aligned_labels[k] = selected[k].labels.astype(int)
        print(
            "ROW-ORDER ALIGNMENT VERIFIED: labels_K5 partition matches clinical "
            f"column {clinical_label_col!r}; K=4/5/6 labels use the same 0..429 row order.\n"
        )
    else:
        fallback_ids = None
        for k in (4, 5, 6):
            if selected[k].ids is not None and not is_sequential_row_index(selected[k].ids):
                fallback_ids = selected[k].ids
                break
        if fallback_ids is None:
            fallback_ids = find_sidecar_ids(selected[5].path.parent)

        for k in (4, 5, 6):
            ids = ensure_ids(selected[k], fallback_ids)
            if is_sequential_row_index(ids):
                raise RuntimeError(
                    f"K={k} provides only row indices 0..429, but row-order verification was not available."
                )
            missing = [sid for sid in ids if sid not in cdf.index]
            if missing:
                raise RuntimeError(
                    f"Clinical file is missing {len(missing)} of the 430 label subjects for K={k}. "
                    f"Examples: {missing[:10]}"
                )
            aligned_ids_by_k[k] = ids
            aligned_labels[k] = selected[k].labels.astype(int)

        ref_set = set(aligned_ids_by_k[5].tolist())
        for k in (4, 6):
            if set(aligned_ids_by_k[k].tolist()) != ref_set:
                raise RuntimeError(f"K={k} and K=5 label files do not contain the same 430 subjects.")

    all_rows: List[Dict[str, object]] = []
    variable_order = ["p_factor", "attention", "internalizing", "externalizing", "age", "ehq"]
    available_order = [v for v in variable_order if v in clinical.vars]
    if "age" not in available_order or "ehq" not in available_order:
        raise RuntimeError("Age and EHQ columns are required.")
    if "sex" not in clinical.vars:
        raise RuntimeError("A sex column is required for the categorical held-out analysis.")
    if len(available_order) < 6:
        missing_vars = [v for v in variable_order if v not in available_order]
        raise RuntimeError(
            "The clinical file does not contain all six variables required to reproduce "
            "the manuscript's six-variable BH-FDR family. Missing: " + ", ".join(missing_vars)
        )

    for k in (4, 5, 6):
        ids = aligned_ids_by_k[k]
        labels = aligned_labels[k]
        # Reorder clinical data to label order.
        ordered = cdf.loc[ids]
        k_rows = []
        for var in available_order:
            col = clinical.vars[var]
            values = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=float)
            res = one_way_effects(values, labels)
            row: Dict[str, object] = {"K": k, "variable": var, "source_column": col, **res}
            k_rows.append(row)
        anova_adj = bh_fdr([float(r["anova_p_raw"]) for r in k_rows])
        kruskal_adj = bh_fdr([float(r["kruskal_p_raw"]) for r in k_rows])
        for r, apv, kpv in zip(k_rows, anova_adj, kruskal_adj):
            r["anova_fdr_p"] = float(apv)
            r["kruskal_fdr_p"] = float(kpv)
            r["fdr_family_size"] = len(k_rows)
            all_rows.append(r)

    # --------------------------------------------------------
    # Held-out categorical sex analysis.
    # Sex is analysed separately from the six continuous
    # variables above. BH-FDR correction is applied across
    # the three neighboring K=4/5/6 sex tests.
    # --------------------------------------------------------
    sex_rows: List[Dict[str, object]] = []

    for k in (4, 5, 6):
        ids = aligned_ids_by_k[k]
        labels = aligned_labels[k]
        ordered = cdf.loc[ids]

        sex_col = clinical.vars["sex"]
        sex_values = ordered[sex_col].to_numpy()

        res = categorical_sex_effects(sex_values, labels)

        sex_rows.append({
            "K": k,
            "source_column": sex_col,
            **res,
        })

    sex_adj = bh_fdr([float(r["p_raw"]) for r in sex_rows])

    for r, adj in zip(sex_rows, sex_adj):
        r["fdr_p"] = float(adj)
        r["fdr_family_size"] = 3

    sex_df = pd.DataFrame(sex_rows)

    all_df = pd.DataFrame(all_rows)
    all_path = outdir / "k456_all_variable_associations.csv"
    all_df.to_csv(all_path, index=False)

    held = all_df[all_df["variable"].isin(["age", "ehq"])].copy()
    held_path = outdir / "k456_heldout_associations.csv"
    held.to_csv(held_path, index=False)

    sex_path = outdir / "k456_sex_association_summary.csv"
    sex_df.to_csv(sex_path, index=False)

    wide_rows = []
    for k in (4, 5, 6):
        row: Dict[str, object] = {"K": k}
        for var in ("age", "ehq"):
            r = held[(held["K"] == k) & (held["variable"] == var)].iloc[0]
            for field in ("n", "eta_squared", "omega_squared", "anova_p_raw", "anova_fdr_p", "kruskal_p_raw", "kruskal_fdr_p"):
                row[f"{var}_{field}"] = r[field]

        sr = sex_df[sex_df["K"] == k].iloc[0]
        row["sex_chi_square"] = sr["chi_square"]
        row["sex_df"] = sr["df"]
        row["sex_p_raw"] = sr["p_raw"]
        row["sex_fdr_p"] = sr["fdr_p"]
        row["sex_cramers_v"] = sr["cramers_v"]
        row["sex_min_expected_count"] = sr["min_expected_count"]
        row["sex_expected_lt5"] = sr["n_expected_lt5"]

        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide_path = outdir / "k456_heldout_robustness_wide.csv"
    wide.to_csv(wide_path, index=False)

    expected_internal = {
        4: (0.417, 0.806, 0.748, 48, 3.50),
        5: (0.479, 0.659, 0.756, 48, 3.52),
        6: (0.454, 0.753, 0.677, 48, 2.52),
    }
    latex_lines = []
    for k in (4, 5, 6):
        r = wide[wide["K"] == k].iloc[0]
        sil, db, ari, mn, bal = expected_internal[k]
        latex_lines.append(
            f"{k} & {sil:.3f} & {db:.3f} & {ari:.3f} & {mn:d} & {bal:.2f} & "
            f"{float(r['age_eta_squared']):.4f} & {format_p(float(r['age_anova_fdr_p']))} & "
            f"{float(r['ehq_eta_squared']):.4f} & {format_p(float(r['ehq_anova_fdr_p']))} \\\\"
        )
    latex_path = outdir / "k456_table_rows_existing_10_column_format.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    latex_rich = []
    for k in (4, 5, 6):
        r = wide[wide["K"] == k].iloc[0]
        latex_rich.append(
            f"{k} & {float(r['age_eta_squared']):.4f} & {float(r['age_omega_squared']):.4f} & "
            f"{format_p(float(r['age_anova_fdr_p']))} & {float(r['ehq_eta_squared']):.4f} & "
            f"{float(r['ehq_omega_squared']):.4f} & {format_p(float(r['ehq_anova_fdr_p']))} \\\\"
        )
    rich_path = outdir / "k456_heldout_effect_rows_eta_omega.tex"
    rich_path.write_text("\n".join(latex_rich) + "\n", encoding="utf-8")

    # Absolute local input paths and subject identifiers are intentionally
    # not written to public outputs.

    print("\n" + "=" * 88)
    print("FINAL HELD-OUT ROBUSTNESS RESULTS")
    print("=" * 88)
    print(wide.to_string(index=False))
    print("\nLaTeX rows for the held-out robustness table:")
    print(latex_path.read_text(encoding="utf-8"))
    print("Saved files:")
    print("\nSex association summary:")
    print(sex_df.to_string(index=False))
    print("\nSaved files:")
    for p in (all_path, held_path, sex_path, wide_path, latex_path, rich_path):
        print(f"  {p}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR: " + str(exc), file=sys.stderr)
        print("The script stopped rather than guessing a file or subject order.", file=sys.stderr)
        raise
