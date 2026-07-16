#!/usr/bin/env python3
"""
Final-pipeline sensitivity analysis for the EEG–phenotype fusion weight.

This script holds the remaining pipeline settings fixed:

- graph neighborhood K = 12
- UMAP neighbors = 30
- subtype count K = 5
- VGAE hidden dimension = 64
- VGAE latent dimension = 16
- VGAE training epochs = 500
- learning rate = 1e-3
- random seed = 42

The tested fusion weights are:

    0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.0

The implementation reuses the final-pipeline functions defined in
12_alpha_joint_selection.py so that the sensitivity experiment and
the selected EPG-SD model use the same graph, VGAE, UMAP, and
spectral-clustering implementation.

Environment variables:

EPGSD_ANALYSIS_ROOT
    Root containing graphs_three_experiments.

EPGSD_SENSITIVITY_OUT
    Output directory for generated results.

Warning:
The per-configuration output folders can include embeddings,
coordinates, and subject-level assignments. Do not commit those
private generated folders to the public repository. Only the
participant-free aggregate summary should be published.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SEED = 42
GRAPH_K = 12
UMAP_NEIGHBORS = 30
FORCED_K = 5

ALPHAS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.6,
    0.9,
    1.0,
]

OUTPUT_ROOT = Path(
    os.environ.get(
        "EPGSD_SENSITIVITY_OUT",
        "results/sensitivity_ablation/table14_final_pipeline",
    )
)


def load_final_pipeline_module():
    source = Path(__file__).with_name(
        "12_alpha_joint_selection.py"
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Required final-pipeline script not found: {source}"
        )

    spec = importlib.util.spec_from_file_location(
        "epgsd_alpha_joint_selection",
        source,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to import final-pipeline script: {source}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def create_public_row(module, row, alpha):
    eta_values = []
    fdr_count = 0

    for variable in module.CLINICAL_VARS:
        eta_key = f"{variable}_eta_squared"
        p_key = f"{variable}_anova_p_fdr"

        eta_value = row.get(eta_key)
        p_value = row.get(p_key)

        if eta_value is not None:
            eta_value = float(eta_value)

            if np.isfinite(eta_value):
                eta_values.append(eta_value)

        if p_value is not None:
            p_value = float(p_value)

            if np.isfinite(p_value) and p_value < 0.05:
                fdr_count += 1

    return {
        "graph_k": GRAPH_K,
        "alpha": alpha,
        "umap_n_neighbors": UMAP_NEIGHBORS,
        "forced_k": FORCED_K,
        "silhouette": row["silhouette"],
        "davies_bouldin": row["davies_bouldin"],
        "calinski_harabasz": row["calinski_harabasz"],
        "dunn_index": row["dunn_index"],
        "min_cluster_size": row["min_cluster_size"],
        "max_cluster_size": row["max_cluster_size"],
        "cluster_size_ratio_max_min":
            row["cluster_size_ratio_max_min"],
        "mean_eta_squared": float(np.mean(eta_values)),
        "fdr_sig_count": f"{fdr_count}/6",
        "age_anova_p_fdr": row.get("age_anova_p_fdr"),
        "age_eta_squared": row.get("age_eta_squared"),
        "ehq_total_anova_p_fdr":
            row.get("ehq_total_anova_p_fdr"),
        "ehq_total_eta_squared":
            row.get("ehq_total_eta_squared"),
    }


def main():
    module = load_final_pipeline_module()

    module.OUTPUT_ROOT = OUTPUT_ROOT / "private_runs"
    module.UMAP_N_GRID = [UMAP_NEIGHBORS]
    module.VGAE_EPOCHS = 500
    module.HIDDEN_DIM = 64
    module.LATENT_DIM = 16
    module.DROPOUT = 0.0
    module.LR = 0.001
    module.SEED = SEED

    # Table 14 does not report stability or permutation metrics.
    module.N_SUBSAMPLE = 1
    module.N_BOOTSTRAP = 1
    module.N_PERMUTATIONS = 1

    module.set_seed(SEED)

    required_files = [
        Path(module.EEG_FILE),
        Path(module.PHENO_FILE),
        Path(module.META_FILE),
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    eeg = np.load(module.EEG_FILE).astype(np.float32)
    phenotype = np.load(
        module.PHENO_FILE
    ).astype(np.float32)
    metadata = pd.read_csv(module.META_FILE)

    if not (
        eeg.shape[0]
        == phenotype.shape[0]
        == len(metadata)
    ):
        raise ValueError(
            "EEG, phenotype, and metadata rows are not aligned."
        )

    eeg = (
        StandardScaler()
        .fit_transform(eeg)
        .astype(np.float32)
    )

    phenotype = (
        StandardScaler()
        .fit_transform(phenotype)
        .astype(np.float32)
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    public_rows = []

    for index, alpha in enumerate(ALPHAS, start=1):
        print()
        print("=" * 80)
        print(
            f"Alpha {index}/{len(ALPHAS)}: "
            f"{alpha:.1f}"
        )
        print("=" * 80)

        rows = module.run_one_setting(
            X_eeg=eeg,
            X_pheno=phenotype,
            meta=metadata,
            graph_k=GRAPH_K,
            alpha=alpha,
        )

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one row for alpha={alpha}; "
                f"found {len(rows)}"
            )

        public_rows.append(
            create_public_row(
                module,
                dict(rows[0]),
                alpha,
            )
        )

        pd.DataFrame(public_rows).to_csv(
            OUTPUT_ROOT
            / "T14_final_pipeline_alpha_sensitivity_RUNNING.csv",
            index=False,
            lineterminator="\n",
        )

    summary = pd.DataFrame(public_rows)

    final_path = (
        OUTPUT_ROOT
        / "T14_final_pipeline_alpha_sensitivity.csv"
    )

    summary.to_csv(
        final_path,
        index=False,
        lineterminator="\n",
    )

    print()
    print(summary.to_string(index=False))
    print()
    print(
        f"PASS: completed rows = {len(summary)}/{len(ALPHAS)}"
    )
    print("Saved:", final_path)


if __name__ == "__main__":
    main()
