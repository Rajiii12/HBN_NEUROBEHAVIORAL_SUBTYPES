# ============================================================
# TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
# EEG + NEO-FFI PHENOTYPE FUSION
#
# STAGE 2A — FROZEN VGAE TRAINING
#
# INPUT
# -----
# N = 85
# node features = 188
#   EEG128 + baseline NEO-FFI60
#
# fused graph:
#   k = 4
#   215 undirected edges
#   1 connected component
#   0 isolates
#
# VGAE
# ----
# 188 -> 64 -> 16
# ReLU
# dropout = 0
# separate mu / logvar heads
# inner-product decoder
# weighted BCE + KL
# Adam lr=1e-3
# 500 fixed epochs
# seed=42
#
# DOWNSTREAM REPRESENTATION
# -------------------------
# posterior mean mu
#
# THIS STAGE DOES NOT:
#   run UMAP
#   cluster subjects
#   select K
#   use diagnosis
#   use ADHD/YBOCS/BDI
# ============================================================

from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# 1. ROOTS
# ============================================================

DRIVE = Path(
    "/content/drive/MyDrive"
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

GRAPH_DIR = (
    EXP3
    / "03_GRAPH"
)

VGAE_DIR = (
    EXP3
    / "04_VGAE"
)


VGAE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 120)
print("TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3")
print("STAGE 2A — FROZEN VGAE TRAINING")
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


def set_seed(seed=42):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# 3. FIXED COMPUTE PLATFORM
#
# CPU is entirely adequate for N=85.
# ============================================================

DEVICE = torch.device(
    "cpu"
)


print(
    "PyTorch:",
    torch.__version__
)

print(
    "Compute device:",
    DEVICE
)


# ============================================================
# 4. FROZEN STAGE1A INPUTS
# ============================================================

STAGE1A_LOCK = (
    LOCK
    / "STAGE1A_FUSED_GRAPH_LOCK.json"
)

SUBJECT_ORDER = (
    LOCK
    / "STAGE0B_R2_PRIMARY_FUSION_SUBJECT_ORDER.csv"
)

NODE_FEATURE_FILE = (
    GRAPH_DIR
    / "STAGE1A_NODE_FEATURES188.npy"
)

ADJ_FILE = (
    GRAPH_DIR
    / "STAGE1A_FUSED_BINARY_ADJ_K4.npy"
)

FUSED_SIM_FILE = (
    GRAPH_DIR
    / "STAGE1A_FUSED_SIMILARITY_ALPHA06.npy"
)

GRAPH_SUMMARY = (
    AUDIT
    / "STAGE1A_FUSED_GRAPH_SUMMARY.json"
)


EXPECTED_HASHES = {

    STAGE1A_LOCK:
        "836526228dba90e4a7333b1df13252e0ddd3966e2ec99fa965ae6157acb2894f",

    SUBJECT_ORDER:
        "8310ccc52f0d0698ed171ded6feb6f14cf4738787b21b93dbedcc0a1d1c99c6e",

    NODE_FEATURE_FILE:
        "1e740ada482052e0eefb6ea57f319c61a03b0466c92c320b0b7de074d0bbf951",

    ADJ_FILE:
        "46dade44f950ffc1c023c167a48bce2a4415b9be5c44459f3244835d826fc153",

    FUSED_SIM_FILE:
        "dffb78d434a699ee4bfacbf8251e1e44003b8b8ed585d1d4490bc9bdf227fddd",

    GRAPH_SUMMARY:
        "bb74c4875839418b5f9c03efd9ad89801eb4f9ff4e889fb52aef0f6bc62f4095"
}


print(
    "\n" + "=" * 120
)

print(
    "FROZEN STAGE1A INPUT HASH VERIFICATION"
)

print(
    "=" * 120
)


for path, expected in EXPECTED_HASHES.items():

    if not path.exists():

        raise FileNotFoundError(
            str(
                path
            )
        )


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
            "Frozen Stage1A input changed. STOP."
        )


print(
    "\nALL STAGE1A INPUT HASHES PASS."
)


# ============================================================
# 5. VERIFY STAGE1A LOCK
# ============================================================

with open(
    STAGE1A_LOCK,
    "r",
    encoding="utf-8"
) as f:

    graph_lock = json.load(
        f
    )


assert graph_lock[
    "status"
] == "FROZEN_BEFORE_VGAE"


assert graph_lock[
    "N"
] == 85


assert graph_lock[
    "view_dimensions"
][
    "node_features"
] == 188


assert graph_lock[
    "fusion"
][
    "alpha"
] == 0.6


assert graph_lock[
    "graph_construction"
][
    "selected_k"
] == 4


assert graph_lock[
    "graph_construction"
][
    "components"
] == 1


assert graph_lock[
    "graph_construction"
][
    "isolates"
] == 0


assert graph_lock[
    "graph_construction"
][
    "undirected_edges"
] == 215


print(
    "\nStage1A graph lock: PASS"
)

print(
    "N = 85"
)

print(
    "Node features = 188"
)

print(
    "Fused graph k = 4"
)

print(
    "Edges = 215"
)


# ============================================================
# 6. LOAD NODE FEATURES + ADJACENCY
# ============================================================

X_np = np.load(
    NODE_FEATURE_FILE
).astype(
    np.float32
)


A_np = np.load(
    ADJ_FILE
).astype(
    np.float32
)


assert X_np.shape == (
    85,
    188
)


assert A_np.shape == (
    85,
    85
)


assert np.isfinite(
    X_np
).all()


assert np.isfinite(
    A_np
).all()


assert np.allclose(
    A_np,
    A_np.T
)


assert np.all(
    np.diag(
        A_np
    )
    ==
    0
)


assert set(
    np.unique(
        A_np
    ).tolist()
).issubset(
    {
        0.0,
        1.0
    }
)


UNDIRECTED_EDGES = int(

    np.count_nonzero(
        np.triu(
            A_np,
            k=1
        )
    )
)


assert UNDIRECTED_EDGES == 215


components, component_labels = connected_components(

    csr_matrix(
        A_np
    ),

    directed=False,

    return_labels=True
)


assert components == 1


degree_raw = A_np.sum(
    axis=1
)


assert np.sum(
    degree_raw == 0
) == 0


print(
    "\nInput node feature shape:",
    X_np.shape
)

print(
    "Adjacency shape:",
    A_np.shape
)

print(
    "Undirected edges:",
    UNDIRECTED_EDGES
)

print(
    "Connected components:",
    components
)


# ============================================================
# 7. FREEZE VGAE TRAINING DESIGN BEFORE FITTING
# ============================================================

SEED = 42

INPUT_DIM = 188

HIDDEN_DIM = 64

LATENT_DIM = 16

LEARNING_RATE = 1e-3

EPOCHS = 500

DROPOUT = 0.0


TRAINING_LOCK_FILE = (
    LOCK
    / "STAGE2A_VGAE_TRAINING_DESIGN_LOCK.json"
)


training_design = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE2A_VGAE_TRAINING",

    "status":
        "FROZEN_BEFORE_FITTING",

    "N":
        85,

    "input_dimension":
        INPUT_DIM,

    "hidden_dimension":
        HIDDEN_DIM,

    "latent_dimension":
        LATENT_DIM,

    "activation":
        "ReLU",

    "dropout":
        DROPOUT,

    "decoder":
        "inner_product",

    "encoder_adjacency":
        "A_plus_I_symmetric_normalization",

    "reconstruction_target":
        "raw_binary_adjacency_offdiagonal_only",

    "reconstruction_loss":
        "binary_cross_entropy_with_logits",

    "positive_class_weight":
        "number_negative_offdiagonal_entries_divided_by_number_positive_offdiagonal_entries",

    "KL_loss":
        "standard_normal_prior",

    "total_loss":
        "weighted_BCE_plus_KL",

    "optimizer":
        "Adam",

    "learning_rate":
        LEARNING_RATE,

    "epochs":
        EPOCHS,

    "early_stopping":
        False,

    "gradient_clipping":
        False,

    "seed":
        SEED,

    "compute_device":
        "CPU",

    "downstream_embedding":
        "posterior_mean_mu",

    "clinical_variables_used":
        False,

    "diagnosis_used":
        False,

    "cluster_labels_used":
        False,

    "input_hashes":
        {

            "Stage1A_lock":
                sha256_file(
                    STAGE1A_LOCK
                ),

            "subject_order":
                sha256_file(
                    SUBJECT_ORDER
                ),

            "node_features188":
                sha256_file(
                    NODE_FEATURE_FILE
                ),

            "adjacency":
                sha256_file(
                    ADJ_FILE
                )
        }
}


if TRAINING_LOCK_FILE.exists():

    with open(
        TRAINING_LOCK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        existing_design = json.load(
            f
        )


    if existing_design != training_design:

        raise RuntimeError(
            "Existing Stage2A training lock differs. STOP."
        )


    print(
        "\nExisting Stage2A training design lock verified."
    )


else:

    with open(
        TRAINING_LOCK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            training_design,
            f,
            indent=2
        )


    print(
        "\nStage2A training design frozen BEFORE VGAE fitting."
    )


print(
    "Training-design SHA256:",
    sha256_file(
        TRAINING_LOCK_FILE
    )
)


# ============================================================
# 8. GRAPH CONVOLUTION
#
# Same dense formulation used in preceding TDBRAIN VGAE:
#
#       A_norm @ (X @ W) + b
# ============================================================

class GraphConvolution(
    nn.Module
):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()


        self.weight = nn.Parameter(

            torch.empty(
                in_features,
                out_features
            )
        )


        self.bias = nn.Parameter(

            torch.zeros(
                out_features
            )
        )


        nn.init.xavier_uniform_(
            self.weight
        )


    def forward(
        self,
        x,
        adj
    ):

        return (

            adj
            @
            (
                x
                @
                self.weight
            )

            +

            self.bias
        )


# ============================================================
# 9. VGAE
# ============================================================

class VGAE(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.gc_hidden = GraphConvolution(

            INPUT_DIM,
            HIDDEN_DIM
        )


        self.gc_mu = GraphConvolution(

            HIDDEN_DIM,
            LATENT_DIM
        )


        self.gc_logvar = GraphConvolution(

            HIDDEN_DIM,
            LATENT_DIM
        )


    def encode(
        self,
        x,
        adj
    ):

        h = F.relu(

            self.gc_hidden(
                x,
                adj
            )
        )


        mu = self.gc_mu(
            h,
            adj
        )


        logvar = self.gc_logvar(
            h,
            adj
        )


        return (
            mu,
            logvar
        )


    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(

            0.5
            *
            logvar
        )


        eps = torch.randn_like(
            std
        )


        return (

            mu

            +

            eps
            *
            std
        )


# ============================================================
# 10. NORMALIZED ADJACENCY
#
# A_tilde = A + I
#
# A_norm =
# D^(-1/2) A_tilde D^(-1/2)
# ============================================================

N = 85


A_tilde = (

    A_np

    +

    np.eye(
        N,
        dtype=np.float32
    )
)


degree = A_tilde.sum(
    axis=1
)


if np.any(
    degree <= 0
):

    raise RuntimeError(
        "Invalid non-positive degree after adding self-loops."
    )


D_inv_sqrt = np.diag(

    1.0

    /

    np.sqrt(
        degree
    )
)


A_norm_np = (

    D_inv_sqrt

    @

    A_tilde

    @

    D_inv_sqrt

).astype(
    np.float32
)


assert A_norm_np.shape == (
    N,
    N
)


assert np.isfinite(
    A_norm_np
).all()


assert np.allclose(
    A_norm_np,
    A_norm_np.T,
    atol=1e-7
)


A_NORM_FILE = (
    VGAE_DIR
    / "STAGE2A_NORMALIZED_ADJACENCY.npy"
)


np.save(
    A_NORM_FILE,
    A_norm_np
)


# ============================================================
# 11. TORCH INPUTS
# ============================================================

X_t = torch.tensor(

    X_np,

    dtype=torch.float32,

    device=DEVICE
)


A_t = torch.tensor(

    A_np,

    dtype=torch.float32,

    device=DEVICE
)


A_norm_t = torch.tensor(

    A_norm_np,

    dtype=torch.float32,

    device=DEVICE
)


# Reconstruction excludes diagonal.

offdiag_np = ~np.eye(
    N,
    dtype=bool
)


offdiag = torch.tensor(

    offdiag_np,

    dtype=torch.bool,

    device=DEVICE
)


target = A_np[
    offdiag_np
]


positive_entries = int(
    target.sum()
)


negative_entries = int(

    len(
        target
    )

    -

    positive_entries
)


assert positive_entries == (
    2
    *
    UNDIRECTED_EDGES
)


pos_weight_value = (

    negative_entries

    /

    positive_entries
)


pos_weight = torch.tensor(

    pos_weight_value,

    dtype=torch.float32,

    device=DEVICE
)


upper_i, upper_j = np.triu_indices(

    N,

    k=1
)


edge_prevalence = float(

    A_np[
        upper_i,
        upper_j
    ].mean()
)


print(
    "\nReconstruction target:"
)

print(
    "Positive off-diagonal entries:",
    positive_entries
)

print(
    "Negative off-diagonal entries:",
    negative_entries
)

print(
    "BCE positive-class weight:",
    f"{pos_weight_value:.6f}"
)

print(
    "Undirected edge prevalence:",
    f"{edge_prevalence:.6f}"
)


# ============================================================
# 12. INITIALIZE FROZEN VGAE
# ============================================================

set_seed(
    SEED
)


vgae = VGAE().to(
    DEVICE
)


parameter_count = sum(

    p.numel()

    for p in vgae.parameters()

    if p.requires_grad
)


EXPECTED_PARAMETER_COUNT = 14176


print(
    "\nVGAE architecture:"
)

print(
    "Input:",
    INPUT_DIM
)

print(
    "Hidden:",
    HIDDEN_DIM
)

print(
    "Latent:",
    LATENT_DIM
)

print(
    "Trainable parameters:",
    parameter_count
)


assert parameter_count == (
    EXPECTED_PARAMETER_COUNT
)


optimizer = torch.optim.Adam(

    vgae.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# 13. TRAIN 500 FIXED EPOCHS
# ============================================================

history_rows = []


print(
    "\n" + "=" * 120
)

print(
    "VGAE TRAINING"
)

print(
    "=" * 120
)


for epoch in range(
    1,
    EPOCHS + 1
):

    vgae.train()


    optimizer.zero_grad(
        set_to_none=True
    )


    mu, logvar = vgae.encode(

        X_t,

        A_norm_t
    )


    z = vgae.reparameterize(

        mu,

        logvar
    )


    logits = (

        z

        @

        z.T
    )


    recon_loss = (
        F.binary_cross_entropy_with_logits(

            logits[
                offdiag
            ],

            A_t[
                offdiag
            ],

            pos_weight=pos_weight
        )
    )


    kl_loss = (

        -0.5

        *

        torch.mean(

            torch.sum(

                1
                +
                logvar
                -
                mu.pow(
                    2
                )
                -
                torch.exp(
                    logvar
                ),

                dim=1
            )
        )
    )


    total_loss = (

        recon_loss

        +

        kl_loss
    )


    if not torch.isfinite(
        total_loss
    ):

        raise RuntimeError(
            f"Non-finite loss at epoch {epoch}. STOP."
        )


    total_loss.backward()


    # IMPORTANT:
    # No gradient clipping.
    # No early stopping.

    optimizer.step()


    history_rows.append({

        "epoch":
            epoch,

        "total_loss":
            float(
                total_loss.item()
            ),

        "reconstruction_loss":
            float(
                recon_loss.item()
            ),

        "KL_loss":
            float(
                kl_loss.item()
            )
    })


    if (

        epoch == 1

        or

        epoch % 100 == 0

        or

        epoch == EPOCHS

    ):

        print(

            f"VGAE {epoch:03d}/{EPOCHS} | "

            f"total={total_loss.item():.6f} | "

            f"recon={recon_loss.item():.6f} | "

            f"KL={kl_loss.item():.6f}"
        )


history = pd.DataFrame(
    history_rows
)


HISTORY_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_TRAINING_HISTORY.csv"
)


history.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# 14. FINAL POSTERIOR MEAN
#
# mu is the ONLY downstream embedding.
# ============================================================

vgae.eval()


with torch.no_grad():

    mu_final, logvar_final = vgae.encode(

        X_t,

        A_norm_t
    )


    reconstruction_prob = torch.sigmoid(

        mu_final

        @

        mu_final.T
    )


mu_np = (
    mu_final
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


logvar_np = (
    logvar_final
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


prob_np = (
    reconstruction_prob
    .cpu()
    .numpy()
)


assert mu_np.shape == (
    85,
    16
)


assert logvar_np.shape == (
    85,
    16
)


assert np.isfinite(
    mu_np
).all()


assert np.isfinite(
    logvar_np
).all()


MU_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.npy"
)


LOGVAR_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_LOGVAR16.npy"
)


np.save(
    MU_FILE,
    mu_np
)


np.save(
    LOGVAR_FILE,
    logvar_np
)


# ============================================================
# 15. SAVE MODEL PARAMETERS
# ============================================================

MODEL_FILE = (
    VGAE_DIR
    / "STAGE2A_VGAE_MODEL_STATE.pt"
)


torch.save(

    {
        "state_dict":
            vgae.state_dict(),

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "latent_dim":
            LATENT_DIM,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE
    },

    MODEL_FILE
)


# ============================================================
# 16. LATENT NON-COLLAPSE AUDIT
# ============================================================

latent_sd = np.std(

    mu_np.astype(
        np.float64
    ),

    axis=0,

    ddof=0
)


ACTIVE_THRESHOLD = 1e-8


active_dimensions = int(

    np.sum(
        latent_sd
        >
        ACTIVE_THRESHOLD
    )
)


latent_rank = int(

    np.linalg.matrix_rank(

        mu_np.astype(
            np.float64
        )
    )
)


LATENT_AUDIT_FILE = (
    AUDIT
    / "STAGE2A_VGAE_LATENT_DIMENSION_AUDIT.csv"
)


latent_audit = pd.DataFrame({

    "latent_dimension":
        np.arange(
            16
        ),

    "posterior_mu_SD":
        latent_sd,

    "active":
        latent_sd
        >
        ACTIVE_THRESHOLD
})


latent_audit.to_csv(
    LATENT_AUDIT_FILE,
    index=False
)


print(
    "\n" + "=" * 120
)

print(
    "VGAE LATENT AUDIT"
)

print(
    "=" * 120
)


print(
    "Embedding shape:",
    mu_np.shape
)

print(
    "Matrix rank:",
    latent_rank,
    "/ 16"
)

print(
    "Active latent dimensions:",
    active_dimensions,
    "/ 16"
)

print(
    "Minimum latent SD:",
    f"{latent_sd.min():.10f}"
)

print(
    "Maximum latent SD:",
    f"{latent_sd.max():.10f}"
)


# Do not silently continue from a collapsed VGAE.

if latent_rank < 16:

    raise RuntimeError(
        "VGAE posterior mean is rank deficient. STOP before UMAP."
    )


if active_dimensions < 16:

    raise RuntimeError(
        "One or more VGAE latent dimensions are inactive. "
        "STOP before UMAP."
    )


print(
    "LATENT NON-COLLAPSE: PASS"
)


# ============================================================
# 17. GRAPH RECONSTRUCTION DIAGNOSTICS
#
# These are training-fit diagnostics only.
# They are NOT external validation.
# ============================================================

y_true = A_np[
    upper_i,
    upper_j
]


y_prob = prob_np[
    upper_i,
    upper_j
]


vgae_auc = float(

    roc_auc_score(
        y_true,
        y_prob
    )
)


vgae_ap = float(

    average_precision_score(
        y_true,
        y_prob
    )
)


print(
    "\nGraph reconstruction diagnostics:"
)

print(
    "ROC-AUC:",
    f"{vgae_auc:.6f}"
)

print(
    "Average precision:",
    f"{vgae_ap:.6f}"
)

print(
    "Edge prevalence:",
    f"{edge_prevalence:.6f}"
)


# ============================================================
# 18. SUBJECT-EMBEDDING TABLE
#
# No diagnosis or clinical variables included.
# ============================================================

subject_order = pd.read_csv(
    SUBJECT_ORDER
)


embedding_table = subject_order[
    [
        "experiment3_row",
        "subject",
        "subject_norm"
    ]
].copy()


for j in range(
    LATENT_DIM
):

    embedding_table[
        f"VGAE_mu_{j:02d}"
    ] = mu_np[
        :,
        j
    ]


EMBEDDING_CSV = (
    VGAE_DIR
    / "STAGE2A_VGAE_POSTERIOR_MU16.csv"
)


embedding_table.to_csv(
    EMBEDDING_CSV,
    index=False
)


# ============================================================
# 19. RESULT SUMMARY
# ============================================================

final_history = history.iloc[
    -1
]


SUMMARY_FILE = (
    AUDIT
    / "STAGE2A_VGAE_SUMMARY.json"
)


summary = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE2A_FROZEN_VGAE",

    "status":
        "COMPUTATIONAL_PASS",

    "N":
        85,

    "input_dimension":
        188,

    "hidden_dimension":
        64,

    "latent_dimension":
        16,

    "trainable_parameters":
        parameter_count,

    "graph_k":
        4,

    "undirected_edges":
        215,

    "graph_components":
        1,

    "graph_isolates":
        0,

    "positive_offdiagonal_entries":
        positive_entries,

    "negative_offdiagonal_entries":
        negative_entries,

    "positive_class_weight":
        float(
            pos_weight_value
        ),

    "edge_prevalence":
        edge_prevalence,

    "epochs":
        500,

    "learning_rate":
        0.001,

    "seed":
        42,

    "compute_device":
        "CPU",

    "final_total_loss":
        float(
            final_history[
                "total_loss"
            ]
        ),

    "final_reconstruction_loss":
        float(
            final_history[
                "reconstruction_loss"
            ]
        ),

    "final_KL_loss":
        float(
            final_history[
                "KL_loss"
            ]
        ),

    "posterior_mu_shape":
        [
            85,
            16
        ],

    "posterior_mu_rank":
        latent_rank,

    "active_latent_dimensions":
        active_dimensions,

    "reconstruction_ROC_AUC":
        vgae_auc,

    "reconstruction_average_precision":
        vgae_ap,

    "downstream_representation":
        "posterior_mean_mu",

    "diagnosis_used":
        False,

    "clinical_scores_used":
        False,

    "Experiment2_labels_used":
        False,

    "UMAP_run":
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
# 20. HUMAN-READABLE REPORT
# ============================================================

REPORT_FILE = (
    AUDIT
    / "STAGE2A_VGAE_REPORT.txt"
)


report = f"""
TDBRAIN TRANSDIAGNOSTIC EXPERIMENT 3
STAGE 2A — FROZEN VGAE

INPUT
-----
Subjects:
85

Node feature dimension:
188

Fused graph k:
4

Undirected graph edges:
215

Connected components:
1

Isolated subjects:
0


VGAE
----
Architecture:
188 -> 64 -> 16

Activation:
ReLU

Dropout:
0.0

Decoder:
Inner product

Optimizer:
Adam

Learning rate:
0.001

Epochs:
500

Seed:
42

Compute:
CPU

Early stopping:
NO

Gradient clipping:
NO


FINAL TRAINING ENDPOINT
-----------------------
Total loss:
{float(final_history["total_loss"]):.10f}

Reconstruction loss:
{float(final_history["reconstruction_loss"]):.10f}

KL loss:
{float(final_history["KL_loss"]):.10f}


POSTERIOR MEAN
--------------
Shape:
85 x 16

Rank:
{latent_rank} / 16

Active latent dimensions:
{active_dimensions} / 16


GRAPH RECONSTRUCTION DIAGNOSTICS
--------------------------------
ROC-AUC:
{vgae_auc:.10f}

Average precision:
{vgae_ap:.10f}

Edge prevalence:
{edge_prevalence:.10f}

These are training-fit graph-reconstruction diagnostics.
They are not clinical or external validation.


DESIGN PROTECTION
-----------------
Diagnosis used:
NO

ADHD / YBOCS / BDI used:
NO

Experiment2 cluster labels used:
NO

UMAP run:
NO

Clustering run:
NO


FINAL STATUS
------------
STAGE 2A COMPUTATIONAL PASS —
FROZEN VGAE TRAINING COMPLETE.

The downstream representation is posterior mean mu.
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report
    )


# ============================================================
# 21. OUTPUT LOCK
# ============================================================

OUTPUT_LOCK_FILE = (
    LOCK
    / "STAGE2A_VGAE_OUTPUT_LOCK.json"
)


output_lock = {

    "experiment":
        "TDBRAIN_TRANSDIAGNOSTIC_EXPERIMENT3_EEG_PHENOTYPE_FUSION",

    "stage":
        "STAGE2A_VGAE_OUTPUT_LOCK",

    "status":
        "FROZEN_BEFORE_UMAP",

    "N":
        85,

    "architecture":
        "188_to_64_to_16",

    "posterior_mu_rank":
        latent_rank,

    "active_latent_dimensions":
        active_dimensions,

    "UMAP_run":
        False,

    "clustering_run":
        False,

    "input_hashes":
        {

            "Stage1A_lock":
                sha256_file(
                    STAGE1A_LOCK
                ),

            "node_features":
                sha256_file(
                    NODE_FEATURE_FILE
                ),

            "adjacency":
                sha256_file(
                    ADJ_FILE
                ),

            "Stage2A_training_design":
                sha256_file(
                    TRAINING_LOCK_FILE
                )
        },

    "output_hashes":
        {

            "normalized_adjacency":
                sha256_file(
                    A_NORM_FILE
                ),

            "posterior_mu":
                sha256_file(
                    MU_FILE
                ),

            "posterior_logvar":
                sha256_file(
                    LOGVAR_FILE
                ),

            "posterior_mu_csv":
                sha256_file(
                    EMBEDDING_CSV
                ),

            "training_history":
                sha256_file(
                    HISTORY_FILE
                ),

            "latent_audit":
                sha256_file(
                    LATENT_AUDIT_FILE
                ),

            "model_state":
                sha256_file(
                    MODEL_FILE
                ),

            "summary":
                sha256_file(
                    SUMMARY_FILE
                ),

            "report":
                sha256_file(
                    REPORT_FILE
                )
        }
}


if OUTPUT_LOCK_FILE.exists():

    with open(
        OUTPUT_LOCK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        existing_output_lock = json.load(
            f
        )


    if existing_output_lock != output_lock:

        raise RuntimeError(
            "Existing Stage2A output lock differs. STOP."
        )


    print(
        "\nExisting Stage2A output lock verified."
    )


else:

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


    print(
        "\nStage2A VGAE output lock created."
    )


# ============================================================
# 22. FINAL HASHES
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "STAGE 2A OUTPUT HASHES"
)

print(
    "=" * 120
)


print(
    "Training design:",
    sha256_file(
        TRAINING_LOCK_FILE
    )
)

print(
    "Normalized adjacency:",
    sha256_file(
        A_NORM_FILE
    )
)

print(
    "Posterior mu16:",
    sha256_file(
        MU_FILE
    )
)

print(
    "Posterior logvar16:",
    sha256_file(
        LOGVAR_FILE
    )
)

print(
    "Embedding CSV:",
    sha256_file(
        EMBEDDING_CSV
    )
)

print(
    "Training history:",
    sha256_file(
        HISTORY_FILE
    )
)

print(
    "Latent audit:",
    sha256_file(
        LATENT_AUDIT_FILE
    )
)

print(
    "Summary:",
    sha256_file(
        SUMMARY_FILE
    )
)

print(
    "Report:",
    sha256_file(
        REPORT_FILE
    )
)

print(
    "Output lock:",
    sha256_file(
        OUTPUT_LOCK_FILE
    )
)


# ============================================================
# 23. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "FINAL STATUS: EXPERIMENT 3 STAGE 2A PASS — "
    "FROZEN VGAE COMPLETE"
)

print(
    "=" * 120
)


print(
    "\nVGAE embedding:",
    mu_np.shape
)

print(
    "Rank:",
    latent_rank,
    "/ 16"
)

print(
    "Active dimensions:",
    active_dimensions,
    "/ 16"
)

print(
    "ROC-AUC:",
    f"{vgae_auc:.6f}"
)

print(
    "Average precision:",
    f"{vgae_ap:.6f}"
)

print(
    "Edge prevalence:",
    f"{edge_prevalence:.6f}"
)


print(
    "\nDOWNSTREAM EMBEDDING = POSTERIOR MEAN mu"
)


print(
    "\nNO UMAP HAS BEEN RUN."
)

print(
    "NO CLUSTER NUMBER HAS BEEN SELECTED."
)

print(
    "NO CLINICAL VARIABLES HAVE BEEN OPENED FOR MODEL SELECTION."
)


print(
    "\nNEXT PIPELINE STAGE:"
)

print(
    "STAGE 3A — UMAP MULTI-SEED STRUCTURAL AUDIT "
    "AND EEG+PHENOTYPE-ONLY K SELECTION."
)