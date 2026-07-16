"""Train the VGAE for EEG-only, phenotype-only, and fused graphs.

The manuscript configuration uses hidden dimension 64, latent dimension 16,
500 epochs, Adam learning rate 0.001, and no dropout. The representation
saved after model.eval() is the posterior mean returned by VGAE.encode().
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse

from vgae_model import VGAEModel, train_vgae_logged

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAPH_ROOT = Path(os.getenv(
    "EPGSD_GRAPH_ROOT",
    "./TECHNICAL_FIX_SPECTRAL_UMAP/graphs_MONTAGE_FIXED_graphK12_alpha06",
))
OUTPUT_ROOT = Path(os.getenv(
    "EPGSD_VGAE_OUTPUT_ROOT",
    "./TECHNICAL_FIX_SPECTRAL_UMAP/vgae_MONTAGE_FIXED_graphK12_alpha06",
))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["EEG_ONLY", "PHENOTYPE_ONLY", "EEG_PHENOTYPE"]

HIDDEN_DIM = 64
LATENT_DIM = 16
EPOCHS = 500
LR = 0.001
KL_WEIGHT = 1.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def train_one_experiment(name):
    print("\n===================================")
    print("VGAE:", name)
    print("===================================")

    in_dir = GRAPH_ROOT / name
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_dir / "node_features.npy")
    A = np.load(in_dir / "adjacency.npy")

    x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    adj = torch.tensor(A, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(DEVICE)

    model = VGAEModel(input_dim=x.shape[1], hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)

    z, history = train_vgae_logged(
        model,
        x,
        edge_index,
        epochs=EPOCHS,
        lr=LR,
        kl_weight=KL_WEIGHT,
        device=DEVICE
    )

    z_np = z.detach().cpu().numpy()

    np.save(out_dir / "posterior_mean.npy", z_np)
    np.save(out_dir / "latent_embeddings.npy", z_np)  # Archived filename.
    pd.DataFrame(z_np).to_csv(out_dir / "posterior_mean.csv", index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "vgae_training_log.csv", index=False)

    torch.save(model.state_dict(), out_dir / "vgae_model.pt")

    summary = pd.DataFrame({
        "metric": ["experiment", "n_subjects", "input_dim", "hidden_dim", "latent_dim", "epochs", "lr"],
        "value": [name, X.shape[0], X.shape[1], HIDDEN_DIM, LATENT_DIM, EPOCHS, LR]
    })
    summary.to_csv(out_dir / "vgae_summary.csv", index=False)

    print("Saved:", out_dir)
    return z_np

def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    for exp in EXPERIMENTS:
        train_one_experiment(exp)
    print("\nDONE. Saved:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
