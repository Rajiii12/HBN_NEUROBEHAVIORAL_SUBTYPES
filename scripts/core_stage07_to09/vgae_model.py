"""PyTorch Geometric variational graph autoencoder components.

The encoder uses a 64-unit graph-convolution layer followed by separate
16-dimensional posterior-mean and log-standard-deviation heads. Training
logs reconstruction, KL, and total losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE

class VGAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        return mu, logstd

class VGAEModel(VGAE):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__(VGAEEncoder(input_dim, hidden_dim, latent_dim))

def train_vgae_logged(model, x, train_pos_edge_index, epochs=500, lr=0.001, kl_weight=1.0, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)

        recon_loss = model.recon_loss(z, train_pos_edge_index)
        kl_loss = (1.0 / x.size(0)) * model.kl_loss()
        total_loss = recon_loss + kl_weight * kl_loss

        total_loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "reconstruction_loss": float(recon_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "kl_weight": float(kl_weight),
            "total_loss": float(total_loss.detach().cpu()),
        }
        history.append(row)

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"recon={row['reconstruction_loss']:.6f} | "
                f"kl={row['kl_loss']:.6f} | "
                f"total={row['total_loss']:.6f}"
            )

    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    return z, history
