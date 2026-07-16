"""
EPG-SD embedding stream 2.

Second of two separately saved, same-architecture frequency-wise
Conv1D autoencoder embedding streams. Five 128-dimensional band
embeddings are produced (640 features before ten-view fusion).
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

SEED = 42
DATA_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/multiview_tensors_FS500")
OUTPUT_DIR = Path("./TECHNICAL_FIX_SPECTRAL_UMAP/cnn_autoencoder_embeddings_norm_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 12
LATENT_DIM = 128
LR = 1e-3
VAL_SPLIT = 0.15
VARIANCE_WEIGHT = 0.01

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def normalize_per_sample_channel(X, eps=1e-6):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)

class EEGDataset(Dataset):
    def __init__(self, X): self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]

class DynamicEEGAutoEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels,64,7,padding=3), nn.GroupNorm(8,64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2), nn.GroupNorm(8,128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128,256,3,padding=1), nn.GroupNorm(16,256), nn.GELU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc_encoder = nn.Linear(256, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Conv1d(256,128,3,padding=1), nn.GELU(),
            nn.Conv1d(128,64,3,padding=1), nn.GELU(),
            nn.Conv1d(64,n_channels,3,padding=1)
        )
    def encode(self, x):
        return self.fc_encoder(self.encoder(x).squeeze(-1))
    def decode(self, z, target_len):
        h = self.fc_decoder(z).unsqueeze(-1)
        h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        return self.decoder(h)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.shape[-1]), z

def latent_variance_loss(z, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))

def train_one_band(X, band):
    X = normalize_per_sample_channel(X)
    print(f"{band} normalized input mean={X.mean():.4f}, std={X.std():.4f}")
    dataset = EEGDataset(X)
    n_val = max(1, int(len(dataset) * VAL_SPLIT)); n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = DynamicEEGAutoEncoder(X.shape[1], LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val = float("inf"); bad = 0; hist = []
    best_path = OUTPUT_DIR / f"{band}_cnn_norm_autoencoder_best.pt"
    for epoch in range(1, EPOCHS+1):
        model.train(); totals=[]; recs=[]; vars_=[]
        for batch in tqdm(train_loader, desc=f"{band} epoch {epoch}/{EPOCHS}"):
            batch = batch.to(DEVICE); opt.zero_grad()
            recon,z = model(batch)
            rec = criterion(recon,batch); var = latent_variance_loss(z)
            loss = rec + VARIANCE_WEIGHT * var
            loss.backward(); opt.step()
            totals.append(loss.item()); recs.append(rec.item()); vars_.append(var.item())
        model.eval(); vt=[]; vr=[]; vv=[]
        with torch.no_grad():
            for batch in val_loader:
                batch=batch.to(DEVICE); recon,z=model(batch)
                rec=criterion(recon,batch); var=latent_variance_loss(z); loss=rec+VARIANCE_WEIGHT*var
                vt.append(loss.item()); vr.append(rec.item()); vv.append(var.item())
        row={"band":band,"epoch":epoch,"train_total":float(np.mean(totals)),"train_reconstruction_mse":float(np.mean(recs)),"train_latent_variance_loss":float(np.mean(vars_)),"val_total":float(np.mean(vt)),"val_reconstruction_mse":float(np.mean(vr)),"val_latent_variance_loss":float(np.mean(vv))}
        hist.append(row)
        print(f"{band} epoch {epoch}: train_total={row['train_total']:.4f} val_total={row['val_total']:.4f} val_rec={row['val_reconstruction_mse']:.4f} val_var={row['val_latent_variance_loss']:.4f}")
        if row["val_total"] < best_val:
            best_val=row["val_total"]; bad=0; torch.save(model.state_dict(), best_path)
        else:
            bad+=1
            if bad>=PATIENCE:
                print("Early stopping."); break
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()
    embs=[]
    with torch.no_grad():
        for batch in tqdm(full_loader, desc=f"extract {band}"):
            embs.append(model.encode(batch.to(DEVICE)).cpu().numpy())
    embs=np.concatenate(embs,axis=0)
    np.save(OUTPUT_DIR / f"{band}_cnn_embedding_fixed.npy", embs)
    pd.DataFrame(embs).to_csv(OUTPUT_DIR / f"{band}_cnn_embedding_fixed.csv", index=False)
    pd.DataFrame(hist).to_csv(OUTPUT_DIR / f"{band}_training_log.csv", index=False)
    print(f"{band} embedding health: mean={embs.mean():.6f}, std={embs.std():.6f}, min={embs.min():.6f}, max={embs.max():.6f}, zero_fraction={(np.abs(embs)<1e-8).mean():.4f}")
    return pd.DataFrame(hist)

def main():
    set_seed(SEED)
    print("\nNORMALIZED CNN AUTOENCODER EMBEDDINGS"); print("Device:", DEVICE); print("DATA_DIR:", DATA_DIR)
    logs=[]
    for band in BANDS:
        X=np.load(DATA_DIR / f"{band}.npy").astype(np.float32)
        print("\nBand:", band, "shape:", X.shape)
        logs.append(train_one_band(X, band))
    pd.concat(logs, ignore_index=True).to_csv(OUTPUT_DIR / "ALL_BANDS_cnn_training_log.csv", index=False)
    print("\nDONE. Saved:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
