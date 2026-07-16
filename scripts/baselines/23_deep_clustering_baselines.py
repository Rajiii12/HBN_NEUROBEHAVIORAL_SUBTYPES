#!/usr/bin/env python3
"""
Deep-clustering baseline comparison:
Add deep clustering baselines:
- DEC
- IDEC
- DeepCluster
- VAE-GMM / VaDE-style
- GAE-KMeans / deep embedded graph clustering-style

All methods use the same final subject-level representation:
EEG embedding 192 + phenotype 4 = 196 features.
K = 5 for all methods.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Fixed settings
# ============================================================
SEED = 42
K = 5
LATENT_DIM = 16
AE_EPOCHS = 300
DEC_EPOCHS = 200
IDEC_EPOCHS = 200
DEEPCLUSTER_EPOCHS = 200
VAE_EPOCHS = 300
GAE_EPOCHS = 300
LR = 1e-3

OUT: Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Clinical helper functions
# ============================================================
def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = pvals[order[i]] * n / rank
        prev = min(prev, val)
        adj[order[i]] = min(prev, 1.0)
    return adj


def eta_squared_anova(y, labels):
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)

    groups = []
    for g in np.unique(labels):
        vals = y[labels == g]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            groups.append(vals)

    if len(groups) < 2:
        return 1.0, 0.0

    _, p = stats.f_oneway(*groups)
    gm = np.nanmean(y)
    sst = np.nansum((y - gm) ** 2)
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    eta2 = ssb / sst if sst > 0 else 0.0
    return float(p), float(eta2)


def clinical_stats(meta, labels):
    vars_cols = {
        "p_factor": "p_factor",
        "attention": "attention",
        "internalizing": "internalizing",
        "externalizing": "externalizing",
        "age": "age",
        "ehq_total": "ehq_total",
    }

    rows = []
    pvals = []
    for var, col in vars_cols.items():
        y = pd.to_numeric(meta[col], errors="coerce").values
        p, e2 = eta_squared_anova(y, labels)
        rows.append({"variable": var, "anova_p": p, "eta_squared": e2})
        pvals.append(p)

    fdr = bh_fdr(pvals)
    for i in range(len(rows)):
        rows[i]["anova_p_fdr"] = float(fdr[i])
        rows[i]["significant_fdr_0p05"] = bool(fdr[i] < 0.05)

    return pd.DataFrame(rows)


def summarize_method(method, X_eval, labels, meta, final_labels=None):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=K)

    sil = silhouette_score(X_eval, labels)
    db = davies_bouldin_score(X_eval, labels)
    ch = calinski_harabasz_score(X_eval, labels)

    min_cluster = int(counts.min())
    max_cluster = int(counts.max())
    balance = float(max_cluster / max(min_cluster, 1))

    clin = clinical_stats(meta, labels)
    clin.to_csv(OUT / f"{method.replace(' ', '_').replace('/', '_')}_clinical_stats.csv", index=False)

    fdr_count = int(clin["significant_fdr_0p05"].sum())
    mean_eta = float(clin["eta_squared"].mean())

    row = {
        "method": method,
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "min_cluster_size": min_cluster,
        "max_cluster_size": max_cluster,
        "cluster_size_ratio_max_min": balance,
        "mean_eta_squared": mean_eta,
        "fdr_sig_count": f"{fdr_count}/6",
    }

    if final_labels is not None:
        row["ARI_vs_EPGSD"] = float(adjusted_rand_score(final_labels, labels))
        row["NMI_vs_EPGSD"] = float(normalized_mutual_info_score(final_labels, labels))

    pd.DataFrame({
        "subject_index": np.arange(len(labels)),
        "subtype": labels
    }).to_csv(OUT / f"{method.replace(' ', '_').replace('/', '_')}_labels.csv", index=False)

    pd.DataFrame({
        "cluster": np.arange(K),
        "size": counts
    }).to_csv(OUT / f"{method.replace(' ', '_').replace('/', '_')}_cluster_sizes.csv", index=False)

    return row


# ============================================================
# Models
# ============================================================
class MLP_AE(nn.Module):
    def __init__(self, in_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr, z


class MLP_VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-5, 5)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xr = self.dec(z)
        return xr, mu, logvar, z


class DeepClusterModel(nn.Module):
    def __init__(self, in_dim, latent_dim=16, n_clusters=5):
        super().__init__()
        self.ae = MLP_AE(in_dim, latent_dim)
        self.head = nn.Linear(latent_dim, n_clusters)

    def forward(self, x):
        xr, z = self.ae(x)
        logits = self.head(z)
        return xr, z, logits


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.lin(adj @ x)


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, latent_dim)

    def encode(self, x, adj):
        h = F.relu(self.gcn1(x, adj))
        z = self.gcn2(h, adj)
        return z

    def decode_logits(self, z):
        return z @ z.T

    def forward(self, x, adj):
        z = self.encode(x, adj)
        logits = self.decode_logits(z)
        return logits, z


# ============================================================
# Training functions
# ============================================================
def train_ae(X):
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model = MLP_AE(X.shape[1], LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, AE_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        xr, z = model(x)
        loss = F.mse_loss(xr, x)
        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == 1:
            print(f"AE epoch={ep:03d} loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        _, z = model(x)
    return model, z.cpu().numpy()


def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p


def soft_assign(z, centers, alpha=1.0):
    dist = torch.sum((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = 1.0 / (1.0 + dist / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


def run_dec(X):
    print("\nRunning DEC")
    ae, z_np = train_ae(X)

    km = KMeans(n_clusters=K, random_state=SEED, n_init=20)
    km.fit(z_np)

    centers = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE))
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(list(ae.encoder.parameters()) + [centers], lr=LR)

    for ep in range(1, DEC_EPOCHS + 1):
        opt.zero_grad()
        z = ae.encoder(x)
        q = soft_assign(z, centers)
        p = target_distribution(q).detach()
        loss = F.kl_div(torch.log(q + 1e-9), p, reduction="batchmean")
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == 1:
            print(f"DEC epoch={ep:03d} KL={loss.item():.5f}")

    ae.eval()
    with torch.no_grad():
        z = ae.encoder(x)
        q = soft_assign(z, centers)
        labels = torch.argmax(q, dim=1).cpu().numpy()
        z_np = z.cpu().numpy()
    return z_np, labels


def run_idec(X):
    print("\nRunning IDEC")
    ae, z_np = train_ae(X)

    km = KMeans(n_clusters=K, random_state=SEED, n_init=20)
    km.fit(z_np)

    centers = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE))
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(list(ae.parameters()) + [centers], lr=LR)

    for ep in range(1, IDEC_EPOCHS + 1):
        opt.zero_grad()
        xr, z = ae(x)
        q = soft_assign(z, centers)
        p = target_distribution(q).detach()
        kl = F.kl_div(torch.log(q + 1e-9), p, reduction="batchmean")
        recon = F.mse_loss(xr, x)
        loss = kl + 0.1 * recon
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == 1:
            print(f"IDEC epoch={ep:03d} loss={loss.item():.5f} KL={kl.item():.5f} recon={recon.item():.5f}")

    ae.eval()
    with torch.no_grad():
        _, z = ae(x)
        q = soft_assign(z, centers)
        labels = torch.argmax(q, dim=1).cpu().numpy()
        z_np = z.cpu().numpy()
    return z_np, labels


def run_deepcluster(X):
    print("\nRunning DeepCluster-style baseline")
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    model = DeepClusterModel(X.shape[1], LATENT_DIM, K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    labels_np = None
    for ep in range(1, DEEPCLUSTER_EPOCHS + 1):
        model.eval()
        with torch.no_grad():
            _, z, _ = model(x)
            z_np = z.cpu().numpy()
        labels_np = KMeans(n_clusters=K, random_state=SEED + ep, n_init=10).fit_predict(z_np)
        labels_t = torch.tensor(labels_np, dtype=torch.long, device=DEVICE)

        model.train()
        opt.zero_grad()
        xr, z, logits = model(x)
        recon = F.mse_loss(xr, x)
        ce = F.cross_entropy(logits, labels_t)
        loss = ce + 0.1 * recon
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == 1:
            print(f"DeepCluster epoch={ep:03d} loss={loss.item():.5f} CE={ce.item():.5f} recon={recon.item():.5f}")

    model.eval()
    with torch.no_grad():
        _, z, logits = model(x)
        z_np = z.cpu().numpy()
        labels = torch.argmax(logits, dim=1).cpu().numpy()

    return z_np, labels


def run_vade_style(X):
    print("\nRunning VAE-GMM / VaDE-style baseline")
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model = MLP_VAE(X.shape[1], LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, VAE_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        xr, mu, logvar, z = model(x)
        recon = F.mse_loss(xr, x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon + 0.01 * kl
        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == 1:
            print(f"VAE epoch={ep:03d} loss={loss.item():.5f} recon={recon.item():.5f} KL={kl.item():.5f}")

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z_np = mu.cpu().numpy()

    labels = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=SEED,
        n_init=10
    ).fit_predict(z_np)

    return z_np, labels


def normalize_adj(A):
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(d + 1e-9)
    return dinv[:, None] * A * dinv[None, :]


def run_gae_kmeans(X, adj_path):
    print("\nRunning GAE-KMeans / deep graph clustering-style baseline")

    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adjacency file: {adj_path}")

    A = np.load(adj_path)
    A_bin = (A > 0).astype(np.float32)
    A_norm = normalize_adj(A).astype(np.float32)

    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    adj_t = torch.tensor(A_norm, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(A_bin, dtype=torch.float32, device=DEVICE)

    model = GAE(X.shape[1], hidden_dim=64, latent_dim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1e-9)).clamp(min=1.0, max=50.0)

    for ep in range(1, GAE_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits, z = model(x, adj_t)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == 1:
            print(f"GAE epoch={ep:03d} loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(x, adj_t)
        z_np = z.cpu().numpy()

    labels = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit_predict(z_np)
    return z_np, labels




def parse_args():
    p=argparse.ArgumentParser(description="DEC, IDEC, DeepCluster, VAE-GMM, and GAE-KMeans baselines")
    p.add_argument("--eeg-features",type=Path,required=True)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--adjacency",type=Path,required=True)
    p.add_argument("--reference-labels",type=Path,default=None,help="Optional final EPG-SD labels used only for ARI/NMI reporting")
    p.add_argument("--out",type=Path,required=True)
    return p.parse_args()

def read_reference_labels(path,expected_n):
    if path is None: return None
    df=pd.read_csv(path); preferred=[c for c in df.columns if any(k in c.lower() for k in ("subtype","cluster","label"))]
    for col in preferred+list(df.select_dtypes(include=[np.number]).columns):
        values=pd.to_numeric(df[col],errors="coerce")
        if len(values)==expected_n and values.notna().all() and np.allclose(values,np.round(values)): return np.round(values).astype(int).to_numpy()
    raise ValueError("Could not identify a discrete reference-label column")

def main():
    global OUT
    a=parse_args(); OUT=a.out.expanduser().resolve(); OUT.mkdir(parents=True,exist_ok=True); set_seed(SEED)
    eeg=np.load(a.eeg_features.expanduser().resolve()); meta=pd.read_csv(a.metadata.expanduser().resolve())
    if eeg.ndim!=2 or eeg.shape[0]!=len(meta) or len(meta)!=430: raise ValueError("Expected aligned 430-subject EEG and metadata inputs")
    pheno_cols=["p_factor","attention","internalizing","externalizing"]
    ph=meta[pheno_cols].apply(pd.to_numeric,errors="coerce"); ph=ph.fillna(ph.median(numeric_only=True)).to_numpy(float)
    X=StandardScaler().fit_transform(np.nan_to_num(np.concatenate([eeg,ph],axis=1)))
    if X.shape!=(430,196): raise ValueError(f"Expected reconstructed baseline input shape (430,196), found {X.shape}")
    reference=read_reference_labels(a.reference_labels,430)
    rows=[]
    set_seed(SEED); z,lab=run_dec(X); np.save(OUT/"DEC_latent.npy",z); rows.append(summarize_method("DEC",z,lab,meta,reference))
    set_seed(SEED); z,lab=run_idec(X); np.save(OUT/"IDEC_latent.npy",z); rows.append(summarize_method("IDEC",z,lab,meta,reference))
    set_seed(SEED); z,lab=run_deepcluster(X); np.save(OUT/"DeepCluster_latent.npy",z); rows.append(summarize_method("DeepCluster-style",z,lab,meta,reference))
    set_seed(SEED); z,lab=run_vade_style(X); np.save(OUT/"VAE_GMM_VaDE_style_latent.npy",z); rows.append(summarize_method("VAE-GMM/VaDE-style",z,lab,meta,reference))
    set_seed(SEED); z,lab=run_gae_kmeans(X,a.adjacency.expanduser().resolve()); np.save(OUT/"GAE_KMeans_latent.npy",z); rows.append(summarize_method("GAE-KMeans graph clustering",z,lab,meta,reference))
    summary=pd.DataFrame(rows); summary.to_csv(OUT/"deep_clustering_baselines_only_summary.csv",index=False)
    latex = []
    for _, r in summary.iterrows():
        method_tex = str(r["method"]).replace("_", r"\_")
        latex.append(
            f"{method_tex} & {r['silhouette']:.3f} & "
            f"{r['davies_bouldin']:.3f} & "
            f"{r['calinski_harabasz']:.1f} & "
            f"{int(r['min_cluster_size'])} & "
            f"{r['cluster_size_ratio_max_min']:.2f} & "
            f"{r['mean_eta_squared']:.3f} & "
            f"{r['fdr_sig_count']} \\\\"
        )
    (OUT/"deep_clustering_baseline_rows.tex").write_text("\n".join(latex)+"\n")
    (OUT/"run_manifest.json").write_text(json.dumps({"eeg_features":a.eeg_features.name,"metadata":a.metadata.name,"adjacency":a.adjacency.name,"reference_labels":a.reference_labels.name if a.reference_labels else None,"phenotype_column_order":pheno_cols,"seed":SEED,"K":K,"latent_dim":LATENT_DIM,"epochs":{"AE":AE_EPOCHS,"DEC":DEC_EPOCHS,"IDEC":IDEC_EPOCHS,"DeepCluster":DEEPCLUSTER_EPOCHS,"VAE":VAE_EPOCHS,"GAE":GAE_EPOCHS},"python":sys.version,"platform":platform.platform()},indent=2))
    print(summary.to_string(index=False)); print("The EPG-SD comparison row is intentionally not recomputed here; use the archived final Stage 09 metrics."); print("Saved outputs to:",OUT); return 0
if __name__=="__main__": raise SystemExit(main())
