#!/usr/bin/env python3
"""Pan-style SPADE and Leal-style manifold-clustering baselines.

Both methods operate on the same saved HBN subject representations. The Pan
adaptation uses covariance/log-covariance features, MDS, a small autoencoder,
and KMeans at K=5. The Leal adaptation uses PCA95, three-dimensional UMAP,
multiple candidate clusterers, and selects the valid partition with the
largest Dunn index. The current native Du-style DBSCAN analysis is provided in
`19_native_dbscan.py` and is intentionally not duplicated here.
"""
from __future__ import annotations
import argparse, json, platform, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan, umap, torch
import torch.nn as nn

SEED=42
PHENOTYPE_COLUMNS=["p_factor","attention","internalizing","externalizing"]
CLINICAL_COLUMNS=["age","p_factor","attention","internalizing","externalizing","ehq_total"]

def parse_args():
    p=argparse.ArgumentParser(description="Pan- and Leal-style adapted EEG baselines")
    p.add_argument("--eeg-features",type=Path,required=True); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    return p.parse_args()

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def scale(x): return StandardScaler().fit_transform(np.nan_to_num(np.asarray(x,float)))
def bh_fdr(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]; n=len(p); out=np.empty(n); run=1.0
    for i in range(n-1,-1,-1): run=min(run,ranked[i]*n/(i+1)); out[order[i]]=min(run,1.0)
    return out

def dunn_index(x,labels):
    x=scale(x); labels=np.asarray(labels); valid=labels!=-1; x=x[valid]; labels=labels[valid]; u=np.unique(labels); D=pairwise_distances(x)
    intra=max(float(D[np.ix_(np.where(labels==c)[0],np.where(labels==c)[0])].max()) for c in u)
    inter=min(float(D[np.ix_(np.where(labels==a)[0],np.where(labels==b)[0])].min()) for i,a in enumerate(u) for b in u[i+1:])
    return float(inter/intra) if intra>0 else float("nan")

def metrics(x,labels):
    x=scale(x); labels=np.asarray(labels,int); valid=labels!=-1; xv=x[valid]; lv=labels[valid]; u,c=np.unique(lv,return_counts=True); noise=int((~valid).sum())
    row={"K":int(len(u)),"assigned_n":int(valid.sum()),"coverage_percent":100*float(valid.mean()),"noise_n":noise,"minimum_cluster_size":int(c.min()),"maximum_cluster_size":int(c.max()),"balance_ratio":float(c.max()/c.min())}
    row.update({"silhouette":float(silhouette_score(xv,lv)),"davies_bouldin":float(davies_bouldin_score(xv,lv)),"calinski_harabasz":float(calinski_harabasz_score(xv,lv)),"dunn":dunn_index(xv,lv)})
    return row

def clinical(meta,labels):
    rows=[]; labels=np.asarray(labels); keep_labels=labels!=-1
    for col in CLINICAL_COLUMNS:
        y=pd.to_numeric(meta[col],errors="coerce").to_numpy(float); valid=np.isfinite(y)&keep_labels; yv=y[valid]; lv=labels[valid]; groups=[yv[lv==c] for c in np.unique(lv) if np.sum(lv==c)>1]
        f,p=stats.f_oneway(*groups); grand=float(np.mean(yv)); total=float(np.sum((yv-grand)**2)); between=float(sum(len(g)*(float(np.mean(g))-grand)**2 for g in groups))
        rows.append({"variable":col,"anova_F":float(f),"anova_p":float(p),"eta_squared":between/total if total>0 else float("nan")})
    df=pd.DataFrame(rows); df["anova_p_fdr"]=bh_fdr(df["anova_p"]); df["significant_fdr_0p05"]=df["anova_p_fdr"]<.05; return df

def covariance_features(x,pseudo_channels=8):
    x=scale(x); n,d=x.shape; usable=(d//pseudo_channels)*pseudo_channels
    if usable<pseudo_channels*2: raise ValueError("Feature dimension is too small for covariance construction")
    x=x[:,:usable]; t=usable//pseudo_channels; diag=[]; logcov=[]
    for row in x:
        cov=np.cov(row.reshape(pseudo_channels,t))+1e-4*np.eye(pseudo_channels); diag.append(np.log(np.diag(cov)+1e-8)); vals,vecs=np.linalg.eigh(cov); vals=np.maximum(vals,1e-8); lc=vecs@np.diag(np.log(vals))@vecs.T; logcov.append(lc[np.triu_indices_from(lc)])
    diag=scale(diag); logcov=scale(logcov); D=pairwise_distances(logcov)
    mds=MDS(n_components=min(8,n-1),dissimilarity="precomputed",random_state=SEED,normalized_stress="auto")
    return scale(np.hstack([diag,logcov,scale(mds.fit_transform(D))]))

class SmallAE(nn.Module):
    def __init__(self,d,z=10):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,z)); self.decoder=nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self,x): z=self.encoder(x); return self.decoder(z),z

def pan_baseline(x):
    set_seed(); cov=covariance_features(x,8).astype(np.float32); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=SmallAE(cov.shape[1],10).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5); xt=torch.tensor(cov,device=device)
    for _ in range(300): opt.zero_grad(); xr,z=model(xt); loss=nn.functional.mse_loss(xr,xt); loss.backward(); opt.step()
    model.eval();
    with torch.no_grad(): z=model.encoder(xt).cpu().numpy()
    z=scale(z); labels=KMeans(n_clusters=5,random_state=SEED,n_init=50).fit_predict(z); return z,labels,{"pseudo_channels":8,"latent_dim":10,"epochs":300,"clustering":"KMeans","K":5}

def leal_baseline(x):
    xp=PCA(n_components=.95,random_state=SEED).fit_transform(scale(x)); xu=umap.UMAP(n_components=3,n_neighbors=30,min_dist=.1,metric="euclidean",random_state=SEED).fit_transform(xp); candidates=[]
    for k in range(2,9):
        candidates.append(("KMeans",k,KMeans(n_clusters=k,random_state=SEED,n_init=50).fit_predict(scale(xu))))
        candidates.append(("Agglomerative-Ward",k,AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(scale(xu))))
        candidates.append(("GMM",k,GaussianMixture(n_components=k,covariance_type="full",random_state=SEED,n_init=10,reg_covar=1e-5).fit_predict(scale(xu))))
    for mcs in [10,15,20,25,30,40,50]: candidates.append(("HDBSCAN",mcs,hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=max(3,mcs//3)).fit_predict(scale(xu))))
    valid=[]
    for method,param,labels in candidates:
        u=np.unique(labels[labels!=-1]); noise=100*float(np.mean(labels==-1))
        if len(u)<2 or noise>40: continue
        row=metrics(xu,labels)
        if np.isfinite(row["dunn"]): valid.append((row["dunn"],row["silhouette"],method,param,labels))
    if not valid: raise RuntimeError("No valid Leal-style candidate met the pre-specified coverage rules")
    _,_,method,param,labels=max(valid,key=lambda v:(v[0],v[1])); return xu,labels,{"selected_method":method,"selected_parameter":param,"selection_metric":"maximum Dunn index","pca_variance":.95,"umap_neighbors":30}

def main():
    a=parse_args(); out=a.out.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); eeg=np.load(a.eeg_features.expanduser().resolve()); meta=pd.read_csv(a.metadata.expanduser().resolve())
    if eeg.shape[0]!=len(meta) or len(meta)!=430: raise ValueError("Expected aligned 430-subject EEG and metadata inputs")
    ph=meta[PHENOTYPE_COLUMNS].apply(pd.to_numeric,errors="coerce"); ph=ph.fillna(ph.median(numeric_only=True)).to_numpy(float); fused=np.hstack([scale(eeg),scale(ph)])
    methods=[("Pan-style SPADE (Riemannian-AE)",*pan_baseline(eeg)),("Leal-style UMAP multi-clustering",*leal_baseline(fused))]
    summaries=[]; clinical_parts=[]; subject_col="subject" if "subject" in meta else ("subject_id" if "subject_id" in meta else None)
    for name,x,labels,params in methods:
        c=clinical(meta,labels); row={"method":name,**metrics(x,labels),"mean_eta_squared":float(c["eta_squared"].mean()),"fdr_significant_count":int(c["significant_fdr_0p05"].sum()),"fdr_family_size":len(c),"parameters":json.dumps(params,sort_keys=True)}; summaries.append(row); c.insert(0,"method",name); clinical_parts.append(c)
        safe="pan" if name.startswith("Pan") else "leal"; pd.DataFrame({"subject":meta[subject_col].astype(str) if subject_col else np.arange(len(meta)),"cluster":labels}).to_csv(out/f"{safe}_labels.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"pan_leal_baseline_summary.csv",index=False); pd.concat(clinical_parts,ignore_index=True).to_csv(out/"pan_leal_clinical_stats.csv",index=False)
    (out/"run_manifest.json").write_text(json.dumps({"eeg_features":a.eeg_features.name,"metadata":a.metadata.name,"seed":SEED,"python":sys.version,"platform":platform.platform()},indent=2)); print(pd.DataFrame(summaries).to_string(index=False)); print("Saved outputs to:",out); return 0
if __name__=="__main__": raise SystemExit(main())
