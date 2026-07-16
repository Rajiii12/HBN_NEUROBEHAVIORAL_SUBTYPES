#!/usr/bin/env python3
"""Jiman-style PCA95–UMAP–KMeans baseline on saved HBN subject features.

The implementation follows the archived benchmark settings used for the
adapted EEG comparison: separately standardized EEG and phenotype blocks,
PCA retaining at least 95% variance, two-dimensional UMAP, and KMeans at K=5.
Clinical variables are evaluated only after the partition has been created.
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import umap

SEED=42
K=5
PHENOTYPE_COLUMNS=["internalizing","attention","externalizing","p_factor"]
CLINICAL_COLUMNS=["age","p_factor","attention","internalizing","externalizing","ehq_total"]

def parse_args():
    p=argparse.ArgumentParser(description="Jiman-style PCA95-UMAP-KMeans baseline")
    p.add_argument("--eeg-features", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()

def bh_fdr(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]; n=len(p); adj=np.empty(n); running=1.0
    for i in range(n-1,-1,-1):
        running=min(running, ranked[i]*n/(i+1)); adj[order[i]]=min(running,1.0)
    return adj

def dunn_index(x, labels):
    labels=np.asarray(labels); unique=np.unique(labels); D=pairwise_distances(x)
    max_intra=max(float(D[np.ix_(np.where(labels==c)[0],np.where(labels==c)[0])].max()) for c in unique)
    min_inter=min(float(cdist(x[labels==a],x[labels==b]).min()) for i,a in enumerate(unique) for b in unique[i+1:])
    return float(min_inter/max_intra) if max_intra>0 else float("nan")

def clinical_stats(meta, labels):
    rows=[]
    for col in CLINICAL_COLUMNS:
        if col not in meta: raise ValueError(f"Missing clinical column: {col}")
        y=pd.to_numeric(meta[col],errors="coerce").to_numpy(float); valid=np.isfinite(y); y=y[valid]; lab=labels[valid]
        groups=[y[lab==c] for c in np.unique(lab) if np.sum(lab==c)>1]
        f,p=stats.f_oneway(*groups); grand=float(np.mean(y)); total=float(np.sum((y-grand)**2)); between=float(sum(len(g)*(float(np.mean(g))-grand)**2 for g in groups))
        rows.append({"variable":col,"anova_F":float(f),"anova_p":float(p),"eta_squared":between/total if total>0 else float("nan")})
    df=pd.DataFrame(rows); df["anova_p_fdr"]=bh_fdr(df["anova_p"].to_numpy()); df["significant_fdr_0p05"]=df["anova_p_fdr"]<.05
    return df

def main():
    a=parse_args(); out=a.out.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    eeg=np.load(a.eeg_features.expanduser().resolve()); meta=pd.read_csv(a.metadata.expanduser().resolve())
    if eeg.ndim!=2 or eeg.shape[0]!=len(meta): raise ValueError(f"EEG shape {eeg.shape} does not align with {len(meta)} metadata rows")
    if len(meta)!=430: raise ValueError(f"Expected 430 subjects, found {len(meta)}")
    ph=meta[PHENOTYPE_COLUMNS].apply(pd.to_numeric,errors="coerce"); ph=ph.fillna(ph.median(numeric_only=True)).to_numpy(float)
    x=np.hstack([StandardScaler().fit_transform(eeg),StandardScaler().fit_transform(ph)])
    pca=PCA(n_components=.95,random_state=SEED); xp=pca.fit_transform(StandardScaler().fit_transform(x))
    reducer=umap.UMAP(n_components=2,n_neighbors=15,min_dist=.1,metric="euclidean",random_state=SEED)
    xu=reducer.fit_transform(xp)
    labels=KMeans(n_clusters=K,n_init=100,random_state=SEED).fit_predict(xu)
    counts=pd.Series(labels).value_counts().sort_index(); clinical=clinical_stats(meta,labels)
    summary=pd.DataFrame([{
        "method":"Jiman-style PCA95-UMAP-KMeans","K":K,"coverage_percent":100.0,
        "silhouette":float(silhouette_score(xu,labels)),"davies_bouldin":float(davies_bouldin_score(xu,labels)),
        "calinski_harabasz":float(calinski_harabasz_score(xu,labels)),"dunn":dunn_index(xu,labels),
        "minimum_cluster_size":int(counts.min()),"maximum_cluster_size":int(counts.max()),
        "balance_ratio":float(counts.max()/counts.min()),"mean_eta_squared":float(clinical["eta_squared"].mean()),
        "fdr_significant_count":int(clinical["significant_fdr_0p05"].sum()),"fdr_family_size":len(clinical),
        "pca_components":int(xp.shape[1]),"pca_variance_retained":float(pca.explained_variance_ratio_.sum())}])
    subject_col="subject" if "subject" in meta.columns else ("subject_id" if "subject_id" in meta.columns else None)
    assignments=pd.DataFrame({"subject":meta[subject_col].astype(str) if subject_col else np.arange(len(meta)),"cluster":labels,"UMAP1":xu[:,0],"UMAP2":xu[:,1]})
    summary.to_csv(out/"jiman_baseline_summary.csv",index=False); clinical.to_csv(out/"jiman_clinical_stats.csv",index=False); assignments.to_csv(out/"jiman_labels.csv",index=False)
    pd.DataFrame({"component":np.arange(1,len(pca.explained_variance_ratio_)+1),"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_explained_variance":np.cumsum(pca.explained_variance_ratio_)}).to_csv(out/"jiman_pca_variance.csv",index=False)
    (out/"run_manifest.json").write_text(json.dumps({"eeg_features":a.eeg_features.name,"metadata":a.metadata.name,"seed":SEED,"K":K,"umap":{"n_neighbors":15,"min_dist":.1,"n_components":2,"metric":"euclidean"},"kmeans_n_init":100,"python":sys.version,"platform":platform.platform()},indent=2))
    print(summary.to_string(index=False)); print("Saved outputs to:",out)
    return 0
if __name__=="__main__": raise SystemExit(main())
