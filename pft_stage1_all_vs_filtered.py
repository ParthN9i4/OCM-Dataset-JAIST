"""
pft_stage1_all_vs_filtered.py
Compare PFT where Stage 1 uses ALL 3,852 literature rows vs the DRST-filtered 782 rows.
Same Stage 2, same asymmetric 5-fold CV, across 10 seeds. Reports RMSE (and MAE, R2) side by side.
Reuses the exact pipeline from the methodology notebook.
"""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata, ttest_rel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb, xgboost as xgb
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---- data (identical to ocm_methodology.ipynb) ----
TARGET='Y(C2), %'
df=pd.read_csv('OCM_lab_data_and_literature_datal.csv')
dl_lab=df[df.year==2025].reset_index(drop=True); dl_lit=df[df.year<=2019].reset_index(drop=True)
EL=[c for c in df.columns if c not in ['Preparation','Temperature_C',TARGET,'year']]
le=LabelEncoder().fit(df.Preparation)
dl_lab['prep_enc']=le.transform(dl_lab.Preparation); dl_lit['prep_enc']=le.transform(dl_lit.Preparation)
F=['Temperature_C','prep_enc']+EL
X_lab=dl_lab[F].values.astype(float); y_lab=dl_lab[TARGET].values
X_lit=dl_lit[F].values.astype(float); y_lit=dl_lit[TARGET].values
sc=StandardScaler(); X_lab_sc=sc.fit_transform(X_lab); X_lit_sc=sc.transform(X_lit)
log(f"lab={X_lab.shape} lit={X_lit.shape}")

def lgb_params(seed=42): return dict(n_estimators=500,learning_rate=0.05,num_leaves=63,max_depth=7,
    min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,
    random_state=seed,n_jobs=-1,verbosity=-1)
def xgb_params(seed=42): return dict(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,
    colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=seed,verbosity=0,n_jobs=-1)
def quantile_normalize_y(ys,yt):
    q=np.clip(rankdata(ys,method='average')/(len(ys)+1),0.01,0.99); return np.quantile(yt,q)

# DRST domain classifier (same as notebook)
rng=np.random.default_rng(42); sub=rng.choice(len(X_lab_sc),min(10000,len(X_lab_sc)),replace=False)
clf=LogisticRegression(C=0.5,max_iter=1000,random_state=42,n_jobs=-1).fit(
    np.vstack([X_lab_sc[sub],X_lit_sc]), np.r_[np.ones(len(sub)),np.zeros(len(X_lit_sc))])
p_lab_lit=clf.predict_proba(X_lit_sc)[:,1]
mask=p_lab_lit>=0.30
log(f"DRST keeps {int(mask.sum())}/{len(mask)} = {100*mask.mean():.1f}% of literature")

def metrics(yt,yp): return (np.sqrt(mean_squared_error(yt,yp)), mean_absolute_error(yt,yp), r2_score(yt,yp))

def baseline(seed=42,cv=5):
    kf=KFold(cv,shuffle=True,random_state=seed); r=[]
    for tr,va in kf.split(X_lab_sc):
        m=lgb.LGBMRegressor(**lgb_params(seed)).fit(X_lab_sc[tr],y_lab[tr])
        r.append(metrics(y_lab[va],m.predict(X_lab_sc[va])))
    return np.mean(r,0)

def pft(Xlp,ylp,seed=42,cv=5):
    kf=KFold(cv,shuffle=True,random_state=seed); r=[]
    for tr,va in kf.split(X_lab_sc):
        ypre=quantile_normalize_y(ylp,y_lab[tr])
        pre=xgb.XGBRegressor(**xgb_params(seed)).fit(np.vstack([Xlp,X_lab_sc[tr]]),np.concatenate([ypre,y_lab[tr]]))
        ptr=pre.predict(X_lab_sc[tr]).reshape(-1,1); pva=pre.predict(X_lab_sc[va]).reshape(-1,1)
        fin=lgb.LGBMRegressor(**lgb_params(seed)).fit(np.hstack([X_lab_sc[tr],ptr]),y_lab[tr])
        r.append(metrics(y_lab[va],fin.predict(np.hstack([X_lab_sc[va],pva]))))
    return np.mean(r,0)

SEEDS=[0,1,2,7,13,21,42,77,123,2025]
rows={'baseline':[], 'pft_filtered_782':[], 'pft_all_3852':[]}
for s in SEEDS:
    b=baseline(s); f=pft(X_lit_sc[mask],y_lit[mask],s); a=pft(X_lit_sc,y_lit,s)
    rows['baseline'].append(b); rows['pft_filtered_782'].append(f); rows['pft_all_3852'].append(a)
    log(f"seed={s:5d}  base RMSE={b[0]:.3f}  PFT-filtered={f[0]:.3f}  PFT-all={a[0]:.3f}")

print("\n================ SUMMARY (mean over 10 seeds) ================")
print(f"{'config':22s} {'RMSE':>16s} {'MAE':>14s} {'R2':>10s}")
arr={}
for k,v in rows.items():
    v=np.array(v); arr[k]=v
    print(f"{k:22s} {v[:,0].mean():.3f} +/- {v[:,0].std(ddof=1):.3f}   {v[:,1].mean():.3f}+/-{v[:,1].std(ddof=1):.3f}   {v[:,2].mean():.3f}")
bf=arr['pft_filtered_782'][:,0]; ba=arr['pft_all_3852'][:,0]
print(f"\nfiltered vs all: paired t p = {ttest_rel(bf,ba)[1]:.3f}  (mean diff all-filtered = {ba.mean()-bf.mean():+.4f} RMSE)")
print(f"PFT-all beats baseline in {(ba<arr['baseline'][:,0]).sum()}/10 seeds; PFT-filtered in {(bf<arr['baseline'][:,0]).sum()}/10")
