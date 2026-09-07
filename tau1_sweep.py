"""
*** SUPERSEDED — DO NOT QUOTE THESE NUMBERS WITHOUT THIS CAVEAT ***
Pre-refactor script, does not import ocm_eval.py. Uses a SINGLE GLOBAL StandardScaler and a SINGLE
GLOBAL DRST domain classifier fit once on the whole lab set before the row-level KFold loop even
starts (both refit per-fold in ocm_eval.py's leak-proof reference implementation), plus a row-level
KFold split (a catalyst can appear in both train and validation). This sweeps the DRST threshold tau
to justify tau=0.30 as a diagnostic, not to report a headline number. Its own figure,
fig_pft_tau1_sweep.png, is cited in ocm_ch9_ch10_script.md and ocm_feedback_responses.md and should
carry this caveat wherever it appears. See ocm_eval.py's PROTOCOL RULES for the leak-proof standard.
"""
import json, warnings, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata
import xgboost as xgb, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
df=pd.read_csv('OCM_lab_data_and_literature_datal.csv'); TARGET='Y(C2), %'
do=df[df.year==2025].reset_index(drop=True); dl=df[df.year<=2019].reset_index(drop=True)
EL=[c for c in df.columns if c not in ['Preparation','Temperature_C',TARGET,'year']]
le=LabelEncoder().fit(df.Preparation); do['prep_enc']=le.transform(do.Preparation); dl['prep_enc']=le.transform(dl.Preparation)
F=['Temperature_C','prep_enc']+EL
Xo=do[F].values.astype(float); yo=do[TARGET].values; Xl=dl[F].values.astype(float); yl=dl[TARGET].values
sc=StandardScaler(); Xo=sc.fit_transform(Xo); Xl=sc.transform(Xl)
xp=lambda:dict(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbosity=0,n_jobs=-1)
lp=lambda:dict(n_estimators=500,learning_rate=0.05,num_leaves=63,max_depth=7,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,n_jobs=-1,verbosity=-1)
def qn(ys,yt): q=np.clip(rankdata(ys,method='average')/(len(ys)+1),0.01,0.99); return np.quantile(yt,q)
rng=np.random.default_rng(42); sub=rng.choice(len(Xo),10000,replace=False)
clf=LogisticRegression(C=0.5,max_iter=1000,random_state=42,n_jobs=-1).fit(np.vstack([Xo[sub],Xl]),np.r_[np.ones(len(sub)),np.zeros(len(Xl))])
p=clf.predict_proba(Xl)[:,1]
def pft(mask):
    Xp,yp_=Xl[mask],yl[mask]; rm=[]
    for tr,va in KFold(5,shuffle=True,random_state=42).split(Xo):
        ypre=qn(yp_,yo[tr]); pre=xgb.XGBRegressor(**xp()).fit(np.vstack([Xp,Xo[tr]]),np.r_[ypre,yo[tr]])
        fin=lgb.LGBMRegressor(**lp()).fit(np.hstack([Xo[tr],pre.predict(Xo[tr]).reshape(-1,1)]),yo[tr])
        rm.append(np.sqrt(mean_squared_error(yo[va],fin.predict(np.hstack([Xo[va],pre.predict(Xo[va]).reshape(-1,1)])))))
    return float(np.mean(rm))
res=[]
for t1 in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.65,0.80]:
    m=p>=t1; n=int(m.sum());
    if n<20: continue
    r=pft(m); res.append((t1,n,r)); log(f"tau1={t1:.2f} n={n} PFT_RMSE={r:.4f}")
json.dump(res,open('tau1_sweep.json','w'))
best=min(res,key=lambda x:x[2])
fig,ax=plt.subplots(figsize=(7.5,4.8))
ax.plot([x[0] for x in res],[x[2] for x in res],'s-',color='#1f4e79',lw=2,label='Two-stage PFT CV RMSE')
ax.axhline(2.1207,color='gray',ls='--',label='Baseline = 2.121')
ax.scatter([best[0]],[best[2]],s=150,facecolors='none',edgecolors='crimson',lw=2.5,zorder=5,label=f'Best τ₁={best[0]:.2f} → {best[2]:.3f}')
ax.axvline(0.30,color='orange',ls=':',label='τ₁=0.30 (deck)')
ax.set_xlabel('DRST Stage-1 filter threshold τ₁'); ax.set_ylabel('5-fold CV RMSE (lab)')
ax.set_title('Q5: Two-stage PFT — why τ₁=0.30'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_pft_tau1_sweep.png',dpi=130)
log(f"best tau1={best}")
