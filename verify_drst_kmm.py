"""
verify_drst_kmm.py — independent check of the single-stage DRST & KMM numbers.
Reuses the exact pipeline (data load, evaluate_cv_ours, DRST classifier) that both
notebooks use, to settle: does single-stage DRST τ=0.30 = 2.019 or ~2.18? and KMM = 2.035 or ~2.26?
No tracked artifacts are modified.
"""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

TARGET='Y(C2), %'
df=pd.read_csv('OCM_lab_data_and_literature_datal.csv')
do=df[df.year==2025].reset_index(drop=True); dl=df[df.year<=2019].reset_index(drop=True)
EL=[c for c in df.columns if c not in ['Preparation','Temperature_C',TARGET,'year']]
le=LabelEncoder().fit(df.Preparation); do['prep_enc']=le.transform(do.Preparation); dl['prep_enc']=le.transform(dl.Preparation)
F=['Temperature_C','prep_enc']+EL
X_ours=do[F].values.astype(float); y_ours=do[TARGET].values
X_lit=dl[F].values.astype(float);  y_lit=dl[TARGET].values
scaler=StandardScaler(); X_ours_sc=scaler.fit_transform(X_ours); X_lit_sc=scaler.transform(X_lit)
log(f"lab={X_ours.shape} lit={X_lit.shape}")

def lgb_params(): return dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=-1)
def qn(ys,yt): q=np.clip(rankdata(ys,method='average')/(len(ys)+1),0.01,0.99); return np.quantile(yt,q)

def evaluate_cv_ours(X_train_extra=None, y_train_extra=None, sample_weight_extra=None, cv=5):
    kf=KFold(n_splits=cv, shuffle=True, random_state=42); rl=[]
    for tr,va in kf.split(X_ours_sc):
        if X_train_extra is not None:
            Xtr=np.vstack([X_ours_sc[tr],X_train_extra]); ytr=np.concatenate([y_ours[tr],y_train_extra])
            sw=np.concatenate([np.ones(len(tr)),sample_weight_extra]) if sample_weight_extra is not None else None
        else:
            Xtr,ytr,sw=X_ours_sc[tr],y_ours[tr],None
        m=lgb.LGBMRegressor(**lgb_params()); m.fit(Xtr,ytr,sample_weight=sw)
        rl.append(np.sqrt(mean_squared_error(y_ours[va],m.predict(X_ours_sc[va]))))
    return float(np.mean(rl))

# DRST classifier (identical to both notebooks)
rng=np.random.default_rng(42); sub=rng.choice(len(X_ours_sc),min(10000,len(X_ours_sc)),replace=False)
clf=LogisticRegression(C=0.5,max_iter=1000,random_state=42,n_jobs=-1).fit(
    np.vstack([X_ours_sc[sub],X_lit_sc]), np.r_[np.ones(len(sub)),np.zeros(len(X_lit_sc))])
p_ours_lit=clf.predict_proba(X_lit_sc)[:,1]

base=evaluate_cv_ours()
log(f"BASELINE = {base:.4f}")
print("\n===== SINGLE-STAGE DRST (raw literature labels) =====")
print(f"{'tau':>6} {'n':>6} {'RMSE':>8}  {'vs base':>8}")
for tau in [0.05,0.20,0.30,0.40,0.85]:
    m=p_ours_lit>=tau; r=evaluate_cv_ours(X_lit_sc[m], y_lit[m])
    print(f"{tau:6.2f} {int(m.sum()):6d} {r:8.4f}  {r-base:+8.4f}")

print("\n===== PROBE: does quantile-normalising the DRST labels reproduce ~2.019? =====")
m=p_ours_lit>=0.30
r_qn=evaluate_cv_ours(X_lit_sc[m], qn(y_lit[m], y_ours))
print(f"DRST tau=0.30, QN labels : RMSE={r_qn:.4f}  (vs base {r_qn-base:+.4f})   [walkthrough claims 2.019]")

print("\n===== KMM (RBF, B=5, soft weights on all 3,852) =====")
def rbf(X,Y,s): return np.exp(-cdist(X,Y,'sqeuclidean')/(2*s**2))
def est_sigma(Xs,Xt,n=2000):
    r=np.random.default_rng(0); a=Xs[r.choice(len(Xs),min(n,len(Xs)),replace=False)]; b=Xt[r.choice(len(Xt),min(n,len(Xt)),replace=False)]
    d=cdist(np.vstack([a,b]),np.vstack([a,b])); return float(np.median(d[d>0]))
def kmm_weights(Xs,Xt,B=5.0,n_tsub=5000):
    r=np.random.default_rng(42); Xt=Xt[r.choice(len(Xt),min(n_tsub,len(Xt)),replace=False)]
    n_s,n_t=len(Xs),len(Xt); sig=est_sigma(Xs,Xt); eps=(np.sqrt(n_s)-1)/np.sqrt(n_s)
    Kss=rbf(Xs,Xs,sig); kap=(n_s/n_t)*rbf(Xs,Xt,sig).sum(1)
    def obj(w): return 0.5*w@Kss@w-kap@w+1e4*max(0.,abs(w.sum()/n_s-1.)-eps)**2
    def grad(w):
        g=Kss@w-kap; d=w.sum()/n_s-1.
        if abs(d)>eps: g+=1e4*2.*d/n_s*np.ones(n_s)
        return g
    res=minimize(obj,np.ones(n_s),jac=grad,method='L-BFGS-B',bounds=[(0.,B)]*n_s,options={'maxiter':1000,'ftol':1e-9,'gtol':1e-6})
    return np.clip(res.x,0.,B)
w=kmm_weights(X_lit_sc,X_ours_sc)
r_kmm=evaluate_cv_ours(X_lit_sc, y_lit, sample_weight_extra=w)
print(f"KMM weighted (raw labels): RMSE={r_kmm:.4f}  (vs base {r_kmm-base:+.4f})   [walkthrough claims 2.035]")

print("\n===== SUMMARY =====")
print(f"baseline................... {base:.4f}")
print(f"DRST tau=0.30 (raw)........ {evaluate_cv_ours(X_lit_sc[p_ours_lit>=0.30], y_lit[p_ours_lit>=0.30]):.4f}   walkthrough: 2.019")
print(f"DRST tau=0.30 (QN)......... {r_qn:.4f}")
print(f"KMM (raw).................. {r_kmm:.4f}   walkthrough: 2.035")
