"""
qn_tradeoff.py
==============
Clean, LEAK-FREE ablation to settle the OOD story.

Two axes:
  - prior source : DRST-filtered (782, lab-like)  vs  full impregnation literature
  - labels       : raw  vs  quantile-normalised (QN, mapped to the lab yield scale)

For every config we report:
  - in-distribution 5-fold CV RMSE (asymmetric: literature only in training, lab-only validation)
  - LEAK-FREE OOD RMSE on the 2,139 non-impregnation literature rows, with the Stage-1 prior
    trained on literature that EXCLUDES those OOD rows.

We also reproduce the OLD leaky number (prior trained on ALL literature incl. the OOD rows and
their true labels) to show why the deck's 3.60 was optimistic.

Outputs: qn_tradeoff.json, fig_qn_tradeoff.png
"""
import json, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata
import xgboost as xgb, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---- data (identical to ocm_analysis.ipynb / feedback_experiments.py) ----
TARGET = 'Y(C2), %'
df = pd.read_csv('OCM_lab_data_and_literature_datal.csv')
do = df[df.year == 2025].reset_index(drop=True)
dl = df[df.year <= 2019].reset_index(drop=True)
EL = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
le = LabelEncoder().fit(df.Preparation)
do['prep_enc'] = le.transform(do.Preparation); dl['prep_enc'] = le.transform(dl.Preparation)
F = ['Temperature_C', 'prep_enc'] + EL
X_ours = do[F].values.astype(float); y_ours = do[TARGET].values
X_lit = dl[F].values.astype(float);  y_lit = dl[TARGET].values
scaler = StandardScaler(); X_ours_sc = scaler.fit_transform(X_ours); X_lit_sc = scaler.transform(X_lit)

ood_mask = (dl['Preparation'] != 'Impregnation').values     # OOD = non-impregnation literature
impreg_mask = ~ood_mask
X_ood, y_ood = X_lit_sc[ood_mask], y_lit[ood_mask]
log(f"lab={X_ours.shape} lit={X_lit.shape}  OOD(non-impreg)={ood_mask.sum()}  impreg={impreg_mask.sum()}")

def xgb_params(): return dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, verbosity=0, n_jobs=-1)
def lgb_params(): return dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)
def qn(ys, yt):
    q = np.clip(rankdata(ys, method='average') / (len(ys) + 1), 0.01, 0.99)
    return np.quantile(yt, q)

# ---- DRST filter (LogReg C=0.5) ----
rng = np.random.default_rng(42); sub = rng.choice(len(X_ours_sc), 10000, replace=False)
clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1).fit(
    np.vstack([X_ours_sc[sub], X_lit_sc]), np.r_[np.ones(len(sub)), np.zeros(len(X_lit_sc))])
p_ours_lit = clf.predict_proba(X_lit_sc)[:, 1]
mask782 = p_ours_lit >= 0.30
log(f"DRST keeps {mask782.sum()} rows; of those {(mask782 & ood_mask).sum()} are non-impregnation (excluded for leak-free OOD)")

# ---- in-distribution CV (asymmetric 5-fold) ----
def cv_indist(prior_mask, bias_correct):
    Xp, yp = X_lit_sc[prior_mask], y_lit[prior_mask]
    rmses = []
    for tr, va in KFold(5, shuffle=True, random_state=42).split(X_ours_sc):
        y_pre = qn(yp, y_ours[tr]) if bias_correct else yp
        pre = xgb.XGBRegressor(**xgb_params()).fit(
            np.vstack([Xp, X_ours_sc[tr]]), np.concatenate([y_pre, y_ours[tr]]))
        fin = lgb.LGBMRegressor(**lgb_params()).fit(
            np.hstack([X_ours_sc[tr], pre.predict(X_ours_sc[tr]).reshape(-1, 1)]), y_ours[tr])
        p = fin.predict(np.hstack([X_ours_sc[va], pre.predict(X_ours_sc[va]).reshape(-1, 1)]))
        rmses.append(np.sqrt(mean_squared_error(y_ours[va], p)))
    return float(np.mean(rmses))

# ---- OOD RMSE (fit on all lab + prior; prior source = prior_mask) ----
def ood_eval(prior_mask, bias_correct):
    Xp, yp = X_lit_sc[prior_mask], y_lit[prior_mask]
    y_pre = qn(yp, y_ours) if bias_correct else yp
    pre = xgb.XGBRegressor(**xgb_params()).fit(
        np.vstack([Xp, X_ours_sc]), np.concatenate([y_pre, y_ours]))
    fin = lgb.LGBMRegressor(**lgb_params()).fit(
        np.hstack([X_ours_sc, pre.predict(X_ours_sc).reshape(-1, 1)]), y_ours)
    return float(np.sqrt(mean_squared_error(
        y_ood, fin.predict(np.hstack([X_ood, pre.predict(X_ood).reshape(-1, 1)])))))

# baseline (no transfer)
base_cv_rmses = []
for tr, va in KFold(5, shuffle=True, random_state=42).split(X_ours_sc):
    m = lgb.LGBMRegressor(**lgb_params()).fit(X_ours_sc[tr], y_ours[tr])
    base_cv_rmses.append(np.sqrt(mean_squared_error(y_ours[va], m.predict(X_ours_sc[va]))))
base_cv = float(np.mean(base_cv_rmses))
mb = lgb.LGBMRegressor(**lgb_params()).fit(X_ours_sc, y_ours)
base_ood = float(np.sqrt(mean_squared_error(y_ood, mb.predict(X_ood))))
log(f"baseline: CV={base_cv:.4f}  OOD={base_ood:.4f}")

# leak-free prior sources (exclude OOD rows)
filt_leakfree = mask782 & ~ood_mask
impreg_leakfree = impreg_mask                      # all impregnation lit = excludes OOD by definition

configs = [
    ("DRST-filtered + raw labels",  filt_leakfree,   False),
    ("DRST-filtered + QN (=PFT)",   filt_leakfree,   True),
    ("Full impreg-lit + raw labels", impreg_leakfree, False),
    ("Full impreg-lit + QN",        impreg_leakfree, True),
]
rows = []
for name, mask, bc in configs:
    cv = cv_indist(mask, bc); od = ood_eval(mask, bc)
    rows.append(dict(config=name, n_prior=int(mask.sum()), cv_rmse=cv, ood_rmse=od))
    log(f"{name:32s}  n={int(mask.sum()):4d}  CV={cv:.4f}  OOD(leak-free)={od:.4f}")

# reproduce the OLD LEAKY 3.60 (prior trained on ALL lit incl. OOD rows, raw labels)
all_mask = np.ones(len(X_lit_sc), dtype=bool)
leaky_ood = ood_eval(all_mask, bias_correct=False)
leaky_cv  = cv_indist(all_mask, bias_correct=False)
log(f"LEAKY repro (all-lit incl. OOD, raw): CV={leaky_cv:.4f}  OOD={leaky_ood:.4f}  <-- old deck 3.60")

RESULTS = dict(baseline=dict(cv_rmse=base_cv, ood_rmse=base_ood),
               configs=rows,
               leaky_repro=dict(cv_rmse=leaky_cv, ood_rmse=leaky_ood, note="prior trained on OOD rows -> optimistic"))
json.dump(RESULTS, open('qn_tradeoff.json', 'w'), indent=2)

# ---- figure: in-dist CV vs leak-free OOD ----
labels = [r['config'] for r in rows]
cvs = [r['cv_rmse'] for r in rows]; ods = [r['ood_rmse'] for r in rows]
colors = ['#8aa0b8', '#1f4e79', '#f0b27a', '#e67e22']
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
x = np.arange(len(labels))
ax[0].bar(x, cvs, color=colors); ax[0].axhline(base_cv, ls='--', c='gray', label=f'baseline {base_cv:.3f}')
ax[0].set_title('In-distribution 5-fold CV RMSE (lab)  ↓ better'); ax[0].set_ylim(1.85, 2.20)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=8); ax[0].legend(fontsize=8)
for i, v in enumerate(cvs): ax[0].text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=8)
ax[1].bar(x, ods, color=colors); ax[1].axhline(base_ood, ls='--', c='gray', label=f'baseline {base_ood:.2f}')
ax[1].axhline(leaky_ood, ls=':', c='crimson', label=f'old leaky repro {leaky_ood:.2f}')
ax[1].set_title('Leak-free OOD RMSE (non-impreg lit)  ↓ better')
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=25, ha='right', fontsize=8); ax[1].legend(fontsize=8)
for i, v in enumerate(ods): ax[1].text(i, v+0.05, f'{v:.2f}', ha='center', fontsize=8)
fig.suptitle('Quantile normalisation trades OOD extrapolation for in-distribution accuracy', fontweight='bold')
plt.tight_layout(); plt.savefig('fig_qn_tradeoff.png', dpi=130); plt.close()
log("DONE -> qn_tradeoff.json, fig_qn_tradeoff.png")
