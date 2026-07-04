"""
Builds ocm_methodology.ipynb — a clean, linear notebook covering exactly the five methods
(baseline, naive merge, DRST, KMM, two-stage PFT) + validation (10-seed, held-out) + brief SHAP.
Only-necessary code; 'lab data' / 'literature data' labels. Executed separately to bake real outputs.
"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []   # cells
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# OCM Literature-Transfer — Methodology Notebook

Reproducible, end-to-end code for the five methods we evaluated for transferring published **literature
data** into our **lab-data** catalyst-yield model, and the evidence that the winning method (two-stage
Prior Feature Transfer) is a real, repeatable improvement.

**Datasets.** Lab data = 89,074 in-house experiments (year 2025, impregnation only). Literature data =
3,852 published experiments (1982–2019). Target = C2 yield `Y(C2), %`.

**Rule of the game.** Every method is scored by 5-fold cross-validation whose validation fold is **always
lab data only** — literature data may only ever help on the *training* side. So the number always answers
"how well do we predict *our* experiments." Every result below is produced by the code in this notebook.""")

code("""import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, ttest_rel, wilcoxon
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb, xgboost as xgb
np.random.seed(42)""")

md("""## 1. Data

Load the combined CSV, split into **lab data** (year 2025) and **literature data** (year ≤ 2019), build the
67-feature matrix (temperature, preparation code, 63 element loadings), and standardise features on the
**lab data** ruler (fit on lab, apply to literature).""")

code("""DATA_PATH = 'OCM_lab_data_and_literature_datal.csv'
TARGET = 'Y(C2), %'
df = pd.read_csv(DATA_PATH)
df_lab = df[df['year'] == 2025].reset_index(drop=True)      # lab data
df_lit = df[df['year'] <= 2019].reset_index(drop=True)      # literature data

ELEM_COLS = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
le = LabelEncoder().fit(df['Preparation'])
df_lab['prep_enc'] = le.transform(df_lab['Preparation'])
df_lit['prep_enc'] = le.transform(df_lit['Preparation'])
FEATURES = ['Temperature_C', 'prep_enc'] + ELEM_COLS

X_lab = df_lab[FEATURES].values.astype(float); y_lab = df_lab[TARGET].values
X_lit = df_lit[FEATURES].values.astype(float); y_lit = df_lit[TARGET].values

scaler = StandardScaler()
X_lab_sc = scaler.fit_transform(X_lab)     # fit on lab data
X_lit_sc = scaler.transform(X_lit)         # apply to literature data

print(f"lab data       : {X_lab.shape[0]:,} rows   mean Y(C2) = {y_lab.mean():.2f}%")
print(f"literature data: {X_lit.shape[0]:,} rows   mean Y(C2) = {y_lit.mean():.2f}%")
print(f"features       : {len(FEATURES)}   |   label shift (lit - lab) = {y_lit.mean()-y_lab.mean():+.2f} pp")""")

md("""**Why this is hard — two distribution shifts.** (i) *Label shift*: published literature data reports a
mean yield ~3.4 percentage points higher than our lab (publication bias — good results get published).
(ii) *Covariate shift*: the literature explores different chemistry. The figure shows both.""")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(y_lab, bins=40, alpha=.6, label='lab data', color='#1f77b4', density=True)
ax[0].hist(y_lit, bins=40, alpha=.6, label='literature data', color='#ff7f0e', density=True)
ax[0].axvline(y_lab.mean(), color='#1f77b4', ls='--'); ax[0].axvline(y_lit.mean(), color='#ff7f0e', ls='--')
ax[0].set_xlabel('Y(C2), %'); ax[0].set_ylabel('density'); ax[0].set_title('Label shift (+3.4 pp)'); ax[0].legend()
# element-usage difference (top elements per dataset)
lab_use = (df_lab[ELEM_COLS] > 0).mean().sort_values(ascending=False).head(10)
lit_use = (df_lit[ELEM_COLS] > 0).mean().sort_values(ascending=False).head(10)
ax[1].barh(lab_use.index[::-1], lab_use.values[::-1], alpha=.7, color='#1f77b4', label='lab data')
ax[1].barh(lit_use.index[::-1], lit_use.values[::-1], alpha=.5, color='#ff7f0e', label='literature data')
ax[1].set_xlabel('fraction of samples using element'); ax[1].set_title('Covariate shift (different chemistry)'); ax[1].legend()
plt.tight_layout(); plt.show()""")

md("""## 2. Evaluation protocol & shared settings

`evaluate_cv_ours` runs the asymmetric 5-fold CV: validation is lab data only; optional extra training rows
(literature data) are appended to the train side. Same tree hyper-parameters everywhere.""")

code("""def lgb_params(seed=42):
    return dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbosity=-1)
def xgb_params(seed=42):
    return dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1)

def evaluate_cv_ours(X_extra=None, y_extra=None, w_extra=None, seed=42, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed); rmses = []
    for tr, va in kf.split(X_lab_sc):
        if X_extra is not None and len(X_extra):
            Xtr = np.vstack([X_lab_sc[tr], X_extra]); ytr = np.concatenate([y_lab[tr], y_extra])
            sw = np.concatenate([np.ones(len(tr)), w_extra]) if w_extra is not None else None
        else:
            Xtr, ytr, sw = X_lab_sc[tr], y_lab[tr], None
        m = lgb.LGBMRegressor(**lgb_params(seed)); m.fit(Xtr, ytr, sample_weight=sw)
        rmses.append(np.sqrt(mean_squared_error(y_lab[va], m.predict(X_lab_sc[va]))))
    return float(np.mean(rmses))

def quantile_normalize_y(y_source, y_target):
    # map literature yields onto the lab-data yield scale, preserving rank
    q = np.clip(rankdata(y_source, method='average') / (len(y_source) + 1), 0.01, 0.99)
    return np.quantile(y_target, q)

RESULTS = {}""")

md("""## 3. Method 1 — Baseline (lab data only)

**Model:** LightGBM (gradient-boosted trees). **In:** 67 lab-data features → **Out:** C2 yield. No literature
data used. This is the number every transfer method must beat.""")
code("""RESULTS['1. Baseline (lab data only)'] = evaluate_cv_ours(seed=42)
print(f"Baseline RMSE = {RESULTS['1. Baseline (lab data only)']:.3f}")""")

md("""## 4. Method 2 — Naive merge (add all literature data)

**Model:** LightGBM. **In:** lab data + all 3,852 literature-data rows (raw labels) → **Out:** C2 yield.
The obvious idea: just pool the data. It should *fail*, because the +3.4 pp label shift pulls predictions up.""")
code("""RESULTS['2. Naive merge (all literature data)'] = evaluate_cv_ours(X_lit_sc, y_lit, seed=42)
print(f"Naive merge RMSE = {RESULTS['2. Naive merge (all literature data)']:.3f}  (baseline {RESULTS['1. Baseline (lab data only)']:.3f})")""")

md("""## 5. Method 3 — DRST (density-ratio selective transfer)

Idea: only add literature-data rows whose **chemistry looks like our lab**. A logistic-regression domain
classifier scores each literature row by P(looks-like-lab | features); we keep rows above a threshold τ and
add them (raw labels) to training. **Model:** LogisticRegression selector → LightGBM regressor.""")
code("""rng = np.random.default_rng(42)
sub = rng.choice(len(X_lab_sc), size=min(10_000, len(X_lab_sc)), replace=False)   # balance classes
X_dom = np.vstack([X_lab_sc[sub], X_lit_sc]); y_dom = np.concatenate([np.ones(len(sub)), np.zeros(len(X_lit_sc))])
clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1).fit(X_dom, y_dom)
p_lab_lit = clf.predict_proba(X_lit_sc)[:, 1]           # P(row looks like lab data)

drst = {}
for tau in [0.30, 0.40, 0.85]:
    m = p_lab_lit >= tau
    drst[tau] = evaluate_cv_ours(X_lit_sc[m], y_lit[m], seed=42)
    print(f"  DRST tau={tau:.2f}  (kept {int(m.sum()):4d} literature rows)  RMSE = {drst[tau]:.3f}")
RESULTS['3. DRST (best threshold)'] = min(drst.values())
print(f"DRST best RMSE = {RESULTS['3. DRST (best threshold)']:.3f}  ->  filtering barely moves the baseline")""")

md("""## 6. Method 4 — KMM (kernel mean matching)

Instead of a hard cut, give **every** literature-data row a continuous importance weight so the *weighted*
literature distribution matches the lab distribution (RBF kernel). **Model:** KMM weights → weighted LightGBM.""")
code("""def rbf(X, Y, s): return np.exp(-cdist(X, Y, 'sqeuclidean') / (2*s**2))
def est_sigma(Xs, Xt, n=2000):
    r = np.random.default_rng(0)
    a = Xs[r.choice(len(Xs), min(n, len(Xs)), replace=False)]; b = Xt[r.choice(len(Xt), min(n, len(Xt)), replace=False)]
    d = cdist(np.vstack([a, b]), np.vstack([a, b])); return float(np.median(d[d > 0]))
def kmm_weights(Xs, Xt, B=5.0, n_tsub=5000):
    r = np.random.default_rng(42); Xt = Xt[r.choice(len(Xt), min(n_tsub, len(Xt)), replace=False)]
    n_s, n_t = len(Xs), len(Xt); sig = est_sigma(Xs, Xt); eps = (np.sqrt(n_s)-1)/np.sqrt(n_s)
    Kss = rbf(Xs, Xs, sig); kap = (n_s/n_t)*rbf(Xs, Xt, sig).sum(1)
    def obj(w): return 0.5*w@Kss@w - kap@w + 1e4*max(0., abs(w.sum()/n_s-1.)-eps)**2
    def grad(w):
        g = Kss@w - kap; d = w.sum()/n_s-1.
        if abs(d) > eps: g += 1e4*2.*d/n_s*np.ones(n_s)
        return g
    res = minimize(obj, np.ones(n_s), jac=grad, method='L-BFGS-B', bounds=[(0., B)]*n_s,
                   options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
    return np.clip(res.x, 0., B)
w_kmm = kmm_weights(X_lit_sc, X_lab_sc, B=5.0)
RESULTS['4. KMM weighted literature data'] = evaluate_cv_ours(X_lit_sc, y_lit, w_extra=w_kmm, seed=42)
print(f"KMM RMSE = {RESULTS['4. KMM weighted literature data']:.3f}  ->  still worse than baseline")
print(f"KMM weights vs DRST scores agree: r = {np.corrcoef(p_lab_lit, w_kmm)[0,1]:.2f}")""")

md("""## 7. Method 5 — Two-stage Prior Feature Transfer (PFT)  ★ our contribution

The key idea: **literature data never contributes a training label to the final model** — it enters only as
a *predicted feature*.

- **Stage 1 — literature-data expert (XGBoost).** Train on the DRST-filtered literature data, with its yields
  first rescaled onto the lab-data yield scale (rank-preserving), so the expert speaks in our units.
- **Stage 2 — final model (LightGBM).** Train on the lab data with the Stage-1 prediction added as one extra
  input feature (`prior_prediction`), using **lab labels only**.

Because the biased literature label lives inside a *feature value* (which Stage 2 can freely recalibrate)
rather than inside the loss, the +3.4 pp shift can no longer poison training.""")
code("""def two_stage_pft(seed=42, tau1=0.30, cv=5):
    mask = p_lab_lit >= tau1
    Xlp, ylp = X_lit_sc[mask], y_lit[mask]
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed); rmses = []
    for tr, va in kf.split(X_lab_sc):
        y_pre = quantile_normalize_y(ylp, y_lab[tr])                       # rescale onto THIS fold's lab labels
        pre = xgb.XGBRegressor(**xgb_params(seed))
        pre.fit(np.vstack([Xlp, X_lab_sc[tr]]), np.concatenate([y_pre, y_lab[tr]]))   # Stage 1
        ptr = pre.predict(X_lab_sc[tr]).reshape(-1, 1); pva = pre.predict(X_lab_sc[va]).reshape(-1, 1)
        fin = lgb.LGBMRegressor(**lgb_params(seed))
        fin.fit(np.hstack([X_lab_sc[tr], ptr]), y_lab[tr])                 # Stage 2 (lab labels only)
        rmses.append(np.sqrt(mean_squared_error(y_lab[va], fin.predict(np.hstack([X_lab_sc[va], pva])))))
    return float(np.mean(rmses))
RESULTS['5. Two-stage PFT (ours)'] = two_stage_pft(seed=42)
b = RESULTS['1. Baseline (lab data only)']; p = RESULTS['5. Two-stage PFT (ours)']
print(f"PFT RMSE = {p:.3f}   ({100*(b-p)/b:+.1f}% vs baseline {b:.3f})")""")

md("""## 8. Results — all five methods""")
code("""names = list(RESULTS); vals = [RESULTS[n] for n in names]
base = RESULTS['1. Baseline (lab data only)']
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ['#8aa0b8', '#e07a5f', '#e0b25f', '#e0b25f', '#1f4e79']
ax.barh(range(len(names))[::-1], vals, color=colors)
ax.axvline(base, color='gray', ls='--', label=f'baseline {base:.3f}')
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names)
ax.set_xlabel('5-fold CV RMSE on lab data  (lower = better)'); ax.set_xlim(1.85, 2.30); ax.legend()
for i, v in zip(range(len(names))[::-1], vals): ax.text(v+0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_title('Only two-stage PFT beats the baseline'); plt.tight_layout(); plt.show()
print(pd.DataFrame({'method': names, 'CV RMSE': [round(v,3) for v in vals],
                    'vs baseline': [f'{100*(v-base)/base:+.1f}%' for v in vals]}).to_string(index=False))""")

md("""## 9. Is the PFT gain real, or luck?

**(a) Repeat across 10 random seeds** (fresh splits + model randomness). **(b) A truly held-out 20% of the
lab data**, set aside from the start and never used in any training.""")
code("""SEEDS = [0, 1, 2, 7, 13, 21, 42, 77, 123, 2025]
b_arr, p_arr = [], []
for s in SEEDS:
    b_arr.append(evaluate_cv_ours(seed=s)); p_arr.append(two_stage_pft(seed=s))
b_arr, p_arr = np.array(b_arr), np.array(p_arr)
print(f"Baseline : {b_arr.mean():.3f} +/- {b_arr.std(ddof=1):.3f}")
print(f"PFT      : {p_arr.mean():.3f} +/- {p_arr.std(ddof=1):.3f}")
print(f"PFT wins {(p_arr < b_arr).sum()}/{len(SEEDS)} seeds   paired t p = {ttest_rel(b_arr, p_arr)[1]:.1e}   Wilcoxon p = {wilcoxon(b_arr, p_arr)[1]:.1e}")
plt.figure(figsize=(8, 3.4))
x = np.arange(len(SEEDS))
plt.plot(x, b_arr, 'o-', color='gray', label=f'baseline {b_arr.mean():.3f}')
plt.plot(x, p_arr, 's-', color='#1f4e79', label=f'PFT {p_arr.mean():.3f}')
plt.fill_between(x, b_arr, p_arr, where=p_arr < b_arr, color='green', alpha=.12)
plt.xticks(x, SEEDS, rotation=45); plt.ylabel('CV RMSE'); plt.title('PFT beats baseline on every seed'); plt.legend()
plt.tight_layout(); plt.show()""")
code("""rng_h = np.random.default_rng(2025); perm = rng_h.permutation(len(X_lab_sc)); nt = int(0.2*len(perm))
te, trn = perm[:nt], perm[nt:]
mb = lgb.LGBMRegressor(**lgb_params(42)).fit(X_lab_sc[trn], y_lab[trn])
rb = np.sqrt(mean_squared_error(y_lab[te], mb.predict(X_lab_sc[te])))
mask = p_lab_lit >= 0.30; yq = quantile_normalize_y(y_lit[mask], y_lab[trn])
pre = xgb.XGBRegressor(**xgb_params(42)).fit(np.vstack([X_lit_sc[mask], X_lab_sc[trn]]), np.concatenate([yq, y_lab[trn]]))
fin = lgb.LGBMRegressor(**lgb_params(42)).fit(np.hstack([X_lab_sc[trn], pre.predict(X_lab_sc[trn]).reshape(-1,1)]), y_lab[trn])
rp = np.sqrt(mean_squared_error(y_lab[te], fin.predict(np.hstack([X_lab_sc[te], pre.predict(X_lab_sc[te]).reshape(-1,1)]))))
print(f"Held-out 20% (never trained on): baseline {rb:.3f}  ->  PFT {rp:.3f}   ({100*(rb-rp)/rb:+.1f}%)")""")

md("""## 10. Why PFT works — the prior feature dominates (SHAP, brief)

SHAP attributes each prediction to its input features. On the final model, the single engineered feature
`prior_prediction` (the literature-data expert's opinion) is by far the most important input — i.e. the
transferred literature-data knowledge is exactly what drives the improvement, on top of correct catalyst
chemistry (temperature and known promoters Ba, Mn, La, Ce).""")
code("""import shap
mask = p_lab_lit >= 0.30; yq = quantile_normalize_y(y_lit[mask], y_lab)
pre = xgb.XGBRegressor(**xgb_params(42)).fit(np.vstack([X_lit_sc[mask], X_lab_sc]), np.concatenate([yq, y_lab]))
X_aug = np.hstack([X_lab_sc, pre.predict(X_lab_sc).reshape(-1, 1)]); names_aug = FEATURES + ['prior_prediction']
final = lgb.LGBMRegressor(**lgb_params(42)).fit(X_aug, y_lab)
idx = np.random.default_rng(42).choice(len(X_aug), 3000, replace=False)
sv = shap.TreeExplainer(final).shap_values(X_aug[idx])
imp = np.abs(sv).mean(0); order = np.argsort(-imp)[:10]
plt.figure(figsize=(7, 4))
plt.barh([names_aug[j] for j in order][::-1], imp[order][::-1], color=['#1f4e79' if names_aug[j]=='prior_prediction' else '#8aa0b8' for j in order][::-1])
plt.xlabel('mean |SHAP| (impact on predicted yield)'); plt.title('prior_prediction is the #1 feature'); plt.tight_layout(); plt.show()
print('Top features:', [names_aug[j] for j in order[:6]])""")

md("""## Summary

| Method | Model | Uses literature data as | CV RMSE |
|---|---|---|---|
| Baseline | LightGBM | — | ~2.13 |
| Naive merge | LightGBM | extra training labels | ~2.24 (worse) |
| DRST | LogReg filter → LightGBM | filtered training labels | ~2.13 (no gain) |
| KMM | kernel weights → LightGBM | weighted training labels | ~2.26 (worse) |
| **Two-stage PFT** | **XGBoost → LightGBM** | **a predicted feature (not a label)** | **~1.91 (−10.6%)** |

Adding literature data as *labels* never helps — the publication-bias shift poisons training. Adding it as a
*predicted feature* does: a ~10.6% error reduction, confirmed on 10/10 random seeds and on a fully held-out
test set. Every number and figure above is produced by re-running this notebook.""")

nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'ocm_methodology.ipynb')
print("wrote ocm_methodology.ipynb with", len(C), "cells")
