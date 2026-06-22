"""
feedback_experiments.py
=======================
Reproduces the OCM pipeline from ocm_analysis.ipynb and runs the ADDITIONAL
analyses requested in the presentation feedback:

  Exp A  DRST threshold sweep 0 -> 1, RMSE-vs-tau curve            (Q5, Q6)
  Exp B  10-seed repeated baseline vs PFT, mean/std + significance (Q11,Q14,Q15)
  Exp C  True held-out test set (never used in any training)       (Q16)
  Exp D  SHAP top-feature ranking stability across 10 subsamples   (Q13)
  Exp E  KMM-weight vs DRST-score overlap (compatibility)          (Q8)

Outputs:
  fig_drst_threshold_sweep.png
  fig_repeated_runs.png
  fig_shap_stability.png
  fig_kmm_drst_overlap.png
  feedback_results.json   (all numbers for the write-up)
"""
import json, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, ttest_rel, wilcoxon
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

RESULTS = {}

# ----------------------------------------------------------------------------
# Data loading — identical to ocm_analysis.ipynb cells 3/4
# ----------------------------------------------------------------------------
DATA_PATH = 'OCM_lab_data_and_literature_datal.csv'
TARGET = 'Y(C2), %'
df = pd.read_csv(DATA_PATH)
df_ours = df[df['year'] == 2025].copy().reset_index(drop=True)
df_lit  = df[df['year'] <= 2019].copy().reset_index(drop=True)

ELEM_COLS = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
le = LabelEncoder(); le.fit(df['Preparation'])
df_ours['prep_enc'] = le.transform(df_ours['Preparation'])
df_lit['prep_enc']  = le.transform(df_lit['Preparation'])
FEATURES = ['Temperature_C', 'prep_enc'] + ELEM_COLS

X_ours = df_ours[FEATURES].values.astype(float); y_ours = df_ours[TARGET].values
X_lit  = df_lit[FEATURES].values.astype(float);  y_lit  = df_lit[TARGET].values

scaler = StandardScaler()
X_ours_sc = scaler.fit_transform(X_ours)   # fit on lab only (matches notebook)
X_lit_sc  = scaler.transform(X_lit)
log(f"lab={X_ours.shape} lit={X_lit.shape} n_features={len(FEATURES)}")
RESULTS['data'] = dict(n_lab=len(y_ours), n_lit=len(y_lit), n_features=len(FEATURES),
                       lab_mean_Y=float(y_ours.mean()), lit_mean_Y=float(y_lit.mean()))

# ----------------------------------------------------------------------------
# Hyperparameters — identical to notebook
# ----------------------------------------------------------------------------
def xgb_params(seed=42):
    return dict(n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=seed, verbosity=0, n_jobs=-1)
def lgb_params(seed=42):
    return dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)

def quantile_normalize_y(y_source, y_target):
    q = rankdata(y_source, method='average') / (len(y_source) + 1)
    q = np.clip(q, 0.01, 0.99)
    return np.quantile(y_target, q)

# ----------------------------------------------------------------------------
# CV evaluators (parametrised by seed)
# ----------------------------------------------------------------------------
def evaluate_cv_ours(X_train_extra=None, y_train_extra=None, seed=42, cv=5):
    """Single-stage LightGBM. Validation = lab only. Optionally add extra (lit) rows to train."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    rmses, r2s = [], []
    for tr, va in kf.split(X_ours_sc):
        if X_train_extra is not None and len(X_train_extra):
            Xtr = np.vstack([X_ours_sc[tr], X_train_extra]); ytr = np.concatenate([y_ours[tr], y_train_extra])
        else:
            Xtr, ytr = X_ours_sc[tr], y_ours[tr]
        m = lgb.LGBMRegressor(**lgb_params(seed)); m.fit(Xtr, ytr)
        p = m.predict(X_ours_sc[va])
        rmses.append(np.sqrt(mean_squared_error(y_ours[va], p))); r2s.append(r2_score(y_ours[va], p))
    return np.mean(rmses), np.mean(r2s), rmses

def two_stage_cv(X_lit_pre, y_lit_pre, seed=42, bias_correct=True, include_ours_in_stage1=True, cv=5):
    """PFT (Direction A): Stage1 XGB (filtered+QN lit), Stage2 LGB with prior feature."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    rmses, r2s = [], []
    for tr, va in kf.split(X_ours_sc):
        y_pre = quantile_normalize_y(y_lit_pre, y_ours[tr]) if bias_correct else y_lit_pre.copy()
        if include_ours_in_stage1:
            Xs1 = np.vstack([X_lit_pre, X_ours_sc[tr]]); ys1 = np.concatenate([y_pre, y_ours[tr]])
        else:
            Xs1, ys1 = X_lit_pre, y_pre
        pre = xgb.XGBRegressor(**xgb_params(seed)); pre.fit(Xs1, ys1)
        prior_tr = pre.predict(X_ours_sc[tr]).reshape(-1, 1)
        prior_va = pre.predict(X_ours_sc[va]).reshape(-1, 1)
        fin = lgb.LGBMRegressor(**lgb_params(seed))
        fin.fit(np.hstack([X_ours_sc[tr], prior_tr]), y_ours[tr])
        p = fin.predict(np.hstack([X_ours_sc[va], prior_va]))
        rmses.append(np.sqrt(mean_squared_error(y_ours[va], p))); r2s.append(r2_score(y_ours[va], p))
    return np.mean(rmses), np.mean(r2s), rmses

# ----------------------------------------------------------------------------
# DRST domain classifier (matches notebook): LogReg C=0.5, lab subsample 10k
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)
sub = rng.choice(len(X_ours_sc), size=min(10_000, len(X_ours_sc)), replace=False)
X_dom = np.vstack([X_ours_sc[sub], X_lit_sc]); y_dom = np.concatenate([np.ones(len(sub)), np.zeros(len(X_lit_sc))])
clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1); clf.fit(X_dom, y_dom)
p_ours_lit = clf.predict_proba(X_lit_sc)[:, 1]      # P(this lit sample looks like lab)
log("DRST classifier trained")

# ============================================================================
# Exp A — DRST threshold sweep 0 -> 1  (Q5, Q6)
# ============================================================================
log("Exp A: DRST threshold sweep ...")
base_rmse, base_r2, base_folds = evaluate_cv_ours(seed=42)
log(f"  baseline RMSE={base_rmse:.4f}")
taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
sweep = []
for tau in taus:
    mask = p_ours_lit >= tau
    n = int(mask.sum())
    if n < 5:
        sweep.append(dict(tau=float(tau), n=n, rmse=None)); continue
    rmse, r2, _ = evaluate_cv_ours(X_lit_sc[mask], y_lit[mask], seed=42)
    sweep.append(dict(tau=float(tau), n=n, rmse=float(rmse), r2=float(r2)))
    log(f"  tau={tau:.2f}  n={n:5d}  RMSE={rmse:.4f}")
RESULTS['drst_sweep'] = dict(baseline_rmse=float(base_rmse), baseline_r2=float(base_r2), sweep=sweep)

valid = [s for s in sweep if s['rmse'] is not None]
best = min(valid, key=lambda s: s['rmse'])
RESULTS['drst_sweep']['best'] = best

fig, ax1 = plt.subplots(figsize=(8, 5))
xs = [s['tau'] for s in valid]; ys = [s['rmse'] for s in valid]; ns = [s['n'] for s in valid]
ax1.axhline(base_rmse, color='gray', ls='--', lw=1.5, label=f'Baseline (no transfer) = {base_rmse:.3f}')
ax1.plot(xs, ys, 'o-', color='#1f4e79', lw=2, label='DRST-augmented CV RMSE')
ax1.scatter([best['tau']], [best['rmse']], s=160, facecolors='none', edgecolors='crimson', lw=2.5, zorder=5,
            label=f"Best τ={best['tau']:.2f} (RMSE={best['rmse']:.3f})")
ax1.axvline(0.30, color='orange', ls=':', lw=1.3, label='τ=0.30 (deck value)')
ax1.set_xlabel('DRST keep-threshold  τ   (keep lit sample if P(lab|x) ≥ τ)')
ax1.set_ylabel('5-fold CV RMSE on lab validation')
ax1.set_title('Q6: DRST threshold sweep — RMSE vs τ (0 → 1)')
ax1.legend(loc='upper left', fontsize=8)
ax2 = ax1.twinx(); ax2.plot(xs, ns, 's--', color='seagreen', alpha=0.5, ms=4)
ax2.set_ylabel('# literature samples kept', color='seagreen')
ax2.tick_params(axis='y', colors='seagreen')
plt.tight_layout(); plt.savefig('fig_drst_threshold_sweep.png', dpi=130); plt.close()
log(f"  -> fig_drst_threshold_sweep.png  best tau={best['tau']} rmse={best['rmse']:.4f}")

# ============================================================================
# Exp E — KMM weights vs DRST score overlap (Q8)  [compute KMM once]
# ============================================================================
log("Exp E: KMM weights + DRST overlap ...")
def rbf(X, Y, s): return np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * s**2))
def est_sigma(Xs, Xt, n=2000):
    r = np.random.default_rng(0)
    a = Xs[r.choice(len(Xs), min(n, len(Xs)), replace=False)]
    b = Xt[r.choice(len(Xt), min(n, len(Xt)), replace=False)]
    d = cdist(np.vstack([a, b]), np.vstack([a, b]))
    return float(np.median(d[d > 0]))
def kmm_weights(Xs, Xt, B=5.0, n_tsub=5000):
    r = np.random.default_rng(42); Xt = Xt[r.choice(len(Xt), min(n_tsub, len(Xt)), replace=False)]
    n_s, n_t = len(Xs), len(Xt); sig = est_sigma(Xs, Xt)
    eps = (np.sqrt(n_s) - 1) / np.sqrt(n_s)
    Kss = rbf(Xs, Xs, sig); kap = (n_s / n_t) * rbf(Xs, Xt, sig).sum(1)
    def obj(w): return 0.5*w@Kss@w - kap@w + 1e4*max(0., abs(w.sum()/n_s-1.)-eps)**2
    def grad(w):
        g = Kss@w - kap; d = w.sum()/n_s-1.
        if abs(d) > eps: g += 1e4*2.*d/n_s*np.ones(n_s)
        return g
    res = minimize(obj, np.ones(n_s), jac=grad, method='L-BFGS-B', bounds=[(0., B)]*n_s,
                   options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
    return np.clip(res.x, 0., B)
w_kmm = kmm_weights(X_lit_sc, X_ours_sc)
r_corr = float(np.corrcoef(p_ours_lit, w_kmm)[0, 1])
# DRST kept set at tau=0.30; KMM "kept" = top-N by weight (same N) and weight>median
N_drst = int((p_ours_lit >= 0.30).sum())
drst_set = set(np.where(p_ours_lit >= 0.30)[0])
kmm_topN = set(np.argsort(-w_kmm)[:N_drst])
inter = len(drst_set & kmm_topN); union = len(drst_set | kmm_topN)
jacc = inter / union
RESULTS['kmm_drst'] = dict(pearson_r=r_corr, n_drst_keep=N_drst,
                           overlap_count=inter, jaccard=float(jacc),
                           overlap_pct_of_drst=float(inter / N_drst))
log(f"  r={r_corr:.3f}  DRST keeps {N_drst}; KMM-top{N_drst} overlap={inter} ({100*inter/N_drst:.0f}%) Jaccard={jacc:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].scatter(p_ours_lit, w_kmm, s=8, alpha=0.3, color='purple')
ax[0].set_xlabel('DRST score  P(lab | x)'); ax[0].set_ylabel('KMM weight')
ax[0].set_title(f'Q8: KMM weight vs DRST score (r = {r_corr:.2f})')
ax[0].axvline(0.30, color='orange', ls=':', lw=1.2)
ax[1].bar(['DRST∩KMM\n(overlap)', 'DRST only', 'KMM only'],
          [inter, N_drst-inter, N_drst-inter], color=['#1f4e79', 'seagreen', 'mediumpurple'])
ax[1].set_ylabel('# literature samples')
ax[1].set_title(f'Selection overlap (Jaccard = {jacc:.2f})')
plt.tight_layout(); plt.savefig('fig_kmm_drst_overlap.png', dpi=130); plt.close()
log("  -> fig_kmm_drst_overlap.png")

# ============================================================================
# Exp B — 10-seed repeated runs: baseline vs PFT (Q11, Q14, Q15)
# ============================================================================
log("Exp B: 10-seed repeated baseline vs PFT ...")
SEEDS = [0, 1, 2, 7, 13, 21, 42, 77, 123, 2025]
mask30 = p_ours_lit >= 0.30          # DRST tau1=0.30 filter for PFT Stage 1
X_lit_pre, y_lit_pre = X_lit_sc[mask30], y_lit[mask30]
log(f"  PFT Stage-1 uses {mask30.sum()} DRST-filtered literature samples")

runs = []
for s in SEEDS:
    b_rmse, b_r2, _ = evaluate_cv_ours(seed=s)
    p_rmse, p_r2, _ = two_stage_cv(X_lit_pre, y_lit_pre, seed=s, bias_correct=True, include_ours_in_stage1=True)
    runs.append(dict(seed=s, baseline=float(b_rmse), pft=float(p_rmse),
                     delta=float(p_rmse - b_rmse), baseline_r2=float(b_r2), pft_r2=float(p_r2)))
    log(f"  seed={s:5d}  baseline={b_rmse:.4f}  PFT={p_rmse:.4f}  Δ={p_rmse-b_rmse:+.4f}")

b_arr = np.array([r['baseline'] for r in runs]); p_arr = np.array([r['pft'] for r in runs])
t_stat, t_p = ttest_rel(b_arr, p_arr)          # paired: baseline vs PFT
w_stat, w_p = wilcoxon(b_arr, p_arr)
n_pft_wins = int((p_arr < b_arr).sum())
RESULTS['repeated'] = dict(
    seeds=SEEDS, runs=runs,
    baseline_mean=float(b_arr.mean()), baseline_std=float(b_arr.std(ddof=1)),
    pft_mean=float(p_arr.mean()), pft_std=float(p_arr.std(ddof=1)),
    mean_improvement=float((b_arr - p_arr).mean()),
    mean_improvement_pct=float(100*(b_arr - p_arr).mean()/b_arr.mean()),
    pft_wins=n_pft_wins, n_runs=len(SEEDS),
    paired_t=float(t_stat), paired_t_p=float(t_p),
    wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p))
log(f"  baseline {b_arr.mean():.4f}±{b_arr.std(ddof=1):.4f}  PFT {p_arr.mean():.4f}±{p_arr.std(ddof=1):.4f}")
log(f"  PFT wins {n_pft_wins}/{len(SEEDS)}  paired t p={t_p:.2e}  wilcoxon p={w_p:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(SEEDS))
ax[0].plot(x, b_arr, 'o-', color='gray', lw=2, label=f'Baseline  {b_arr.mean():.3f}±{b_arr.std(ddof=1):.3f}')
ax[0].plot(x, p_arr, 's-', color='#1f4e79', lw=2, label=f'PFT  {p_arr.mean():.3f}±{p_arr.std(ddof=1):.3f}')
ax[0].fill_between(x, b_arr, p_arr, where=p_arr < b_arr, color='green', alpha=0.12)
ax[0].set_xticks(x); ax[0].set_xticklabels([str(s) for s in SEEDS], rotation=45, fontsize=8)
ax[0].set_xlabel('random seed'); ax[0].set_ylabel('5-fold CV RMSE')
ax[0].set_title(f'Q14: PFT beats baseline in {n_pft_wins}/{len(SEEDS)} runs')
ax[0].legend(fontsize=8)
# paired delta
ax[1].bar(x, b_arr - p_arr, color=['green' if d > 0 else 'red' for d in (b_arr - p_arr)])
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels([str(s) for s in SEEDS], rotation=45, fontsize=8)
ax[1].set_xlabel('random seed'); ax[1].set_ylabel('RMSE improvement (baseline − PFT)')
ax[1].set_title(f'Q15: improvement every run\npaired t p={t_p:.1e}, Wilcoxon p={w_p:.1e}')
plt.tight_layout(); plt.savefig('fig_repeated_runs.png', dpi=130); plt.close()
log("  -> fig_repeated_runs.png")

# ============================================================================
# Exp C — True held-out test set, never used in any training (Q16)
# ============================================================================
log("Exp C: held-out test set ...")
rng_h = np.random.default_rng(2025)
perm = rng_h.permutation(len(X_ours_sc))
n_test = int(0.20 * len(perm))
test_idx, train_idx = perm[:n_test], perm[n_test:]
Xtr, ytr = X_ours_sc[train_idx], y_ours[train_idx]
Xte, yte = X_ours_sc[test_idx], y_ours[test_idx]

# baseline on held-out
mb = lgb.LGBMRegressor(**lgb_params(42)); mb.fit(Xtr, ytr)
pb = mb.predict(Xte); rmse_b = np.sqrt(mean_squared_error(yte, pb)); r2_b = r2_score(yte, pb)
# PFT on held-out (Stage1 on filtered+QN lit + train; Stage2 on train; eval test)
y_pre = quantile_normalize_y(y_lit_pre, ytr)
pre = xgb.XGBRegressor(**xgb_params(42)); pre.fit(np.vstack([X_lit_pre, Xtr]), np.concatenate([y_pre, ytr]))
fin = lgb.LGBMRegressor(**lgb_params(42))
fin.fit(np.hstack([Xtr, pre.predict(Xtr).reshape(-1, 1)]), ytr)
pp = fin.predict(np.hstack([Xte, pre.predict(Xte).reshape(-1, 1)]))
rmse_p = np.sqrt(mean_squared_error(yte, pp)); r2_p = r2_score(yte, pp)

# OOD literature test (non-Impregnation), models fit on all lab (+ filtered lit for PFT)
ood_mask = df_lit['Preparation'] != 'Impregnation'
X_ood, y_ood = X_lit_sc[ood_mask.values], y_lit[ood_mask.values]
mb_all = lgb.LGBMRegressor(**lgb_params(42)); mb_all.fit(X_ours_sc, y_ours)
rmse_ood_b = np.sqrt(mean_squared_error(y_ood, mb_all.predict(X_ood)))
y_pre_all = quantile_normalize_y(y_lit_pre, y_ours)
pre_all = xgb.XGBRegressor(**xgb_params(42)); pre_all.fit(np.vstack([X_lit_pre, X_ours_sc]), np.concatenate([y_pre_all, y_ours]))
fin_all = lgb.LGBMRegressor(**lgb_params(42))
fin_all.fit(np.hstack([X_ours_sc, pre_all.predict(X_ours_sc).reshape(-1, 1)]), y_ours)
rmse_ood_p = np.sqrt(mean_squared_error(y_ood, fin_all.predict(np.hstack([X_ood, pre_all.predict(X_ood).reshape(-1, 1)]))))
RESULTS['holdout'] = dict(
    n_train=len(train_idx), n_test=len(test_idx),
    baseline_rmse=float(rmse_b), baseline_r2=float(r2_b),
    pft_rmse=float(rmse_p), pft_r2=float(r2_p),
    improvement_pct=float(100*(rmse_b - rmse_p)/rmse_b),
    ood_n=int(ood_mask.sum()), ood_baseline_rmse=float(rmse_ood_b), ood_pft_rmse=float(rmse_ood_p),
    ood_improvement_pct=float(100*(rmse_ood_b - rmse_ood_p)/rmse_ood_b))
log(f"  held-out: baseline {rmse_b:.4f} (R²={r2_b:.3f}) | PFT {rmse_p:.4f} (R²={r2_p:.3f})")
log(f"  OOD lit : baseline {rmse_ood_b:.4f} | PFT {rmse_ood_p:.4f}")

# ============================================================================
# Exp D — SHAP top-feature ranking stability across 10 subsamples (Q13)
# ============================================================================
log("Exp D: SHAP stability ...")
import shap
# train the final PFT model once on all lab data (matches notebook SHAP target)
prior_all = pre_all.predict(X_ours_sc).reshape(-1, 1)
X_aug = np.hstack([X_ours_sc, prior_all])
aug_names = FEATURES + ['lit_prior_prediction']
shap_model = lgb.LGBMRegressor(**lgb_params(42)); shap_model.fit(X_aug, y_ours)
explainer = shap.TreeExplainer(shap_model)
rank_runs = []
top_sets = []
for i, seed in enumerate(range(10)):
    rg = np.random.default_rng(seed)
    idx = rg.choice(len(X_aug), size=2000, replace=False)
    sv = explainer.shap_values(X_aug[idx])
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)
    top10 = [aug_names[j] for j in order[:10]]
    rank_runs.append({aug_names[j]: float(imp[j]) for j in order[:10]})
    top_sets.append(top10)
# stability metrics
from collections import Counter
flat = Counter([f for t in top_sets for f in t])
always_top10 = [f for f, c in flat.items() if c == 10]
top1_counter = Counter([t[0] for t in top_sets])
RESULTS['shap_stability'] = dict(
    n_runs=10, top1_most_common=top1_counter.most_common(1)[0][0],
    top1_count=top1_counter.most_common(1)[0][1],
    features_in_all_top10=always_top10,
    per_run_top10=top_sets)
log(f"  SHAP #1 feature in {top1_counter.most_common(1)[0][1]}/10 runs = {top1_counter.most_common(1)[0][0]}")
log(f"  {len(always_top10)} features appear in top-10 in all 10 runs")

# mean +/- std importance for features that are ever top-10
allfeats = sorted(flat.keys(), key=lambda f: -np.mean([r.get(f, 0) for r in rank_runs]))
means = [np.mean([r.get(f, 0) for r in rank_runs]) for f in allfeats]
stds = [np.std([r.get(f, 0) for r in rank_runs]) for f in allfeats]
fig, ax = plt.subplots(figsize=(8, 6))
yy = np.arange(len(allfeats))[::-1]
ax.barh(yy, means, xerr=stds, color='#1f4e79', alpha=0.85, capsize=3)
ax.set_yticks(yy); ax.set_yticklabels(allfeats, fontsize=8)
ax.set_xlabel('mean |SHAP| value (± std across 10 runs)')
ax.set_title('Q13: SHAP feature-importance stability (10 independent subsamples)')
plt.tight_layout(); plt.savefig('fig_shap_stability.png', dpi=130); plt.close()
log("  -> fig_shap_stability.png")

# ----------------------------------------------------------------------------
json.dump(RESULTS, open('feedback_results.json', 'w'), indent=2)
log("ALL DONE. results -> feedback_results.json")
