"""
taniike_validation.py — the stricter validation Prof. Taniike requested, in four parts.

  A. Catalyst-grouped 5-fold CV (all rows of a catalyst in one fold; 917 groups) — baseline vs
     PFT (DRST-filtered and all-literature Stage 1), 5 seeds, per-fold train-only DRST classifier
     and per-fold scaler (nothing from a test fold touches any part of the pipeline).
  B. Catalyst-family holdout — remove ALL catalysts containing a given element (Ba/La/Ti/Zr/Ce),
     train on the rest, predict the family. Two literature variants: lit-all (literature may
     contain the element) and lit-excl (element-containing literature removed too).
  C. Catalyst-level metrics, computed for A and B on test predictions only:
     predicted-max vs observed-max yield per catalyst (Pearson + Spearman), enrichment of true
     top-10% catalysts among the top-10% predicted, and precision@20 (top-20 predicted that are
     true top-10% — a realistic synthesis budget).
  D. Quantile-normalisation ablation (Taniike's follow-up email): Stage-1 label treatments
     qn_joint (current) / raw_joint / qn_litonly / rank_litonly, under BOTH row-level and
     grouped protocols, 3 seeds.

Every number is written to taniike_validation.json. Protocol labels are attached to every result —
row-level and grouped numbers must never be shown side-by-side unlabeled.

NOTE vs published numbers: the published row-level PFT (1.909 +/- 0.002, 10 seeds) used a single
global DRST classifier; here the classifier and scaler are refit per fold on train rows only
(stricter). Small differences vs published row-level numbers are expected and are labeled.
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb, xgboost as xgb

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---- data (identical loading to ocm_methodology.ipynb) ----
TARGET = 'Y(C2), %'
df = pd.read_csv('OCM_lab_data_and_literature_datal.csv')
dl_lab = df[df.year == 2025].reset_index(drop=True)
dl_lit = df[df.year <= 2019].reset_index(drop=True)
EL = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
le = LabelEncoder().fit(df.Preparation)
dl_lab['prep_enc'] = le.transform(dl_lab.Preparation)
dl_lit['prep_enc'] = le.transform(dl_lit.Preparation)
F = ['Temperature_C', 'prep_enc'] + EL
X_lab = dl_lab[F].values.astype(float); y_lab = dl_lab[TARGET].values
X_lit = dl_lit[F].values.astype(float); y_lit = dl_lit[TARGET].values

# catalyst identity = preparation + full element-loading vector (temperature is a condition, not identity)
cat_str = dl_lab[['Preparation'] + EL].astype(str).agg('|'.join, axis=1)
groups, cat_uniques = pd.factorize(cat_str)
n_cat = len(cat_uniques)
log(f"lab={X_lab.shape} lit={X_lit.shape} catalysts={n_cat}")

def lgb_params(seed=42): return dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=seed, n_jobs=-1, verbosity=-1)
def xgb_params(seed=42): return dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1)
def quantile_normalize_y(ys, yt):
    q = np.clip(rankdata(ys, method='average') / (len(ys) + 1), 0.01, 0.99)
    return np.quantile(yt, q)

def grouped_folds(seed, n_folds=5):
    """All rows of a catalyst go to exactly one fold; group-to-fold assignment shuffled by seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cat)
    fold_of_group = np.empty(n_cat, dtype=int)
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        fold_of_group[chunk] = k
    fold = fold_of_group[groups]
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_folds)]

def row_folds(seed, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    return list(kf.split(X_lab))

def drst_mask(Xlab_tr_sc, Xlit_sc, tau=0.30, seed=42):
    """Domain classifier trained on TRAIN-fold lab rows only -> which literature rows look lab-like."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xlab_tr_sc), size=min(10_000, len(Xlab_tr_sc)), replace=False)
    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1).fit(
        np.vstack([Xlab_tr_sc[sub], Xlit_sc]),
        np.r_[np.ones(len(sub)), np.zeros(len(Xlit_sc))])
    return clf.predict_proba(Xlit_sc)[:, 1] >= tau

def stage1_data(kind, Xlit_f, ylit_f, Xlab_tr, ylab_tr):
    """Stage-1 design matrix + labels for one ablation config."""
    if kind == 'qn_joint':      # current PFT: lit labels QN'd onto lab-train scale, joint training
        return np.vstack([Xlit_f, Xlab_tr]), np.concatenate([quantile_normalize_y(ylit_f, ylab_tr), ylab_tr])
    if kind == 'raw_joint':     # raw literature yields, joint training (scale mismatch left in)
        return np.vstack([Xlit_f, Xlab_tr]), np.concatenate([ylit_f, ylab_tr])
    if kind == 'qn_litonly':    # literature-only expert, QN labels
        return Xlit_f, quantile_normalize_y(ylit_f, ylab_tr)
    if kind == 'rank_litonly':  # literature-only expert, pure rank labels (ordering only, no scale)
        return Xlit_f, rankdata(ylit_f, method='average') / (len(ylit_f) + 1)
    raise ValueError(kind)

def run_fold(tr, va, seed, model, s1_kind='qn_joint', lit_X=None, lit_y=None, use_filter=True):
    """Train on lab[tr] (+ literature per config), predict lab[va]. Scaler + DRST are train-only."""
    sc = StandardScaler().fit(X_lab[tr])
    Xtr, Xva = sc.transform(X_lab[tr]), sc.transform(X_lab[va])
    if model == 'baseline':
        m = lgb.LGBMRegressor(**lgb_params(seed)).fit(Xtr, y_lab[tr])
        return m.predict(Xva)
    Xl = sc.transform(lit_X if lit_X is not None else X_lit)
    yl = lit_y if lit_y is not None else y_lit
    if use_filter:
        m = drst_mask(Xtr, Xl)
        Xl, yl = Xl[m], yl[m]
    Xs1, ys1 = stage1_data(s1_kind, Xl, yl, Xtr, y_lab[tr])
    pre = xgb.XGBRegressor(**xgb_params(seed)).fit(Xs1, ys1)
    ptr = pre.predict(Xtr).reshape(-1, 1); pva = pre.predict(Xva).reshape(-1, 1)
    fin = lgb.LGBMRegressor(**lgb_params(seed)).fit(np.hstack([Xtr, ptr]), y_lab[tr])
    return fin.predict(np.hstack([Xva, pva]))

def row_metrics(yt, yp):
    return dict(rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                mae=float(mean_absolute_error(yt, yp)), r2=float(r2_score(yt, yp)))

def cat_metrics(cat_codes, yt, yp, top_frac=0.10, budget=20):
    """Catalyst-level: max over each catalyst's test rows, predicted vs observed."""
    g = pd.DataFrame({'c': cat_codes, 't': yt, 'p': yp}).groupby('c').agg(t=('t', 'max'), p=('p', 'max'))
    n = len(g); K = max(1, int(round(top_frac * n)))
    top_true = set(g.nlargest(K, 't').index)
    top_pred = list(g.nlargest(K, 'p').index)
    prec = float(np.mean([c in top_true for c in top_pred]))
    B = min(budget, n)
    prec_budget = float(np.mean([c in top_true for c in g.nlargest(B, 'p').index]))
    return dict(n_catalysts=int(n),
                pearson_max=float(np.corrcoef(g.t, g.p)[0, 1]),
                spearman_max=float(spearmanr(g.t, g.p)[0]),
                precision_top10pct=prec, enrichment_top10pct=float(prec / top_frac),
                precision_at20_vs_top10pct=prec_budget)

def run_cv(folds, seed, model, **kw):
    """Run a full CV; return pooled row metrics + catalyst metrics over all test predictions."""
    yp_all = np.empty(len(y_lab)); per_fold_rmse = []
    for tr, va in folds:
        yp = run_fold(tr, va, seed, model, **kw)
        yp_all[va] = yp
        per_fold_rmse.append(np.sqrt(mean_squared_error(y_lab[va], yp)))
    out = row_metrics(y_lab, yp_all)
    out['rmse_foldmean'] = float(np.mean(per_fold_rmse))   # comparable to published per-fold-mean RMSE
    out.update(cat_metrics(groups, y_lab, yp_all))
    return out

RESULTS = {'meta': {
    'n_lab_rows': int(len(y_lab)), 'n_lit_rows': int(len(y_lit)), 'n_catalysts': int(n_cat),
    'note': ('Grouped protocol: all rows of a catalyst in one fold. Per-fold train-only scaler + DRST '
             'classifier everywhere (stricter than published row-level numbers, which used a global '
             'classifier). rmse_foldmean is the number comparable to published per-fold-mean RMSE.')}}

# =====================================================================================
# PART A — catalyst-grouped CV vs row-level CV (baseline, PFT-filtered, PFT-all), 5 seeds
# =====================================================================================
SEEDS_A = [0, 1, 2, 7, 13]
log("PART A: grouped vs row-level CV")
A = {}
for proto, mkfolds in [('grouped', grouped_folds), ('row', row_folds)]:
    for name, kw in [('baseline', dict(model='baseline')),
                     ('pft_filtered', dict(model='pft', s1_kind='qn_joint', use_filter=True)),
                     ('pft_all_lit', dict(model='pft', s1_kind='qn_joint', use_filter=False))]:
        runs = []
        for s in SEEDS_A:
            r = run_cv(mkfolds(s), s, **kw)
            runs.append(r)
            log(f"  A {proto:7s} {name:13s} seed={s:2d}  RMSE(foldmean)={r['rmse_foldmean']:.3f}  "
                f"spearman_max={r['spearman_max']:.3f}  enrich@10%={r['enrichment_top10pct']:.2f}x")
        A[f'{proto}/{name}'] = {
            'per_seed': runs,
            'rmse_foldmean_mean': float(np.mean([r['rmse_foldmean'] for r in runs])),
            'rmse_foldmean_std': float(np.std([r['rmse_foldmean'] for r in runs], ddof=1)),
            'spearman_max_mean': float(np.mean([r['spearman_max'] for r in runs])),
            'pearson_max_mean': float(np.mean([r['pearson_max'] for r in runs])),
            'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
            'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in runs]))}
RESULTS['A_grouped_vs_row'] = A

# =====================================================================================
# PART B — catalyst-family holdout (train never sees any catalyst containing the element)
# =====================================================================================
log("PART B: family holdout")
B = {}
lab_has = {el: (dl_lab[el].values > 0) for el in ['Ba', 'La', 'Ti', 'Zr', 'Ce']}
lit_has = {el: (dl_lit[el].values > 0) for el in ['Ba', 'La', 'Ti', 'Zr', 'Ce']}
for el, mask_lab in lab_has.items():
    te = np.where(mask_lab)[0]; tr = np.where(~mask_lab)[0]
    fam_cats = np.unique(groups[te])
    entry = {'n_test_rows': int(len(te)), 'n_test_catalysts': int(len(fam_cats)),
             'n_lit_rows_with_element': int(lit_has[el].sum())}
    for name, kw in [
            ('baseline', dict(model='baseline')),
            ('pft_lit_all', dict(model='pft', s1_kind='qn_joint', use_filter=True)),
            ('pft_lit_excl', dict(model='pft', s1_kind='qn_joint', use_filter=True,
                                  lit_X=X_lit[~lit_has[el]], lit_y=y_lit[~lit_has[el]]))]:
        yp = run_fold(tr, te, 42, **kw)
        r = row_metrics(y_lab[te], yp)
        r.update(cat_metrics(groups[te], y_lab[te], yp))
        entry[name] = r
        log(f"  B holdout={el:2s} {name:12s} RMSE={r['rmse']:.3f} spearman_max={r['spearman_max']:.3f} "
            f"enrich@10%={r['enrichment_top10pct']:.2f}x  (n_cat={r['n_catalysts']})")
    B[el] = entry
RESULTS['B_family_holdout'] = B

# =====================================================================================
# PART D — QN ablation: Stage-1 label treatment, row-level AND grouped protocols, 3 seeds
# =====================================================================================
SEEDS_D = [0, 1, 2]
log("PART D: QN ablation")
D = {}
for proto, mkfolds in [('row', row_folds), ('grouped', grouped_folds)]:
    for kind in ['qn_joint', 'raw_joint', 'qn_litonly', 'rank_litonly']:
        runs = []
        for s in SEEDS_D:
            r = run_cv(mkfolds(s), s, model='pft', s1_kind=kind, use_filter=True)
            runs.append(r)
            log(f"  D {proto:7s} {kind:12s} seed={s}  RMSE(foldmean)={r['rmse_foldmean']:.3f}  "
                f"spearman_max={r['spearman_max']:.3f}")
        D[f'{proto}/{kind}'] = {
            'per_seed': runs,
            'rmse_foldmean_mean': float(np.mean([r['rmse_foldmean'] for r in runs])),
            'rmse_foldmean_std': float(np.std([r['rmse_foldmean'] for r in runs], ddof=1)),
            'spearman_max_mean': float(np.mean([r['spearman_max'] for r in runs]))}
RESULTS['D_qn_ablation'] = D

json.dump(RESULTS, open('taniike_validation.json', 'w'), indent=1)
log("wrote taniike_validation.json")

# =====================================================================================
# Summary tables (percentages recomputed from the stated bases)
# =====================================================================================
print("\n================= SUMMARY =================")
print("\n--- A. Row-level vs catalyst-grouped CV (5 seeds; per-fold-mean RMSE) ---")
for proto in ['row', 'grouped']:
    b = A[f'{proto}/baseline']['rmse_foldmean_mean']
    for name in ['baseline', 'pft_filtered', 'pft_all_lit']:
        a = A[f'{proto}/{name}']
        d = 100 * (a['rmse_foldmean_mean'] - b) / b
        print(f"{proto:8s} {name:13s} RMSE {a['rmse_foldmean_mean']:.3f} +/- {a['rmse_foldmean_std']:.3f}"
              f"  ({d:+.1f}% vs {proto} baseline {b:.3f})   spearman_max {a['spearman_max_mean']:.3f}"
              f"   enrich@10% {a['enrichment_mean']:.2f}x   prec@20 {a['precision_at20_mean']:.2f}")
print("\n--- B. Family holdout (seed 42) ---")
for el, e in B.items():
    for name in ['baseline', 'pft_lit_all', 'pft_lit_excl']:
        r = e[name]
        print(f"holdout {el:2s} ({e['n_test_catalysts']:3d} cats) {name:12s} RMSE {r['rmse']:.3f}"
              f"  spearman_max {r['spearman_max']:.3f}  enrich@10% {r['enrichment_top10pct']:.2f}x")
print("\n--- D. QN ablation (3 seeds; per-fold-mean RMSE) ---")
for key, d in D.items():
    print(f"{key:22s} RMSE {d['rmse_foldmean_mean']:.3f} +/- {d['rmse_foldmean_std']:.3f}"
          f"   spearman_max {d['spearman_max_mean']:.3f}")
log("done")
