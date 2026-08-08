"""
phase5_target_audit.py — P1: is our primary target (per-catalyst MAX yield) trustworthy?

Motivation: Spearman(n_measurements, observed_max) = 0.293. The 47 catalysts with <20 measurements
average an observed max of 3.28 vs 10.86 for the other 870, and NONE reach the global top decile.
max-of-n is a biased estimator (it grows with n), so part of our target may reflect measurement
effort rather than chemistry. Everything in Phases 2-4 sits on this target, so it is audited before
anything is built on top of it.

  A. Confound quantification — n vs observed max, by count bracket.
  B. Sampling-artifact simulation — take high-count catalysts, subsample to n in {1,5,10,20,50},
     recompute max. Isolates the PURE sampling effect with chemistry held fixed.
  C. Artifact vs selection — logistic classifier on composition -> low/high count. If compositions
     are indistinguishable (AUC ~0.5) the low maxima are a sampling artifact; if separable, the
     lab was selecting chemically distinct (probably poor) catalysts for early stopping.
  D. Sensitivity — headline V0 metrics under 3 target definitions (max / 90th pct / mean-of-top-5)
     x 2 catalyst sets (all 917 / exclude n<20), grouped protocol, 5 seeds.
  E. Bootstrap CIs on enrichment@10% and precision@20 (currently ~18 catalysts per fold - coarse).
  F. Nested-CV check — tuned params were selected on grouped CV and reported on grouped CV. Hold out
     20% of catalysts, tune-free evaluation, compare to the reported number.

Output: phase5_target_audit.json
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from ocm_eval import Data, lgb_params, cat_metrics, TARGET

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']

grp = d.dl_lab.groupby(d.groups)
counts = grp.size().values
Xc = grp.agg(**{c: (c, 'first') for c in el_cols}).values.astype(float)
y_max = grp[TARGET].max().values
y_p90 = grp[TARGET].quantile(0.90).values
y_top5 = grp[TARGET].apply(lambda s: s.nlargest(min(5, len(s))).mean()).values
n_cat = len(y_max)
LOW = counts < 20
log(f"{n_cat} catalysts; {LOW.sum()} with <20 measurements")

RES = {'meta': {'n_catalysts': int(n_cat), 'n_low_count': int(LOW.sum()),
                'protocol': 'catalyst-grouped 5-fold CV, formulation B, tuned LGBM'}}

# ---------------------------------------------------------------- A. confound
RES['A_confound'] = {
    'spearman_n_vs_max': float(spearmanr(counts, y_max)[0]),
    'spearman_n_vs_p90': float(spearmanr(counts, y_p90)[0]),
    'low_count_mean_max': float(y_max[LOW].mean()), 'high_count_mean_max': float(y_max[~LOW].mean()),
    'low_count_in_global_top10pct': int((y_max[LOW] >= np.quantile(y_max, 0.9)).sum())}
log(f"A: spearman(n,max)={RES['A_confound']['spearman_n_vs_max']:.3f}  "
    f"low={RES['A_confound']['low_count_mean_max']:.2f} high={RES['A_confound']['high_count_mean_max']:.2f}")

# ---------------------------------------------------------------- B. sampling simulation
# Chemistry held fixed: subsample rows of well-measured catalysts, see how observed max decays.
rich = np.where(counts >= 100)[0]
ylists = {g: v[TARGET].values for g, v in d.dl_lab.groupby(d.groups) if len(v) >= 100}
B = {}
rng = np.random.default_rng(0)
for n_sub in [1, 5, 10, 20, 50]:
    drops = []
    for g in rich:
        ys = ylists[g]
        sub_max = [rng.choice(ys, size=n_sub, replace=False).max() for _ in range(50)]
        drops.append(np.mean(sub_max) - ys.max())
    B[f'n={n_sub}'] = {'mean_max_deficit': float(np.mean(drops)), 'std': float(np.std(drops))}
    log(f"B subsample to n={n_sub:2d}: observed max is {np.mean(drops):+.2f} vs full-measurement max")
RES['B_sampling_simulation'] = B

# ---------------------------------------------------------------- C. artifact vs selection
sc_all = StandardScaler().fit(Xc)
auc = cross_val_score(LogisticRegression(max_iter=2000, class_weight='balanced'),
                      sc_all.transform(Xc), LOW.astype(int), cv=5, scoring='roc_auc')
RES['C_artifact_vs_selection'] = {'auc_composition_predicts_lowcount_mean': float(auc.mean()),
                                  'auc_std': float(auc.std()), 'per_fold': [float(a) for a in auc]}
log(f"C: AUC(composition -> low-count) = {auc.mean():.3f} +/- {auc.std():.3f} "
    f"({'chemically distinct -> selection effect' if auc.mean() > 0.7 else 'indistinguishable -> sampling artifact'})")

# ---------------------------------------------------------------- D. sensitivity
def folds_for(n, seed, n_folds=5):
    r = np.random.default_rng(seed); perm = r.permutation(n)
    fog = np.empty(n, dtype=int)
    for k, ch in enumerate(np.array_split(perm, n_folds)):
        fog[ch] = k
    return fog

def eval_target(X, y, seed):
    fog = folds_for(len(y), seed)
    yp = np.empty(len(y))
    for k in range(5):
        tr, va = np.where(fog != k)[0], np.where(fog == k)[0]
        s = StandardScaler().fit(X[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(s.transform(X[tr]), y[tr])
        yp[va] = m.predict(s.transform(X[va]))
    return cat_metrics(np.arange(len(y)), y, yp), yp

SEEDS = [0, 1, 2, 7, 13]
D_res = {}
for tname, tvals in [('max', y_max), ('p90', y_p90), ('top5mean', y_top5)]:
    for sname, keep in [('all_917', np.ones(n_cat, bool)), ('excl_lowcount', ~LOW)]:
        runs = [eval_target(Xc[keep], tvals[keep], s)[0] for s in SEEDS]
        D_res[f'{tname}/{sname}'] = {
            'n': int(keep.sum()),
            'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
            'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
            'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
            'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in runs]))}
        log(f"D {tname:9s}/{sname:14s} n={keep.sum():3d}  spearman="
            f"{D_res[f'{tname}/{sname}']['spearman_mean']:.3f}  "
            f"enrich={D_res[f'{tname}/{sname}']['enrichment_mean']:.2f}x")
RES['D_sensitivity'] = D_res

# ---------------------------------------------------------------- E. bootstrap CIs
_, yp_ref = eval_target(Xc, y_max, 0)
rngb = np.random.default_rng(42)
boot_e, boot_p, boot_s = [], [], []
for _ in range(1000):
    idx = rngb.choice(n_cat, n_cat, replace=True)
    r = cat_metrics(np.arange(len(idx)), y_max[idx], yp_ref[idx])
    boot_e.append(r['enrichment_top10pct']); boot_p.append(r['precision_at20_vs_top10pct'])
    boot_s.append(r['spearman_max'])
RES['E_bootstrap_ci'] = {
    'spearman_ci95': [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
    'enrichment_ci95': [float(np.percentile(boot_e, 2.5)), float(np.percentile(boot_e, 97.5))],
    'precision_at20_ci95': [float(np.percentile(boot_p, 2.5)), float(np.percentile(boot_p, 97.5))]}
log(f"E: spearman 95% CI {RES['E_bootstrap_ci']['spearman_ci95']}, "
    f"enrichment {RES['E_bootstrap_ci']['enrichment_ci95']}")

# ---------------------------------------------------------------- F. nested-CV / tuning optimism
F = {}
for seed in SEEDS:
    r = np.random.default_rng(1000 + seed)
    perm = r.permutation(n_cat); cut = int(0.2 * n_cat)
    te, tr = perm[:cut], perm[cut:]
    s = StandardScaler().fit(Xc[tr])
    for label, params in [('tuned', lgb_params(seed, **TUNED)), ('default', lgb_params(seed))]:
        m = lgb.LGBMRegressor(**params).fit(s.transform(Xc[tr]), y_max[tr])
        rr = cat_metrics(np.arange(len(te)), y_max[te], m.predict(s.transform(Xc[te])))
        F.setdefault(label, []).append(rr['spearman_max'])
RES['F_holdout_check'] = {k: {'spearman_mean': float(np.mean(v)), 'spearman_std': float(np.std(v, ddof=1))}
                          for k, v in F.items()}
log(f"F holdout: tuned={RES['F_holdout_check']['tuned']['spearman_mean']:.3f} "
    f"default={RES['F_holdout_check']['default']['spearman_mean']:.3f}")

json.dump(RES, open('phase5_target_audit.json', 'w'), indent=1)
log("wrote phase5_target_audit.json")

print("\n================= AUDIT SUMMARY =================")
print(f"A. Spearman(n_measurements, observed_max) = {RES['A_confound']['spearman_n_vs_max']:.3f}")
print(f"B. Pure sampling effect (chemistry fixed): subsampling a well-measured catalyst to")
for k, v in B.items():
    print(f"     {k:5s} -> observed max {v['mean_max_deficit']:+.2f} yield points")
print(f"C. AUC(composition -> low-count) = {RES['C_artifact_vs_selection']['auc_composition_predicts_lowcount_mean']:.3f}")
print("D. Headline metric sensitivity:")
for k, v in D_res.items():
    print(f"     {k:24s} n={v['n']:3d}  spearman {v['spearman_mean']:.3f}  enrich {v['enrichment_mean']:.2f}x"
          f"  prec@20 {v['precision_at20_mean']:.2f}")
print(f"E. 95% CIs (n=917): spearman {RES['E_bootstrap_ci']['spearman_ci95'][0]:.3f}-{RES['E_bootstrap_ci']['spearman_ci95'][1]:.3f}"
      f"  enrichment {RES['E_bootstrap_ci']['enrichment_ci95'][0]:.2f}-{RES['E_bootstrap_ci']['enrichment_ci95'][1]:.2f}x")
print(f"F. True-holdout spearman: tuned {RES['F_holdout_check']['tuned']['spearman_mean']:.3f}"
      f" vs default {RES['F_holdout_check']['default']['spearman_mean']:.3f}")
log("done")
