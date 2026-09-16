"""
phase10_condition_subsampling.py — how much of the 135-condition grid does a catalyst actually
need before its ranking is well estimated?

MOTIVATION. Each catalyst is run at 5 temperatures under ~27 unrecorded condition settings per
temperature (phase8_target_robustness.py). If most of those repeats are redundant for RANKING
purposes -- as opposed to for estimating the exact yield -- a screening campaign could measure far
fewer conditions per candidate and test more candidates for the same reactor time.

METHOD. Subsample the EXISTING data down to k measurements per (catalyst, temperature) cell (or
down to a subset of temperatures), rebuild the per-catalyst MAX label from only that subset, retrain
under catalyst-grouped CV, and score every variant against the TRUE observed max computed from the
FULL 89,074-row dataset -- never against the reduced variant's own label. This is the same rule
phase8 and phase9 followed: a target that looks good against itself is not evidence of anything.

Because a single random subsample could be a lucky draw, every k is repeated across N_DRAWS
independent random subsamples (in addition to the 5 CV seeds), and both the across-draw and
across-seed spread are reported.

Output: phase10_condition_subsampling.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from ocm_eval import Data, lgb_params, cat_metrics, TARGET

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SEEDS = [0, 1, 2, 7, 13]
N_DRAWS = 5

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']
lab = d.dl_lab.copy(); lab['cat_id'] = d.groups; n_cat = d.n_cat
Xc = lab.groupby('cat_id')[el_cols].first().values.astype(float)
TRUE_MAX = lab.groupby('cat_id')[TARGET].max().values   # ground truth: full 89,074 rows, never varies
cell_index = lab.groupby(['cat_id', 'Temperature_C']).indices  # {(cat,T): row positions}


def fold_assignment(seed):
    r = np.random.default_rng(seed); perm = r.permutation(n_cat); f = np.empty(n_cat, int)
    for i, ch in enumerate(np.array_split(perm, 5)): f[ch] = i
    return f


def eval_label(y, seed):
    f = fold_assignment(seed); yp = np.full(n_cat, np.nan); ok = ~np.isnan(y)
    for k in range(5):
        tr = np.where((f != k) & ok)[0]; va = np.where(f == k)[0]
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xc[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), y[tr])
        yp[va] = m.predict(sc.transform(Xc[va]))
    return yp


def label_from_subset(idx):
    sub = lab.loc[idx]
    return sub.groupby('cat_id')[TARGET].max().reindex(range(n_cat)).values


def score(y, n_meas):
    preds = [eval_label(y, s) for s in SEEDS]
    ms = [cat_metrics(np.arange(n_cat), TRUE_MAX, p) for p in preds]
    bias = np.nanmean(y - TRUE_MAX)
    return {'n_measurements': int(n_meas), 'n_catalysts_covered': int((~np.isnan(y)).sum()),
            'spearman_mean': float(np.mean([m['spearman_max'] for m in ms])),
            'spearman_std_across_seeds': float(np.std([m['spearman_max'] for m in ms], ddof=1)),
            'enrichment_mean': float(np.mean([m['enrichment_top10pct'] for m in ms])),
            'precision_at20_mean': float(np.mean([m['precision_at20_vs_top10pct'] for m in ms])),
            'label_bias_vs_true_max': float(bias)}


# ============================================================ A. condition subsampling
log("A. condition subsampling: k measurements per (catalyst, temperature) cell")
A = {}
for k in [1, 2, 3, 5, 8, 13, 20, 27]:
    draws = []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(1000 + draw)
        idx = []
        for _, ii in cell_index.items():
            idx.extend(ii if len(ii) <= k else rng.choice(ii, k, replace=False))
        y = label_from_subset(idx)
        draws.append(score(y, len(idx)))
    A[str(k)] = {
        'n_measurements_mean': float(np.mean([r['n_measurements'] for r in draws])),
        'spearman_mean_across_draws': float(np.mean([r['spearman_mean'] for r in draws])),
        'spearman_std_across_draws': float(np.std([r['spearman_mean'] for r in draws], ddof=1)) if N_DRAWS > 1 else 0.0,
        'enrichment_mean_across_draws': float(np.mean([r['enrichment_mean'] for r in draws])),
        'label_bias_mean_across_draws': float(np.mean([r['label_bias_vs_true_max'] for r in draws])),
        'n_draws': N_DRAWS}
    log(f"  k={k:2d}  n_meas~{A[str(k)]['n_measurements_mean']:6.0f}  "
        f"spearman={A[str(k)]['spearman_mean_across_draws']:.4f} +/- {A[str(k)]['spearman_std_across_draws']:.4f} "
        f"(across {N_DRAWS} draws)  enrich={A[str(k)]['enrichment_mean_across_draws']:.2f}x  "
        f"bias={A[str(k)]['label_bias_mean_across_draws']:+.2f}")

# ============================================================ B. temperature ablation
log("B. temperature ablation")
TEMP_SETS = [([700.0], '700_only'), ([750.0], '750_only'), ([800.0], '800_only'),
             ([850.0], '850_only'), ([900.0], '900_only'),
             ([700.0, 750.0], '700_750'), ([800.0, 850.0], '800_850'),
             ([750.0, 800.0, 850.0], '750_800_850'),
             ([700.0, 750.0, 800.0, 850.0, 900.0], 'all_five')]
B = {}
for temps, name in TEMP_SETS:
    sub = lab[lab.Temperature_C.isin(temps)]
    y = sub.groupby('cat_id')[TARGET].max().reindex(range(n_cat)).values
    B[name] = score(y, len(sub))
    log(f"  {name:14s} n_meas={B[name]['n_measurements']:6d}  spearman={B[name]['spearman_mean']:.4f}  "
        f"enrich={B[name]['enrichment_mean']:.2f}x")

# ============================================================ C. where does the optimum occur?
log("C. temperature of each catalyst's true optimum")
imax = lab.loc[lab.groupby('cat_id')[TARGET].idxmax()]
vc = imax.Temperature_C.value_counts().sort_index()
C = {str(T): {'n_catalysts': int(c), 'fraction': float(c / n_cat)} for T, c in vc.items()}
for T, c in vc.items(): log(f"  {T:5.0f} C: {c:4d} catalysts ({100*c/n_cat:.1f}%)")

# ============================================================ D. why condition-axis compresses
between_sd = float(TRUE_MAX.std())
within_sd = float(lab.groupby(['cat_id', 'Temperature_C'])[TARGET].std().median())
D = {'between_catalyst_sd_of_true_max': between_sd, 'median_within_cell_sd': within_sd,
     'ratio': between_sd / within_sd}
log(f"D. between-catalyst SD {between_sd:.2f} / median within-cell SD {within_sd:.2f} "
    f"= {D['ratio']:.1f}x")

json.dump({'meta': {'protocol': 'formulation B, composition-only, tuned LGBM, catalyst-grouped 5-fold, 5 seeds',
                    'seeds': SEEDS, 'n_draws_per_k': N_DRAWS,
                    'scored_against': 'TRUE observed max from the FULL 89,074-row dataset, always',
                    'caveat': 'assumes the sampled conditions are individually selectable; depends on '
                              'whether the 27 slots per cell are distinct conditions or time-on-stream '
                              'samples (open question, see SESSION_CONTEXT.md)'},
           'A_condition_subsampling': A, 'B_temperature_ablation': B,
           'C_temperature_of_optimum': C, 'D_variance_ratio': D},
          open('phase10_condition_subsampling.json', 'w'), indent=1)
log("wrote phase10_condition_subsampling.json")
