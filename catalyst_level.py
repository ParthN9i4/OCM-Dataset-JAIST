"""
catalyst_level.py — Phase 2: is a DIRECT catalyst-level max-yield model better than
predicting rows and aggregating?

The screening objective (per Prof. Taniike) is per-catalyst: will an unseen catalyst reach a high
maximum yield somewhere in the standard condition battery? Three formulations, all evaluated on the
SAME catalyst-fold assignments (per seed) so the comparison is apples-to-apples:

  A  row-level model (tuned params) -> predict all ~97 rows/catalyst -> take max per catalyst.
  B  direct model: one row per catalyst (prep + 65 element loadings; temperature dropped),
     target = max observed yield across all that catalyst's rows. 917 training examples.
  B2 middle ground: one row per (catalyst, temperature) [max over the ~20 rows of that cell,
     which are different unrecorded reaction-condition settings, not replicates],
     model keeps temperature, predict -> max over the 5 temperatures per catalyst.

Metrics (catalyst level, primary): Spearman/Pearson of predicted vs observed max yield,
enrichment@top-10%, precision@20. Protocol: catalyst-grouped 5-fold CV, 5 seeds, tuned LGBM params
from grouped_tuning.json (B/B2 also report default params — tiny-sample behaviour may differ).

Output: catalyst_level.json
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from ocm_eval import Data, lgb_params, run_fold, cat_metrics

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
log(f"lab rows={len(d.y_lab)} catalysts={d.n_cat}; tuned overrides loaded")

# ---- catalyst-level tables ----
el_cols = [c for c in d.features if c not in ('Temperature_C',)]          # prep_enc + 65 elements
cat_df = d.dl_lab.groupby(d.groups).agg(**{c: (c, 'first') for c in el_cols},
                                        y_max=('Y(C2), %', 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values     # 917 x 66
ct_df = d.dl_lab.groupby([d.groups, 'Temperature_C']).agg(
    **{c: (c, 'first') for c in el_cols}, y_max=('Y(C2), %', 'max')).reset_index(level=1)
ct_groups = ct_df.index.values                                             # catalyst id per (cat,temp) row
Xct = ct_df[['Temperature_C'] + el_cols].values.astype(float); yct = ct_df['y_max'].values
log(f"catalyst table {Xc.shape}; catalyst-temperature table {Xct.shape}")


def fold_assignment(seed, n_folds=5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d.n_cat)
    fog = np.empty(d.n_cat, dtype=int)
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        fog[chunk] = k
    return fog                                                             # fold of each catalyst id


def eval_A(seed, fog):
    """Row-level tuned model, aggregate max per catalyst (same fold assignment)."""
    fold = fog[d.groups]
    yp_all = np.empty(len(d.y_lab))
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        yp_all[va] = run_fold(d, tr, va, seed, 'baseline', lgbp=lgb_params(seed, **TUNED))
    return cat_metrics(d.groups, d.y_lab, yp_all)


def eval_B(seed, fog, params):
    """Direct per-catalyst max-yield model (no temperature)."""
    fold_c = fog[np.arange(d.n_cat)]
    yp = np.empty(d.n_cat)
    for k in range(5):
        tr, va = np.where(fold_c != k)[0], np.where(fold_c == k)[0]
        m = lgb.LGBMRegressor(**params).fit(Xc[tr], yc[tr])
        yp[va] = m.predict(Xc[va])
    return cat_metrics(np.arange(d.n_cat), yc, yp)


def eval_B2(seed, fog, params):
    """Per (catalyst, temperature) max model, then max over temperatures."""
    fold_ct = fog[ct_groups]
    yp = np.empty(len(yct))
    for k in range(5):
        tr, va = np.where(fold_ct != k)[0], np.where(fold_ct == k)[0]
        m = lgb.LGBMRegressor(**params).fit(Xct[tr], yct[tr])
        yp[va] = m.predict(Xct[va])
    return cat_metrics(ct_groups, yct, yp)


SEEDS = [0, 1, 2, 7, 13]
RES = {}
for name, fn in [('A_row_aggregate_tuned', lambda s, f: eval_A(s, f)),
                 ('B_direct_tuned', lambda s, f: eval_B(s, f, lgb_params(s, **TUNED))),
                 ('B_direct_default', lambda s, f: eval_B(s, f, lgb_params(s))),
                 ('B2_cat_temp_tuned', lambda s, f: eval_B2(s, f, lgb_params(s, **TUNED))),
                 ('B2_cat_temp_default', lambda s, f: eval_B2(s, f, lgb_params(s)))]:
    runs = []
    for s in SEEDS:
        r = fn(s, fold_assignment(s))
        runs.append(r)
        log(f"{name:22s} seed={s:2d}  spearman={r['spearman_max']:.3f}  pearson={r['pearson_max']:.3f}"
            f"  enrich={r['enrichment_top10pct']:.2f}x  prec@20={r['precision_at20_vs_top10pct']:.2f}")
    RES[name] = {'per_seed': runs,
                 'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
                 'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
                 'pearson_mean': float(np.mean([r['pearson_max'] for r in runs])),
                 'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
                 'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in runs]))}

json.dump({'meta': {'protocol': 'catalyst-grouped 5-fold CV, 5 seeds, identical fold assignments across formulations',
                    'tuned_overrides': TUNED, 'seeds': SEEDS},
           'results': RES}, open('catalyst_level.json', 'w'), indent=1)
log("wrote catalyst_level.json")

print("\n================= SUMMARY (catalyst-grouped, 5 seeds) =================")
for name, r in RES.items():
    print(f"{name:22s} spearman {r['spearman_mean']:.3f} +/- {r['spearman_std']:.3f}"
          f"   pearson {r['pearson_mean']:.3f}   enrich {r['enrichment_mean']:.2f}x"
          f"   prec@20 {r['precision_at20_mean']:.2f}")
log("done")
