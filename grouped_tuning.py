"""
grouped_tuning.py — Phase 1b: re-tune the LightGBM baseline under CATALYST-GROUPED CV.

The current hyper-parameters were inherited from the row-level (leaky) regime; unseen-catalyst
generalisation may prefer stronger regularisation. Random search (seeded, 30 configs incl. the
current default as config 0), each scored by grouped CV over 2 seeds; the best and the default are
then confirmed on 5 seeds with catalyst-level metrics.

Honesty notes:
- Search and confirmation share the same fold family (no nested CV) — the winner's confirmed number
  carries mild selection optimism. Fair comparisons downstream are unaffected: Phase-3 PFT variants
  will use the SAME tuned parameters in Stage 2, so both arms share whatever optimism exists.
- Protocol label for every number here: catalyst-grouped 5-fold CV, per-fold train-only scaler.

Output: grouped_tuning.json
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
from ocm_eval import Data, grouped_folds, run_cv, lgb_params

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
log(f"lab rows={len(d.y_lab)} catalysts={d.n_cat}")

SPACE = dict(
    n_estimators=[300, 500, 800, 1200],
    learning_rate=[0.02, 0.03, 0.05, 0.08],
    num_leaves=[15, 31, 63, 127],
    max_depth=[4, 5, 7, -1],
    min_child_samples=[20, 50, 100, 200],
    subsample=[0.6, 0.8, 1.0],
    colsample_bytree=[0.5, 0.7, 0.8, 1.0],
    reg_alpha=[0.0, 0.1, 1.0],
    reg_lambda=[0.1, 1.0, 5.0, 20.0],
)

rng = np.random.default_rng(7)
configs = [dict()]                                    # config 0 = current default (no overrides)
seen = {tuple(sorted(dict().items()))}
while len(configs) < 30:
    c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    key = tuple(sorted((k, float(v)) for k, v in c.items()))
    if key not in seen:
        seen.add(key)
        configs.append(c)

SEARCH_SEEDS = [0, 1]
results = []
for i, over in enumerate(configs):
    rs = [run_cv(d, grouped_folds(d, seed=s), s, 'baseline', lgbp=lgb_params(s, **over))
          for s in SEARCH_SEEDS]
    rmse = float(np.mean([r['rmse_foldmean'] for r in rs]))
    spear = float(np.mean([r['spearman_max'] for r in rs]))
    results.append({'config_id': i, 'overrides': {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                                  for k, v in over.items()},
                    'rmse_2seed': rmse, 'spearman_2seed': spear})
    log(f"cfg {i:2d} rmse={rmse:.3f} spearman={spear:.3f} {over if over else '(current default)'}")

results.sort(key=lambda r: r['rmse_2seed'])
best = results[0]
default_row = next(r for r in results if r['config_id'] == 0)
log(f"best cfg {best['config_id']} rmse_2seed={best['rmse_2seed']:.3f} vs default {default_row['rmse_2seed']:.3f}")

# ---- confirmation: best vs default, 5 seeds, full metrics ----
CONF_SEEDS = [0, 1, 2, 7, 13]
conf = {}
for label, over in [('default', {}), ('tuned', best['overrides'])]:
    rs = [run_cv(d, grouped_folds(d, seed=s), s, 'baseline', lgbp=lgb_params(s, **over))
          for s in CONF_SEEDS]
    conf[label] = {
        'overrides': over,
        'rmse_foldmean_mean': float(np.mean([r['rmse_foldmean'] for r in rs])),
        'rmse_foldmean_std': float(np.std([r['rmse_foldmean'] for r in rs], ddof=1)),
        'spearman_max_mean': float(np.mean([r['spearman_max'] for r in rs])),
        'pearson_max_mean': float(np.mean([r['pearson_max'] for r in rs])),
        'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in rs])),
        'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in rs])),
        'per_seed': rs}
    log(f"confirm {label}: RMSE {conf[label]['rmse_foldmean_mean']:.3f} +/- {conf[label]['rmse_foldmean_std']:.3f}"
        f"  spearman {conf[label]['spearman_max_mean']:.3f}  enrich {conf[label]['enrichment_mean']:.2f}x")

json.dump({'meta': {'protocol': 'catalyst-grouped 5-fold CV, per-fold train-only scaler',
                    'search_seeds': SEARCH_SEEDS, 'confirm_seeds': CONF_SEEDS,
                    'note': 'search/confirm share folds (no nested CV) — winner carries mild selection optimism'},
           'search': results, 'confirmation': conf},
          open('grouped_tuning.json', 'w'), indent=1)
log("wrote grouped_tuning.json")

b0 = conf['default']['rmse_foldmean_mean']; b1 = conf['tuned']['rmse_foldmean_mean']
print("\n================= SUMMARY (catalyst-grouped protocol) =================")
print(f"default params : RMSE {b0:.3f} +/- {conf['default']['rmse_foldmean_std']:.3f}   "
      f"spearman {conf['default']['spearman_max_mean']:.3f}   enrich {conf['default']['enrichment_mean']:.2f}x")
print(f"tuned params   : RMSE {b1:.3f} +/- {conf['tuned']['rmse_foldmean_std']:.3f}   "
      f"spearman {conf['tuned']['spearman_max_mean']:.3f}   enrich {conf['tuned']['enrichment_mean']:.2f}x")
print(f"tuned vs default: {100*(b1-b0)/b0:+.1f}% RMSE   overrides: {best['overrides']}")
log("done")
