"""
phase8_denoised_target.py — does denoising the training TARGET with the five-number summary
of repeated measurements change what the model learns?

USER'S IDEA: each (catalyst, temperature) cell has ~10-25 repeat measurements that share the
EXACT same recorded inputs (composition + temperature) but different yields, because ~27
reaction-condition settings are not recorded as features (see phase5_target_audit.py). Instead of
using every raw repeat, or blindly taking the single max, compress each cell down to its
five-number summary (P0, P25, P50, P75, P100) and build the catalyst-level target from THAT.

WHY THIS IS WELL MOTIVATED (not just "using less data"): phase5_target_audit.py's sampling
simulation showed max-of-n is upward-biased purely from sampling noise -- a catalyst measured
once shows a max ~6 points below the same catalyst measured 100 times. Our current target,
y_max_raw, is exactly "the max of every noisy repeat across the whole catalyst" -- it inherits
that upward bias. Taking the MEDIAN within each (catalyst, temperature) cell first denoises the
per-condition estimate; taking the MAX of those denoised per-condition values across a catalyst's
temperatures then answers "what is the best ACHIEVABLE condition for this catalyst" without a
single lucky noisy reading being able to set the target.

FOUR TARGETS COMPARED (917 catalysts, identical folds/seeds throughout):
  A. y_max_raw            <- CURRENT/CONTROL: max over every raw row of the catalyst
  B. y_median_raw          <- median over every raw row of the catalyst (the literal "use median" idea)
  C. y_max_of_cell_medians <- for each (catalyst,temp) cell take the median (denoise), then max across
                              the catalyst's cells (the five-number-summary idea, using P50 per cell)
  D. y_max_of_cell_p75     <- same, but P75 per cell instead of median (a lighter touch than C)

CRITICAL METHODOLOGY (same trap phase5 already caught once): a target that looks good against ITS
OWN definition is not evidence of anything -- p90/top5mean looked better than max against
themselves and collapsed to noise when scored against the real objective. So every variant here is
trained on ITS OWN target, but every variant is SCORED against the same thing: the catalyst's TRUE
observed max yield (the actual screening objective). If a denoised target does not improve ranking
against true max, it does not help, no matter how much "more information" it uses.

Output: phase8_denoised_target.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, cat_metrics, TARGET

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']

lab = d.dl_lab.copy()
lab['cat_id'] = d.groups
n_cat = d.n_cat

# ---------------------------------------------------------------- per-cell five-number summary
cell = lab.groupby(['cat_id', 'Temperature_C'])[TARGET]
cell_stats = cell.agg(n='size', p0='min', p25=lambda s: s.quantile(.25),
                      p50='median', p75=lambda s: s.quantile(.75), p100='max').reset_index()
log(f"(catalyst, temperature) cells: {len(cell_stats):,}  |  mean rows/cell: {cell_stats.n.mean():.1f}")

# ---------------------------------------------------------------- catalyst-level tables
Xc_df = lab.groupby('cat_id')[el_cols].first()
Xc = Xc_df.values.astype(float)

y_max_raw = lab.groupby('cat_id')[TARGET].max().values                    # A: current/control
y_median_raw = lab.groupby('cat_id')[TARGET].median().values               # B: literal median idea

by_cat = cell_stats.groupby('cat_id')
y_max_of_cell_medians = by_cat['p50'].max().values                         # C: denoise then max
y_max_of_cell_p75     = by_cat['p75'].max().values                         # D: lighter denoise

TRUE_MAX = y_max_raw   # the real screening objective every variant is judged against

log(f"catalysts: {n_cat}")
log(f"target A (max, current)          mean={y_max_raw.mean():.2f}")
log(f"target B (median of raw rows)    mean={y_median_raw.mean():.2f}")
log(f"target C (max of cell medians)   mean={y_max_of_cell_medians.mean():.2f}")
log(f"target D (max of cell P75s)      mean={y_max_of_cell_p75.mean():.2f}")

# ---------------------------------------------------------------- catalyst-grouped CV
def fold_assignment(seed, n=n_cat, k=5):
    r = np.random.default_rng(seed); perm = r.permutation(n); f = np.empty(n, int)
    for i, ch in enumerate(np.array_split(perm, k)): f[ch] = i
    return f

def eval_target(y_train, seed):
    """Train on y_train under grouped CV; ALWAYS score predictions against TRUE_MAX."""
    f = fold_assignment(seed)
    yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), y_train[tr])
        yp[va] = m.predict(sc.transform(Xc[va]))
    return cat_metrics(np.arange(n_cat), TRUE_MAX, yp)   # <- scored against TRUE_MAX, not y_train

SEEDS = [0, 1, 2, 7, 13]
VARIANTS = [('A_max_raw_CONTROL', y_max_raw), ('B_median_raw', y_median_raw),
            ('C_max_of_cell_medians', y_max_of_cell_medians), ('D_max_of_cell_p75', y_max_of_cell_p75)]

RES = {}
for name, ytr in VARIANTS:
    runs = [eval_target(ytr, s) for s in SEEDS]
    RES[name] = {
        'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
        'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
        'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
        'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in runs])),
        'per_seed_spearman': [float(r['spearman_max']) for r in runs]}
    log(f"{name:24s} spearman={RES[name]['spearman_mean']:.3f} +/- {RES[name]['spearman_std']:.3f}  "
        f"enrich={RES[name]['enrichment_mean']:.2f}x  prec@20={RES[name]['precision_at20_mean']:.2f}")

base = RES['A_max_raw_CONTROL']
for name, _ in VARIANTS[1:]:
    d_sp = RES[name]['spearman_mean'] - base['spearman_mean']
    wins = sum(a > b for a, b in zip(RES[name]['per_seed_spearman'], base['per_seed_spearman']))
    RES[name]['delta_spearman_vs_control'] = float(d_sp)
    RES[name]['seed_wins_vs_control'] = int(wins)

json.dump({'meta': {'n_cells': int(len(cell_stats)), 'mean_rows_per_cell': float(cell_stats.n.mean()),
                    'seeds': SEEDS, 'scored_against': 'TRUE observed max (y_max_raw) in every case'},
           'results': RES}, open('phase8_denoised_target.json', 'w'), indent=1)
log("wrote phase8_denoised_target.json")

print("\n================= SUMMARY (all scored against TRUE max) =================")
print(f"{'target':26s} {'Spearman':>16s} {'enrich':>8s} {'prec@20':>8s} {'delta':>8s} {'wins':>6s}")
for name, _ in VARIANTS:
    r = RES[name]
    extra = f"{r.get('delta_spearman_vs_control',0):+8.3f} {r.get('seed_wins_vs_control','-'):>5}/5" if name != 'A_max_raw_CONTROL' else " "*15
    print(f"{name:26s} {r['spearman_mean']:.3f} +/- {r['spearman_std']:.3f}  {r['enrichment_mean']:6.2f}x {r['precision_at20_mean']:8.2f} {extra}")
log("done")
