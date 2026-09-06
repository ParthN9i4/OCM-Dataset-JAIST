"""
phase9_equal_effort_eval.py — how much of our screening score is chemistry, and how much is a record
of which experiments the lab chose to finish?

MOTIVATION. Grid coverage in this dataset is not random and it is coupled to performance. Each
catalyst-temperature cell is a designed grid of up to 27 unrecorded reaction-condition settings
(5 x 27 = 135 per catalyst). Cells that were run further contain better yields:
Spearman(cell size, cell max yield) = +0.441, and mean cell yield rises monotonically from 2.22 %
in cells with 1-5 rows to 6.05 % in cells with 27. Unpromising combinations appear to have been
abandoned partway. Because our target is the catalyst's observed maximum, a catalyst that was
abandoned early has both fewer draws AND was abandoned because it looked bad.

phase5 already established the confound exists (Spearman(n_rows, observed max) = +0.293) and showed
that excluding the 47 lowest-count catalysts barely moves the headline. This script asks the harder
question phase5 did not: is the HEADLINE ITSELF inflated, and by how much?

THREE INSTRUMENTS.

  1. EQUAL-EFFORT EVALUATION SET. Catalysts with >= MIN_CELL rows in at least one temperature cell.
     On that set the coupling is gone by measurement, not by assumption, so any score computed there
     cannot be riding on effort. Reported alongside the all-917 number as TWO LABELED QUANTITIES,
     never as a replacement.

  2. RANDOM-SUBSET CONTROL. The equal-effort set is smaller, and Spearman moves with sample size.
     So we draw many RANDOM subsets of exactly the same size from the SAME predictions and report
     their interval. Only if the equal-effort value falls outside that interval is the drop real.
     Without this control the whole comparison is uninterpretable.

  3. EFFORT-ONLY NEGATIVE CONTROL. Fit the identical model with n_rows as the TARGET -- it never
     sees a yield -- then rank catalysts by predicted effort and score that ranking against the
     observed max. This is "is your model just learning which experiments you finished?" made
     numeric. It is the sharpest single diagnostic in this script.

DELIBERATELY NOT DONE: relabeling the target to correct for coverage. Rarefaction to a common depth
and deficit-extrapolation to the full 27-point grid were both piloted; the corrected labels correlate
with the plain observed max at 0.977 and 0.9989 respectively, and every shortlist change they induce
sits inside seed noise. Coverage-weighted training also fails a catalyst-clustered bootstrap. The
correction belongs in the EVALUATION, not the label. phase8_target_robustness.py separately settled
that the label choice itself does not matter.

Protocol: formulation B, composition-only, tuned LGBM, catalyst-grouped 5-fold CV, train-only scaler,
5 seeds. Predictions are seed-averaged before scoring; note this differs from the per-seed-mean
convention used for the 0.761 headline elsewhere, and both are reported so they are never confused.
Output: phase9_equal_effort_eval.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, cat_metrics, TARGET

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

MIN_CELL = 20        # equal-effort threshold: >= this many rows in at least one temperature cell
N_SUBSET = 300       # random-subset control draws
N_BOOT = 3000        # catalyst-clustered bootstrap resamples
SEEDS = [0, 1, 2, 7, 13]

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']
lab = d.dl_lab.copy(); lab['cat_id'] = d.groups
n_cat = d.n_cat

Xc = lab.groupby('cat_id')[el_cols].first().values.astype(float)
y_max = lab.groupby('cat_id')[TARGET].max().values
n_rows = lab.groupby('cat_id').size().values
max_cell = lab.groupby(['cat_id', 'Temperature_C']).size().groupby(level=0).max().values

EE = max_cell >= MIN_CELL          # the equal-effort set
log(f"{n_cat} catalysts; equal-effort set (>={MIN_CELL} rows in >=1 cell): {EE.sum()}")


def oof(target, seed):
    """Out-of-fold predictions, catalyst-grouped 5-fold. Never trained on y_max unless asked."""
    r = np.random.default_rng(seed); perm = r.permutation(n_cat); f = np.empty(n_cat, int)
    for i, ch in enumerate(np.array_split(perm, 5)): f[ch] = i
    yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), target[tr])
        yp[va] = m.predict(sc.transform(Xc[va]))
    return yp


def score(mask, pred):
    """Catalyst-level metrics restricted to `mask`, always scored against the observed max."""
    m = cat_metrics(np.arange(int(mask.sum())), y_max[mask], pred[mask])
    return {'n': int(mask.sum()), 'spearman': float(m['spearman_max']),
            'enrichment': float(m['enrichment_top10pct']),
            'precision_at20': float(m['precision_at20_vs_top10pct'])}


# ------------------------------------------------------------------ models
per_seed_real = [oof(y_max, s) for s in SEEDS]
P = np.mean(per_seed_real, axis=0)                                  # real model, seed-averaged
E = np.mean([oof(n_rows.astype(float), s) for s in SEEDS], axis=0)  # EFFORT-ONLY control
log("fitted real model and effort-only control")

ALL = np.ones(n_cat, bool)
RES = {
    'real_model': {'all_917': score(ALL, P), 'equal_effort': score(EE, P)},
    'effort_only_control': {'all_917': score(ALL, E), 'equal_effort': score(EE, E)},
    'headline_conventions': {
        'per_seed_mean_spearman_all': float(np.mean([spearmanr(y_max, p)[0] for p in per_seed_real])),
        'seed_averaged_spearman_all': float(spearmanr(y_max, P)[0]),
        'note': 'these are two different quantities; 0.761 elsewhere is the per-seed mean'},
}

# ------------------------------------------------------------------ random-subset control
rng = np.random.default_rng(3)
vals = []
for _ in range(N_SUBSET):
    idx = rng.choice(n_cat, int(EE.sum()), replace=False)
    m = np.zeros(n_cat, bool); m[idx] = True
    vals.append(score(m, P)['spearman'])
vals = np.array(vals)
ee_sp = RES['real_model']['equal_effort']['spearman']
RES['random_subset_control'] = {
    'n_draws': N_SUBSET, 'subset_size': int(EE.sum()),
    'spearman_mean': float(vals.mean()),
    'ci95_low': float(np.percentile(vals, 2.5)), 'ci95_high': float(np.percentile(vals, 97.5)),
    'equal_effort_spearman': float(ee_sp),
    'equal_effort_below_interval': bool(ee_sp < np.percentile(vals, 2.5)),
    'interpretation': ('if equal_effort_below_interval is true the drop is a real coverage effect, '
                       'not an artifact of evaluating on fewer catalysts')}

# ------------------------------------------------------------------ the coupling itself
RES['coupling'] = {
    'spearman_nrows_vs_observed_max_all': float(spearmanr(n_rows, y_max)[0]),
    'spearman_nrows_vs_observed_max_equal_effort': float(spearmanr(n_rows[EE], y_max[EE])[0]),
    'spearman_prediction_vs_nrows_all': float(spearmanr(P, n_rows)[0])}

# ------------------------------------------------------------------ coverage-stratified
q = pd.qcut(pd.Series(n_rows), 4, labels=['Q1_fewest', 'Q2', 'Q3', 'Q4_most'])
RES['by_coverage_quartile'] = {}
for lvl in q.cat.categories:
    m = (q == lvl).values
    s = score(m, P); s['mean_n_rows'] = float(n_rows[m].mean()); s['label_sd'] = float(y_max[m].std())
    RES['by_coverage_quartile'][lvl] = s

# ------------------------------------------------------------------ the campaign regime
RES['restricted_to_model_top_k'] = {}
for K in [20, 50, 100, 150, 300]:
    top = np.argsort(-P)[:K]
    RES['restricted_to_model_top_k'][f'top_{K}'] = {
        'spearman_within': float(spearmanr(y_max[top], P[top])[0]),
        'mean_observed_max': float(y_max[top].mean())}
RES['restricted_to_model_top_k']['note'] = (
    'a synthesis campaign only ever operates inside this regime; the library-wide Spearman is not '
    'the number that governs whether a campaign of a given size can detect anything')

# ------------------------------------------------------------------ clustered bootstrap
bidx = [rng.integers(0, n_cat, n_cat) for _ in range(N_BOOT)]
def boot_ci(mask, pred):
    sub = np.where(mask)[0]; pos = {c: i for i, c in enumerate(sub)}
    out = []
    for b in bidx:
        bb = [c for c in b if c in pos]
        if len(bb) > 30:
            out.append(spearmanr(y_max[bb], pred[bb])[0])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
for nm, msk in [('all_917', ALL), ('equal_effort', EE)]:
    lo, hi = boot_ci(msk, P)
    RES['real_model'][nm]['spearman_ci95'] = [lo, hi]
log("bootstrap done")

RES['verdict'] = {
    'headline_should_be_reported_as': (
        f"Spearman {RES['real_model']['all_917']['spearman']:.3f} across all 917 catalysts, and "
        f"{RES['real_model']['equal_effort']['spearman']:.3f} on the {int(EE.sum())} whose measurement "
        f"effort is comparable; enrichment@10% "
        f"{RES['real_model']['all_917']['enrichment']:.2f}x and "
        f"{RES['real_model']['equal_effort']['enrichment']:.2f}x respectively"),
    'why_the_effort_control_matters': (
        f"a model trained only to predict how many measurements a catalyst received -- it never sees a "
        f"yield -- reaches Spearman {RES['effort_only_control']['all_917']['spearman']:.3f} against the "
        f"observed max, but its enrichment is only "
        f"{RES['effort_only_control']['all_917']['enrichment']:.2f}x. Rank correlation is partly "
        f"purchasable from experimental effort; enrichment is not. This is why enrichment stays primary."),
    'not_done': ('target relabeling for coverage (rarefaction, deficit-extrapolation, coverage-weighted '
                 'training) -- all piloted, all null; the correction belongs in the evaluation'),
}

json.dump({'meta': {'protocol': 'formulation B, composition-only, tuned LGBM, catalyst-grouped 5-fold',
                    'seeds': SEEDS, 'min_cell_for_equal_effort': MIN_CELL,
                    'n_random_subset_draws': N_SUBSET, 'n_bootstrap': N_BOOT,
                    'predictions': 'seed-averaged before scoring'},
           'results': RES}, open('phase9_equal_effort_eval.json', 'w'), indent=1)
log("wrote phase9_equal_effort_eval.json")

print("\n================= COVERAGE-CORRECTED REPORTING =================")
print(f"{'':34s} {'n':>5s} {'Spearman':>10s} {'enrich':>9s} {'prec@20':>8s}")
for lbl, key in [('REAL model, all catalysts', 'all_917'), ('REAL model, equal-effort set', 'equal_effort')]:
    r = RES['real_model'][key]
    print(f"{lbl:34s} {r['n']:5d} {r['spearman']:10.4f} {r['enrichment']:8.2f}x {r['precision_at20']:8.2f}")
for lbl, key in [('EFFORT-ONLY control, all', 'all_917'), ('EFFORT-ONLY control, equal-effort', 'equal_effort')]:
    r = RES['effort_only_control'][key]
    print(f"{lbl:34s} {r['n']:5d} {r['spearman']:10.4f} {r['enrichment']:8.2f}x {r['precision_at20']:8.2f}")

c = RES['random_subset_control']
print(f"\nRANDOM-SUBSET CONTROL ({c['n_draws']} draws of {c['subset_size']} from the SAME predictions):")
print(f"   Spearman {c['spearman_mean']:.4f}  95% range [{c['ci95_low']:.4f}, {c['ci95_high']:.4f}]")
print(f"   equal-effort value {c['equal_effort_spearman']:.4f} -> "
      f"{'BELOW the interval: the drop is REAL' if c['equal_effort_below_interval'] else 'inside: could be a size artifact'}")

k = RES['coupling']
print(f"\nCOUPLING  Spearman(n_rows, observed max): all {k['spearman_nrows_vs_observed_max_all']:+.4f}"
      f"  ->  equal-effort {k['spearman_nrows_vs_observed_max_equal_effort']:+.4f}")

print("\nBY COVERAGE QUARTILE")
for lvl, r in RES['by_coverage_quartile'].items():
    print(f"   {lvl:10s} n={r['n']:3d} mean_rows={r['mean_n_rows']:6.1f} "
          f"spearman={r['spearman']:.3f} enrich={r['enrichment']:.2f}x label_sd={r['label_sd']:.2f}")

print("\nINSIDE THE MODEL'S OWN TOP-K (the regime a synthesis campaign lives in)")
for kk in [20, 50, 100, 150, 300]:
    r = RES['restricted_to_model_top_k'][f'top_{kk}']
    print(f"   top-{kk:<4d} spearman_within={r['spearman_within']:+.3f}  mean observed max={r['mean_observed_max']:.2f}%")

print(f"\nREPORT AS: {RES['verdict']['headline_should_be_reported_as']}")
print(f"\nEFFORT CONTROL: {RES['verdict']['why_the_effort_control_matters']}")
log("done")
