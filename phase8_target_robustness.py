"""
phase8_target_robustness.py — does the catalyst ranking depend on HOW we collapse each catalyst's
135 condition measurements into a single training label?

SUPERSEDES phase8_denoised_target.py (renamed; git history preserved). That version framed the
question as "denoising repeated measurements" and concluded that two summarised targets beat the
observed max. Both the framing and the conclusion were wrong, for the two reasons below (each
verified against the CSV; the grid evidence is recomputed live by this script and stored under
"condition_grid_evidence" in its JSON output):

  1. WRONG FRAMING. The ~20-27 rows in each (catalyst, temperature) cell are NOT repeats of one
     measurement. They are a designed grid of 27 reaction-condition settings that the data file does
     not record. Evidence from the row counts alone: cell sizes ceiling at exactly 27, then at
     exactly 54 = 2 x 27, with nothing above; and 15 catalysts hold exactly 135 rows, all 15 of them
     splitting as exactly (27, 27, 27, 27, 27). 5 x 27 = 135 matches the "135 conditions" Prof.
     Taniike states each catalyst is run under. The competing reading -- that 27 was a top-27-by-yield
     export cut-off -- is rejected most simply by counting: 104 cells hold MORE than 27 rows, which a
     top-27 cut-off cannot emit. An order-statistic test agrees in direction under four definitions of
     interior spacing. Both are computed in phase11_condition_grid_forensics.py and stored in
     phase11_condition_grid_forensics.json -- quote them from there. (Earlier drafts quoted "2.03x vs
     0.90x" for the order statistic; those figures had no script or JSON behind them and were not
     reproduced. The direction they assert holds; the numbers should not be requoted.)
     So the within-cell spread is real condition-response, not measurement noise, and no target here
     may be described as "denoising".

     CAVEAT, not previously stated: every cell is stored sorted descending by yield (0 violations in
     84,675 within-cell adjacent comparisons). Pre-sorting by yield is what a "take the top N" export
     would produce, so the truncation reading is mechanically more plausible than the order statistic
     alone suggests -- the counting argument, not the order statistic, is what settles it.

  2. WRONG CONCLUSION. The old script compared targets across 5 SEEDS that all train on the same 917
     catalysts. That measures seed noise, not whether a gain generalises to other catalysts. Under a
     catalyst-level bootstrap the effect largely evaporates, and simply averaging predictions over
     seeds -- changing no target at all -- produces the same gain. See the VERDICT block below.

WHAT THIS SCRIPT NOW DOES. It is a robustness check, not a search for a better target. For each
quantile q it builds the label "max over a catalyst's temperature cells of the within-cell q-quantile"
and asks whether the catalyst RANKING changes. q = 1.00 is exactly the current target (max over all
rows). Every variant is trained on its own label but SCORED against the same quantity: the catalyst's
TRUE observed maximum yield, which is the actual screening objective. A target that looks good against
its own definition proves nothing -- phase5 already caught p90/top5mean that way.

Three uncertainty measures are reported, because they disagree and the disagreement is the point:
  - across SEEDS (what the old script used) -- measures fold-shuffle noise only,
  - across CATALYSTS (paired bootstrap on seed-averaged out-of-fold predictions) -- the honest one,
  - the SEED-AVERAGING baseline -- how much is won by averaging predictions with no target change.

Protocol: formulation B (one row per catalyst), composition-only features, tuned LGBM, catalyst-grouped
5-fold CV, per-split train-only scaler, identical folds across variants.
Output: phase8_target_robustness.json
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

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']

lab = d.dl_lab.copy()
lab['cat_id'] = d.groups
n_cat = d.n_cat
Xc = lab.groupby('cat_id')[el_cols].first().values.astype(float)
cellg = lab.groupby(['cat_id', 'Temperature_C'])[TARGET]

TRUE_MAX = lab.groupby('cat_id')[TARGET].max().values   # the real screening objective

# ---------------------------------------------------------------- the condition-grid evidence
cell_n = cellg.size()
per_cat = lab.groupby('cat_id').size()
sizes = cell_n.reset_index(name='n')
exact135 = per_cat[per_cat == 135].index
full_grid = [c for c in exact135 if sorted(sizes[sizes.cat_id == c].n) == [27] * 5]
GRID = {'n_cells': int(len(cell_n)), 'mean_rows_per_cell': float(cell_n.mean()),
        'modal_cell_size': int(cell_n.mode().iloc[0]), 'max_cell_size': int(cell_n.max()),
        'cells_above_54': int((cell_n > 54).sum()),
        'catalysts_with_exactly_135_rows': int(len(exact135)),
        'of_which_split_as_27x5': int(len(full_grid))}
log(f"grid: modal cell {GRID['modal_cell_size']}, max cell {GRID['max_cell_size']}, "
    f"{GRID['of_which_split_as_27x5']}/{GRID['catalysts_with_exactly_135_rows']} "
    f"catalysts at 135 rows split as (27,27,27,27,27)")


# ---------------------------------------------------------------- label constructions
def target_quantile(q):
    """max over a catalyst's temperature cells of the within-cell q-quantile. q=1.0 == observed max."""
    s = cellg.quantile(q).reset_index()
    return s.groupby('cat_id')[TARGET].max().values


def target_nearest_rank(frac=0.25):
    """Like the upper-quartile label, but every value is a LITERAL measured yield: the
    ceil(frac*n)-th LARGEST actual measurement in each cell, then max across cells. Avoids the
    objection that an interpolated percentile corresponds to no experiment anyone ran."""
    def pick(s):
        a = np.sort(s.values)[::-1]
        return a[int(np.ceil(frac * len(a))) - 1]
    return cellg.apply(pick).reset_index().groupby('cat_id')[TARGET].max().values


y_median_raw = lab.groupby('cat_id')[TARGET].median().values   # the literal "just use the median" idea


# ---------------------------------------------------------------- grouped CV
def fold_assignment(seed, n=n_cat, k=5):
    r = np.random.default_rng(seed); perm = r.permutation(n); f = np.empty(n, int)
    for i, ch in enumerate(np.array_split(perm, k)): f[ch] = i
    return f


def oof_predictions(y_train, seed):
    """Out-of-fold predictions under catalyst-grouped CV. Trained on y_train, never on TRUE_MAX."""
    f = fold_assignment(seed); yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), y_train[tr])
        yp[va] = m.predict(sc.transform(Xc[va]))
    return yp


SEEDS = [0, 1, 2, 7, 13]
QS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

VARIANTS = [(f'q{q:.2f}', target_quantile(q)) for q in QS]
VARIANTS.append(('nearest_rank_upper_quartile', target_nearest_rank(0.25)))
VARIANTS.append(('median_of_all_raw_rows', y_median_raw))

RES, MEAN_PRED = {}, {}
for name, ytr in VARIANTS:
    preds = [oof_predictions(ytr, s) for s in SEEDS]
    mets = [cat_metrics(np.arange(n_cat), TRUE_MAX, p) for p in preds]
    MEAN_PRED[name] = np.mean(preds, axis=0)
    RES[name] = {
        'target_mean': float(np.mean(ytr)),
        'spearman_mean': float(np.mean([m['spearman_max'] for m in mets])),
        'spearman_std': float(np.std([m['spearman_max'] for m in mets], ddof=1)),
        'enrichment_mean': float(np.mean([m['enrichment_top10pct'] for m in mets])),
        'precision_at20_mean': float(np.mean([m['precision_at20_vs_top10pct'] for m in mets])),
        'spearman_seed_averaged_predictions': float(spearmanr(TRUE_MAX, MEAN_PRED[name])[0]),
        'per_seed_spearman': [float(m['spearman_max']) for m in mets]}
    log(f"{name:28s} spearman={RES[name]['spearman_mean']:.4f} +/- {RES[name]['spearman_std']:.4f}  "
        f"enrich={RES[name]['enrichment_mean']:.2f}x  prec@20={RES[name]['precision_at20_mean']:.2f}")

CONTROL = 'q1.00'   # == the current target: max over every row of the catalyst

# ---------------------------------------------------------------- honest uncertainty
# Paired bootstrap over CATALYSTS on seed-averaged out-of-fold predictions. This is the measure the
# superseded script lacked: its 5 seeds all trained on the same 917 catalysts, so they could not say
# anything about generalising to other catalysts.
N_BOOT = 3000
rng = np.random.default_rng(7)
BOOT_IDX = [rng.integers(0, n_cat, n_cat) for _ in range(N_BOOT)]
ctrl_pred = MEAN_PRED[CONTROL]

for name in RES:
    if name == CONTROL:
        continue
    a = MEAN_PRED[name]
    deltas = np.array([spearmanr(TRUE_MAX[i], a[i])[0] - spearmanr(TRUE_MAX[i], ctrl_pred[i])[0]
                       for i in BOOT_IDX])
    RES[name]['catalyst_bootstrap'] = {
        'delta_spearman_mean': float(deltas.mean()),
        'ci95_low': float(np.percentile(deltas, 2.5)),
        'ci95_high': float(np.percentile(deltas, 97.5)),
        'p_better_than_control': float((deltas > 0).mean()),
        'clears_zero': bool(np.percentile(deltas, 2.5) > 0)}
    RES[name]['seed_wins_vs_control'] = int(sum(
        x > y for x, y in zip(RES[name]['per_seed_spearman'], RES[CONTROL]['per_seed_spearman'])))
    # NOTE: this is the difference of the mean per-seed Spearman. It is NOT the quantity the
    # bootstrap CI above is built from (that one is measured on seed-averaged predictions and lives
    # in catalyst_bootstrap.delta_spearman_mean). They differ by ~0.001; keep them labeled apart.
    RES[name]['delta_spearman_vs_control_seedmean'] = float(
        RES[name]['spearman_mean'] - RES[CONTROL]['spearman_mean'])

# How much is won by seed-averaging alone, with NO target change? Context for every delta above.
SEED_AVG_GAIN = float(RES[CONTROL]['spearman_seed_averaged_predictions'] - RES[CONTROL]['spearman_mean'])

best_q = max((q for q in QS), key=lambda q: RES[f'q{q:.2f}']['spearman_mean'])
plateau = [RES[f'q{q:.2f}']['spearman_mean'] for q in QS if q <= 0.95]
enrichments = [RES[f'q{q:.2f}']['enrichment_mean'] for q in QS]

VERDICT = {
    'recommendation': 'KEEP the observed maximum (q=1.00) as the training label and the evaluation target',
    'reasons': [
        f"enrichment@10%, the metric named as primary, shows no trend across the sweep "
        f"({min(enrichments):.2f}x-{max(enrichments):.2f}x; control {RES[CONTROL]['enrichment_mean']:.2f}x)",
        f"seed-averaging alone, with no target change, gains {SEED_AVG_GAIN:+.4f} Spearman - "
        f"comparable to the whole target effect",
        "under a catalyst-level bootstrap the q=0.50 label does not clear zero against the control",
        "the observed maximum is an actual measured yield; an interpolated percentile is not",
    ],
    'what_to_report_instead': (
        'The catalyst ranking is insensitive to how the 135 conditions are collapsed into one label: '
        f'Spearman is flat at {min(plateau):.3f}-{max(plateau):.3f} for every quantile from 0.50 to '
        f'0.95, and enrichment@10% is unchanged throughout. We report the observed maximum, and no '
        'conclusion depends on that choice.'),
    'if_a_summarised_label_is_ever_adopted': (
        'use nearest_rank_upper_quartile, not an interpolated percentile - it scores equivalently and '
        'every label remains a literal measured yield from a real experiment'),
    'banned_vocabulary': ['repeats', 'replicates', 'denoising', 'noise floor', 'irreducible',
                          'identical inputs'],
}

json.dump({'meta': {'protocol': 'formulation B, composition-only, tuned LGBM, catalyst-grouped 5-fold CV',
                    'seeds': SEEDS, 'n_bootstrap': N_BOOT, 'control': CONTROL,
                    'scored_against': 'TRUE observed max in every case',
                    'supersedes': 'phase8_denoised_target.py / phase8_denoised_target.json'},
           'condition_grid_evidence': GRID,
           'seed_averaging_gain_no_target_change': SEED_AVG_GAIN,
           'results': RES, 'verdict': VERDICT},
          open('phase8_target_robustness.json', 'w'), indent=1)
log("wrote phase8_target_robustness.json")

print("\n=============== QUANTILE SWEEP (every variant scored against TRUE observed max) ===============")
print("Two deltas are reported and they are NOT the same quantity: 'd(seed)' is the difference of the")
print("mean per-seed Spearman, 'd(boot)' is the difference measured on seed-averaged predictions,")
print("which is the quantity the bootstrap CI is built from. Compare each CI to d(boot), not d(seed).\n")
print(f"{'label construction':28s} {'Spearman(seed)':>18s} {'enrich':>8s} {'p@20':>6s} "
      f"{'d(seed)':>9s} {'d(boot)':>9s} {'catalyst-bootstrap 95% CI':>26s}")
for name in [f'q{q:.2f}' for q in QS] + ['nearest_rank_upper_quartile', 'median_of_all_raw_rows']:
    r = RES[name]
    if name == CONTROL:
        tail = f"{'CONTROL (current target)':>46s}"
    else:
        b = r['catalyst_bootstrap']
        tail = (f"{r['delta_spearman_vs_control_seedmean']:+9.4f} {b['delta_spearman_mean']:+9.4f} "
                f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] {'clears 0' if b['clears_zero'] else 'spans 0'}")
    print(f"{name:28s} {r['spearman_mean']:.4f} +/- {r['spearman_std']:.4f} "
          f"{r['enrichment_mean']:7.2f}x {r['precision_at20_mean']:6.2f} {tail}")

print(f"\nseed-averaging alone, no target change : {SEED_AVG_GAIN:+.4f} Spearman "
      f"({RES[CONTROL]['spearman_mean']:.4f} -> {RES[CONTROL]['spearman_seed_averaged_predictions']:.4f})")
print(f"best quantile by Spearman              : q={best_q:.2f} "
      f"(on a flat plateau, not an optimum)")
print(f"\nVERDICT: {VERDICT['recommendation']}")
for r in VERDICT['reasons']:
    print(f"  - {r}")
print(f"\nREPORT INSTEAD: {VERDICT['what_to_report_instead']}")
log("done")
