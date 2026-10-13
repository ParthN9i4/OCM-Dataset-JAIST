"""
phase8b_label_sweep.py -- the committed training-label sweep for formulation B.

QUESTION. Formulation B trains on one row per catalyst, so a catalyst's measurements (135 condition
slots; up to 257 rows for a doubled catalyst) must be collapsed into one training label. Does that
choice change how well the model ranks catalysts by their observed maximum yield? Does the adopted
label rank at least as well as the observed max?

WHY THIS SCRIPT EXISTS. phase8_target_robustness.py swept q0.50-q1.00 and recommended keeping the
observed max. The PI has since adopted the per-cell median ("B_median"), and that choice has been
communicated to the collaborator. An uncommitted ad hoc analysis then added low quantiles, a
mean-of-five label and a five-row arm. This script makes that sweep reproducible, scores every arm the
same way, and adds a fair version of the five-row arm.

PROTOCOL (as phase8). Data.load(); one row per catalyst; features = the catalyst's composition columns
(66 columns: prep_enc + 65 element loadings. prep_enc and 31 of the element columns are constant in
the lab data, so 34 columns vary across the 917 catalysts; stored and asserted); catalyst-grouped
5-fold CV with catalyst_folds(seed); seeds 0, 1, 2, 7, 13; TUNED LightGBM from grouped_tuning.json
(overrides stored in the JSON); StandardScaler fit on the training fold only. Every arm uses the same
folds. One difference from phase8: n_jobs=2 here, the lgb_params default n_jobs=-1 there. R1's exact
reproduction shows it has no effect. A label is built from measured rows only: a slot excluded by the
lab's mass-balance check is missing, never yield 0 (phase12_runs_and_coverage.json
exclusion_test.rule_R4 verdict).

LABELS. A cell is a (catalyst, temperature) pair. L_q = max over the catalyst's cells of the
within-cell q-quantile (pandas linear interpolation; phase8 target_quantile).
  q0.00    cell minimum, then max over temperatures.
  q0.25    cell lower quartile, then max.
  q0.50    ADOPTED. Per-cell median, then max over temperatures (= "B_median" in phase9/phase11).
  q0.75    cell upper quartile, then max.
  q1.00    observed max (= TRUE_MAX, "B_max").
  nearest_rank_upper_quartile   phase8: the ceil(0.25 n)-th largest measured yield in each cell, then max.
  global_min               min over all the catalyst's rows.
  median_of_all_raw_rows   phase8: median over all the catalyst's rows, ignoring cells.
  mean_of_five             mean of L_0.00, L_0.25, L_0.50, L_0.75, L_1.00.
  five_row_unfair          five training rows per catalyst: identical features, labels L_0.00..L_1.00.
                           TUNED as is. Reproduces the earlier ad hoc test.
  five_row_fair            as five_row_unfair, but min_child_samples = 5 x TUNED's (100 rows).
  For the five-row arms the catalyst prediction is the model's prediction for that catalyst. All five
  of its rows get the same prediction (asserted).

WHY THE EARLIER FIVE-ROW TEST WAS UNFAIR. The ad hoc analysis gave each catalyst five training rows
with identical features and the labels L_0.00..L_1.00, and kept TUNED unchanged. A tree cannot
separate rows with identical features. All five land in the same leaf and get one prediction. Under
squared error that prediction is fitted to the mean of the five labels, so the arm learns roughly the
mean_of_five label. What did change is the regularisation. min_child_samples counts rows, so TUNED's
20 rows per leaf meant 4 catalysts per leaf instead of 20. The arm measured a looser model, not a
different label. five_row_fair restores 20 catalysts per leaf (100 rows). It is still not identical
to mean_of_five: reg_lambda acts against 5x the hessian per catalyst, and histogram binning counts
rows. The JSON stores how close five_row_fair and mean_of_five predictions are.

SCORING. Every arm is trained on its own label and scored against TRUE_MAX, the observed max. Never
against its own label. Populations (catalysts):
  all_917   all catalysts.
  EE_771    equal effort: >= 20 rows in some (catalyst, temperature) cell (Data.load groups and
            temperatures; equal_effort_mask).
  ND_889    without the 28 doubled catalysts (doubled_catalyst_mask on load_with_conditions; its
            catalyst codes are aligned with Data.load, asserted below).
Metrics: Spearman(prediction, TRUE_MAX); enrichment@10% (top-10% overlap / 0.10); precision@20 (share
of the top 20 predicted that are in the true top 10%). Two estimators, always labelled:
  per_seed_mean               metric on each seed's out-of-fold predictions, then mean over seeds.
  seed_averaged_predictions   out-of-fold predictions averaged over the seeds, then the metric.
The paired catalyst bootstrap always uses seed-averaged predictions. It reports a bootstrap-mean delta
(the quantity phase8 stored) and a point delta on the full population; both are labelled.
Descriptive diagnostics, no test: the seed-averaging gain per arm (seed-averaged minus per-seed mean
Spearman), and each arm's point delta vs q1.00 when only the first k seeds are averaged. They explain
why the uncommitted 2-seed audit found larger deltas than this 5-seed run. Seed spread per arm: the
median over catalysts of the across-seed SD (ddof=1) of the out-of-fold prediction, and the mean
Spearman between the out-of-fold predictions of every pair of seeds. This describes how much a model
varies between seeds. It is never used to test a difference between labels.

CONDITION-MIX FACTS (load_with_conditions; no model, no seed). Spearman across all observed cells
between the share of CH4:O2 = 6 rows in the cell and (a) the cell median, (b) the cell max. The share
of catalysts whose single highest row is at CH4:O2 = 2. The change in the q0.50 and q1.00 labels when
each catalyst's single highest row is dropped (Data.load rows; descriptive). These are descriptive and
carry no test. One wording rule: the caution says the cell max shows "about the same association" if
|rho(max)| >= |rho(median)| - 0.02; otherwise "a weaker association".

WORDING RULES (descriptive; they change no reading).
  W1 five_row_unfair. Its reading says its predictions "vary more between seeds" only if its median
     across-seed SD is above both mean_of_five's and five_row_fair's AND its mean pairwise seed Spearman
     is below both of theirs. Otherwise it says the seed spread does not single it out. Not measured
     with one seed (--dev).
  W2 boundary. A bootstrap reading is listed as "near the zero boundary" if the CI bound nearer zero is
     closer to zero than 10% of the CI width. This file holds over 100 bootstrap CIs at 95% each, so
     some "detected" readings are expected by chance; boundary readings should not be over-read.

RULES (fixed before the results).
  R1 reproduction. The q0.50, q0.75 and q1.00 arms, and the two other arms phase8 also ran
     (nearest_rank_upper_quartile, median_of_all_raw_rows), must reproduce phase8_target_robustness.json
     exactly (absolute difference 0): per_seed_spearman, target_mean, spearman_mean,
     spearman_seed_averaged_predictions, enrichment_mean, precision_at20_mean, and for every arm but
     q1.00 the catalyst_bootstrap mean, CI bounds and p_better_than_control. The script stops if R1
     fails. In --dev only seed 0's per-seed Spearman and target_mean are checked.
  R2 phase9. q1.00 and q0.50 seed-averaged Spearman, enrichment and precision@20 on all_917 and EE_771
     must equal phase9_equal_effort_eval.json exactly. The script stops if R2 fails. Skipped in --dev.
  R3 detection. A difference between two labels is "detected" only if the 95% CI of the paired
     catalyst bootstrap (paired_catalyst_bootstrap, 3000 resamples, seed 7) excludes zero. Otherwise
     it is "no detectable change". Seed-to-seed spread is never a test; per-seed values are stored for
     checking only.
  R4 adopted-label claim. The verdict says q0.50 "ranks at least as well as the observed max" only if
     no 95% CI of q0.50 - q1.00 lies entirely below zero, over 2 metrics (Spearman, enrichment) x 3
     populations. This is a no-detectable-loss claim, not a superiority claim. If R4 fails, the verdict
     names the detected loss instead. The verdict also lists, per metric, which label the point estimate
     favours, the largest loss the CI does not exclude, and every label detectably better than q1.00.
  R5 five-row identity. Each catalyst's five rows get exactly the same prediction: max spread 0 over
     len(seeds) x 2 arms x 917 catalyst predictions. A fixed tree model gives identical outputs for
     identical inputs, so this holds by construction. It checks the row stacking and reshape only. It
     is not evidence about labels. The script stops if R5 fails.
  R6 supersedes wording. The verdict says "the evidence permits either" (q0.50 or q1.00) only if all six
     95% CIs of q0.50 - q1.00 (2 metrics x 3 populations) include zero. Otherwise it names each
     detected difference instead.

Output:  phase8b_label_sweep.json
Usage:   python3 phase8b_label_sweep.py [--dev]
         --dev = seed 0 only (all 5 folds, so every catalyst gets an out-of-fold prediction),
         300 bootstrap resamples, R2 and the 2-seed audit check skipped, no file written.
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from ocm_eval import (Data, load_with_conditions, lgb_params, cat_metrics, TARGET, catalyst_folds,
                      equal_effort_mask, design_axes, doubled_catalyst_mask, paired_catalyst_bootstrap,
                      MIN_CELL)

DEV = '--dev' in sys.argv
SEEDS = [0] if DEV else [0, 1, 2, 7, 13]
N_BOOT = 300 if DEV else 3000
BOOT_SEED = 7
N_JOBS = 2
OUT_JSON = 'phase8b_label_sweep.json'
Q5 = [0.00, 0.25, 0.50, 0.75, 1.00]
POP_NAMES = ['all_917', 'EE_771', 'ND_889']
METRICS = ['spearman', 'enrichment', 'precision_at20']
BOOT_METRICS = ['spearman', 'enrichment']
MAX_LABEL, MED_LABEL = 'q1.00', 'q0.50'
ADOPTED_TEXT = 'q0.50 (per-cell median, then max over temperatures)'
CAUTION_TOL = 0.02         # caution wording: "about the same association" if |rho(max)| >= |rho(median)| - this
W2_FRAC = 0.10             # rule W2: boundary if the CI bound nearer zero is within this share of the CI width

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def _f(v, fmt='+.4f'):
    """Format v; if a nonzero v would print as zero, add decimals (up to 8) so it does not read as 0."""
    s = format(v, fmt)
    sign, dec = fmt.split('.')
    d = int(dec.rstrip('f'))
    while v != 0 and float(s) == 0 and d < 8:
        d += 1
        s = format(v, f"{sign}.{d}f")
    return s

# ------------------------------------------------------------------------------- data (as phase8)
d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']
lab = d.dl_lab.copy()
lab['cat_id'] = d.groups
n_cat = d.n_cat
Xc = lab.groupby('cat_id')[el_cols].first().values.astype(float)
cellg = lab.groupby(['cat_id', 'Temperature_C'])[TARGET]
TRUE_MAX = lab.groupby('cat_id')[TARGET].max().values          # the screening objective
assert n_cat == 917 and Xc.shape == (917, 66) and lab.prep_enc.nunique() == 1
N_VARYING = int((Xc.std(0) > 0).sum())                          # columns that differ between catalysts
assert N_VARYING == 34, N_VARYING
ROWS_PER_CAT = np.bincount(d.groups, minlength=n_cat)


def target_quantile(q):
    """phase8: max over a catalyst's temperature cells of the within-cell q-quantile."""
    s = cellg.quantile(q).reset_index()
    return s.groupby('cat_id')[TARGET].max().values


def target_nearest_rank(frac=0.25):
    """phase8: the ceil(frac*n)-th largest measured yield in each cell, then max across cells."""
    def pick(s):
        a = np.sort(s.values)[::-1]
        return a[int(np.ceil(frac * len(a))) - 1]
    return cellg.apply(pick).reset_index().groupby('cat_id')[TARGET].max().values


LQ = {q: target_quantile(q) for q in Q5}
FIVE = np.column_stack([LQ[q] for q in Q5])                     # n_cat x 5 labels for the five-row arms
LABELS = {
    'q0.00': LQ[0.00], 'q0.25': LQ[0.25], 'q0.50': LQ[0.50], 'q0.75': LQ[0.75], 'q1.00': LQ[1.00],
    'nearest_rank_upper_quartile': target_nearest_rank(0.25),
    'global_min': lab.groupby('cat_id')[TARGET].min().values,
    'median_of_all_raw_rows': lab.groupby('cat_id')[TARGET].median().values,
    'mean_of_five': FIVE.mean(1),
}
ARMS = list(LABELS) + ['five_row_unfair', 'five_row_fair']
assert np.array_equal(LABELS['q1.00'], TRUE_MAX)
median_label_check = float(np.abs(LABELS['q0.50'] - cellg.median().groupby(level=0).max().values).max())

# ------------------------------------------------------------------------------- populations
dfc, _feat_c, gc, n_cat_c = load_with_conditions()
yc = dfc[TARGET].values.astype(float)
true_max_c = pd.Series(yc).groupby(gc).max().reindex(range(n_cat)).values
rows_d = np.bincount(d.groups, minlength=n_cat); rows_c = np.bincount(gc, minlength=n_cat)
# codes from load_with_conditions must be the Data.load codes: same row counts and same max per code
assert n_cat_c == n_cat and np.array_equal(rows_d, rows_c) and np.allclose(true_max_c, TRUE_MAX, atol=1e-9)
EE = equal_effort_mask(d.groups, lab.Temperature_C.values, n_cat, MIN_CELL)
assert np.array_equal(EE, equal_effort_mask(gc, dfc.Temperature_C.values, n_cat, MIN_CELL))
DBL = doubled_catalyst_mask(dfc, gc, n_cat)
POPS = {'all_917': np.ones(n_cat, bool), 'EE_771': EE, 'ND_889': ~DBL}
assert POPS['EE_771'].sum() == 771 and POPS['ND_889'].sum() == 889
log(f"{n_cat} catalysts; EE {EE.sum()}; ND {(~DBL).sum()}; seeds {SEEDS}; n_boot {N_BOOT}")

# ------------------------------------------------------------------------------- grouped CV
FIVE_ROW_CHECK = {'n_catalyst_predictions_checked': 0, 'max_abs_spread_across_five_rows': 0.0}


def oof(seed, y=None, five=False, mcs=None):
    """Out-of-fold catalyst predictions. Single-row arms train on y; five-row arms on FIVE."""
    over = dict(TUNED) if mcs is None else {**TUNED, 'min_child_samples': mcs}
    f = catalyst_folds(seed, n_cat); yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        if five:      # label-major stacking: all L_0.00 rows, then all L_0.25 rows, ...
            Xtr, ytr, Xva = np.vstack([Xc[tr]] * 5), FIVE[tr].T.reshape(-1), np.vstack([Xc[va]] * 5)
        else:
            Xtr, ytr, Xva = Xc[tr], y[tr], Xc[va]
        sc = StandardScaler().fit(Xtr)
        m = lgb.LGBMRegressor(**lgb_params(seed, **over, n_jobs=N_JOBS)).fit(sc.transform(Xtr), ytr)
        p = m.predict(sc.transform(Xva))
        if five:
            p5 = p.reshape(5, len(va))
            if not (p5 == p5[0]).all():
                raise RuntimeError('R5 failed: five rows of one catalyst got different predictions')
            FIVE_ROW_CHECK['n_catalyst_predictions_checked'] += len(va)
            FIVE_ROW_CHECK['max_abs_spread_across_five_rows'] = max(
                FIVE_ROW_CHECK['max_abs_spread_across_five_rows'], float(np.ptp(p5, axis=0).max()))
            p = p5[0]
        yp[va] = p
    return yp


def score(mask, pred):
    """Catalyst-level metrics on `mask`, scored against TRUE_MAX (phase9 convention)."""
    m = cat_metrics(np.arange(int(mask.sum())), TRUE_MAX[mask], pred[mask])
    return {'spearman': float(m['spearman_max']), 'enrichment': float(m['enrichment_top10pct']),
            'precision_at20': float(m['precision_at20_vs_top10pct'])}


PRED = {}                                         # arm -> list of per-seed OOF predictions
for name in ARMS:
    if name == 'five_row_unfair':
        PRED[name] = [oof(s, five=True) for s in SEEDS]
    elif name == 'five_row_fair':
        PRED[name] = [oof(s, five=True, mcs=5 * TUNED['min_child_samples']) for s in SEEDS]
    else:
        PRED[name] = [oof(s, y=LABELS[name]) for s in SEEDS]
    log(f"trained {name}")
MP = {k: np.mean(v, axis=0) for k, v in PRED.items()}   # seed-averaged OOF predictions

RES = {}
for name in ARMS:
    RES[name] = {}
    for pn in POP_NAMES:
        per = [score(POPS[pn], p) for p in PRED[name]]
        RES[name][pn] = {
            'n_catalysts': int(POPS[pn].sum()),
            'per_seed': {m: [r[m] for r in per] for m in METRICS},
            'per_seed_mean': {m: float(np.mean([r[m] for r in per])) for m in METRICS},
            'seed_averaged_predictions': score(POPS[pn], MP[name])}

# ------------------------------------------------------------------------------- paired bootstrap
BOOT = {}
for ref in [MAX_LABEL, MED_LABEL]:
    BOOT[f'vs_{ref}'] = {}
    for name in ARMS:
        if name == ref:
            continue
        BOOT[f'vs_{ref}'][name] = {}
        for pn in POP_NAMES:
            BOOT[f'vs_{ref}'][name][pn] = {}
            for met in BOOT_METRICS:
                b = paired_catalyst_bootstrap(TRUE_MAX, MP[name], MP[ref],
                                              mask=None if pn == 'all_917' else POPS[pn],
                                              n_boot=N_BOOT, seed=BOOT_SEED, metric=met)
                b['reading'] = 'detected' if b['clears_zero'] else 'no detectable change'
                BOOT[f'vs_{ref}'][name][pn][met] = b
    log(f"bootstraps vs {ref} done")

# ------------------------------------------------------------------------------- R1: phase8 reproduction
P8 = json.load(open('phase8_target_robustness.json'))['results']
PHASE8_CHECK, r1_ok = {}, True


def _cmp(mine, ref):
    global r1_ok
    if isinstance(mine, list):
        diff = float(max(abs(a - b) for a, b in zip(mine, ref))) if len(mine) == len(ref) else float('inf')
    else:
        diff = float(abs(mine - ref))
    r1_ok &= diff == 0.0
    return {'this_script': mine, 'phase8_json': ref, 'abs_diff': diff, 'exact': diff == 0.0}


for name in [MED_LABEL, 'q0.75', MAX_LABEL, 'nearest_rank_upper_quartile', 'median_of_all_raw_rows']:
    r, p8 = RES[name]['all_917'], P8[name]
    c = {'target_mean': _cmp(float(np.mean(LABELS[name])), p8['target_mean'])}
    if DEV:
        c['per_seed_spearman_seed0'] = _cmp(r['per_seed']['spearman'][0], p8['per_seed_spearman'][0])
    else:
        c['per_seed_spearman'] = _cmp(r['per_seed']['spearman'], p8['per_seed_spearman'])
        c['spearman_mean'] = _cmp(r['per_seed_mean']['spearman'], p8['spearman_mean'])
        c['spearman_seed_averaged_predictions'] = _cmp(r['seed_averaged_predictions']['spearman'],
                                                       p8['spearman_seed_averaged_predictions'])
        c['enrichment_mean'] = _cmp(r['per_seed_mean']['enrichment'], p8['enrichment_mean'])
        c['precision_at20_mean'] = _cmp(r['per_seed_mean']['precision_at20'], p8['precision_at20_mean'])
        if name != MAX_LABEL:
            b, b8 = BOOT[f'vs_{MAX_LABEL}'][name]['all_917']['spearman'], p8['catalyst_bootstrap']
            c['catalyst_bootstrap_delta_mean'] = _cmp(b['delta_boot_mean'], b8['delta_spearman_mean'])
            c['catalyst_bootstrap_ci95_low'] = _cmp(b['ci95_low'], b8['ci95_low'])
            c['catalyst_bootstrap_ci95_high'] = _cmp(b['ci95_high'], b8['ci95_high'])
            c['catalyst_bootstrap_p_better'] = _cmp(b['p_a_better'], b8['p_better_than_control'])
    PHASE8_CHECK[name] = c
PHASE8_CHECK['population'] = 'all_917'
PHASE8_CHECK['estimator'] = (
    'per_seed_spearman, spearman_mean, enrichment_mean, precision_at20_mean: per-seed out-of-fold '
    'metrics (mean over seeds); spearman_seed_averaged_predictions: on seed-averaged out-of-fold '
    f'predictions; catalyst_bootstrap_*: arm - q1.00 Spearman, paired catalyst bootstrap on seed-averaged '
    f'predictions, {N_BOOT} resamples, seed {BOOT_SEED}; target_mean: label mean, no model')
PHASE8_CHECK['rule_R1_passes'] = bool(r1_ok)
if not r1_ok:
    raise RuntimeError('R1 failed: phase8_target_robustness.json not reproduced exactly')
log("R1 passes: phase8 reproduced exactly" + (" (dev: seed 0 only)" if DEV else ""))

# ------------------------------------------------------------------------------- R2: phase9 reproduction
PHASE9_CHECK = {'skipped_in_dev': True} if DEV else {}
if not DEV:
    P9 = json.load(open('phase9_equal_effort_eval.json'))['results']
    r2_ok = True
    for name, key in [(MAX_LABEL, 'real_model'), (MED_LABEL, 'median_trained_model')]:
        for pn, p9 in [('all_917', 'all_917'), ('EE_771', 'equal_effort')]:
            for m in METRICS:
                mine = RES[name][pn]['seed_averaged_predictions'][m]
                diff = float(abs(mine - P9[key][p9][m]))
                r2_ok &= diff == 0.0
                PHASE9_CHECK[f'{name}_{pn}_{m}_seed_averaged'] = {
                    'this_script': mine, 'phase9_json': P9[key][p9][m], 'abs_diff': diff}
    PHASE9_CHECK['rule_R2_passes'] = bool(r2_ok)
    if not r2_ok:
        raise RuntimeError('R2 failed: phase9_equal_effort_eval.json not reproduced exactly')
    log("R2 passes: phase9 reproduced exactly")

# ------------------------------------------------------------------------------- five-row diagnostics
n_expected_five = len(SEEDS) * 2 * n_cat
r5_ok = (FIVE_ROW_CHECK['n_catalyst_predictions_checked'] == n_expected_five
         and FIVE_ROW_CHECK['max_abs_spread_across_five_rows'] == 0.0)
FIVE_ROW_CHECK['n_catalyst_predictions_expected'] = n_expected_five
FIVE_ROW_CHECK['rule_R5_passes'] = bool(r5_ok)
FIVE_ROW_CHECK['note'] = ('max_abs_spread_across_five_rows is the largest spread among the five '
                          'predictions of any catalyst; R5 requires exactly 0. It holds by construction '
                          '(identical inputs to a fixed tree model) and checks the row stacking and reshape '
                          'only; it is not evidence about labels.')
if not r5_ok:
    raise RuntimeError('R5 failed: five-row prediction check')
for nm in ['five_row_unfair', 'five_row_fair']:
    FIVE_ROW_CHECK[f'{nm}_vs_mean_of_five_seed_averaged_predictions'] = {
        'population': 'all_917', 'estimator': 'on seed-averaged out-of-fold predictions; descriptive, no test',
        'pearson': float(np.corrcoef(MP[nm], MP['mean_of_five'])[0, 1]),
        'spearman': float(spearmanr(MP[nm], MP['mean_of_five'])[0]),
        'max_abs_diff_yield_pct': float(np.abs(MP[nm] - MP['mean_of_five']).max())}
# seed spread per arm (descriptive; describes model variance between seeds, never a test)
SEED_SPREAD = {'what': ('all_917; median over catalysts of the across-seed SD (ddof=1) of the out-of-fold '
                        'prediction (yield %), and mean Spearman between the out-of-fold predictions of every '
                        'pair of seeds; descriptive, not a test of any difference')}
if len(SEEDS) > 1:
    for k in ARMS:
        P = np.vstack(PRED[k])
        SEED_SPREAD[k] = {
            'median_across_seed_sd_yield_pct': float(np.median(P.std(axis=0, ddof=1))),
            'mean_pairwise_seed_spearman': float(np.mean([spearmanr(P[i], P[j])[0]
                                                          for i in range(len(P)) for j in range(i + 1, len(P))]))}
    _sp = {k: SEED_SPREAD[k] for k in ['five_row_unfair', 'mean_of_five', 'five_row_fair']}
    w1 = (_sp['five_row_unfair']['median_across_seed_sd_yield_pct']
          > max(_sp['mean_of_five']['median_across_seed_sd_yield_pct'],
                _sp['five_row_fair']['median_across_seed_sd_yield_pct'])
          and _sp['five_row_unfair']['mean_pairwise_seed_spearman']
          < min(_sp['mean_of_five']['mean_pairwise_seed_spearman'],
                _sp['five_row_fair']['mean_pairwise_seed_spearman']))
    spread_txt = (f"Seed spread (median across-seed SD of catalyst predictions; mean pairwise seed Spearman): "
                  + "; ".join(f"{k} {v['median_across_seed_sd_yield_pct']:.3f} yield-%, {v['mean_pairwise_seed_spearman']:.3f}"
                              for k, v in _sp.items())
                  + (". Its predictions vary more between seeds than those of mean_of_five and five_row_fair, "
                     "consistent with a less regularised model." if w1 else
                     ". The seed spread does not single it out."))
else:
    w1 = None
    spread_txt = "Seed spread not measured (one seed)."
SEED_SPREAD['rule_W1_five_row_unfair_varies_more'] = w1
_u = BOOT[f'vs_{MAX_LABEL}']['five_row_unfair']['all_917']['spearman']
FIVE_ROW_CHECK['five_row_unfair_reading'] = (
    f"all_917: per-seed mean Spearman {RES['five_row_unfair']['all_917']['per_seed_mean']['spearman']:.4f} vs "
    f"q1.00 {RES[MAX_LABEL]['all_917']['per_seed_mean']['spearman']:.4f}; on seed-averaged predictions "
    f"{RES['five_row_unfair']['all_917']['seed_averaged_predictions']['spearman']:.4f} vs "
    f"{RES[MAX_LABEL]['all_917']['seed_averaged_predictions']['spearman']:.4f}; bootstrap delta "
    f"{_u['delta_boot_mean']:+.4f} [{_u['ci95_low']:+.4f}, {_u['ci95_high']:+.4f}], "
    f"{'detected' if _u['clears_zero'] else 'no detectable change'}. Seed-averaging gain: five_row_unfair "
    f"{RES['five_row_unfair']['all_917']['seed_averaged_predictions']['spearman'] - RES['five_row_unfair']['all_917']['per_seed_mean']['spearman']:+.4f}, "
    f"q1.00 {RES[MAX_LABEL]['all_917']['seed_averaged_predictions']['spearman'] - RES[MAX_LABEL]['all_917']['per_seed_mean']['spearman']:+.4f} "
    f"(the gain is not specific to this arm; see seed_averaging_gain). " + spread_txt)
FIVE_ROW_CHECK['min_child_samples'] = {
    'TUNED_rows': TUNED['min_child_samples'], 'five_row_unfair_rows': TUNED['min_child_samples'],
    'five_row_unfair_catalysts_equivalent': TUNED['min_child_samples'] / 5,
    'five_row_fair_rows': 5 * TUNED['min_child_samples'],
    'five_row_fair_catalysts_equivalent': TUNED['min_child_samples']}

# ------------------------------------------------------------------------------- condition-mix facts
axc = design_axes(dfc)
ratio = axc.ch4_o2.values
cells = pd.DataFrame({'c': gc, 't': dfc.Temperature_C.values, 'y': yc, 'r6': ratio == 6.0, 'r2': ratio == 2.0})
cs = cells.groupby(['c', 't']).agg(share6=('r6', 'mean'), med=('y', 'median'), mx=('y', 'max'),
                                   n=('y', 'size'))
cs_nd = cs[~DBL[cs.index.get_level_values('c')]]
# within cells holding both ratios: is the ratio-6 median below the ratio-2 median?
w = cells.assign(rr=ratio).groupby(['c', 't', 'rr']).y.median().unstack('rr')
both = w[[2.0, 6.0]].dropna()
CELL_MIX = {
    'population': 'all observed (catalyst, temperature) cells, conditions file; no model',
    'n_cells': int(len(cs)),
    'spearman_share_ch4o2_6_vs_cell_median': float(spearmanr(cs.share6, cs.med)[0]),
    'spearman_share_ch4o2_6_vs_cell_max': float(spearmanr(cs.share6, cs.mx)[0]),
    'ND_889_cells': {'n_cells': int(len(cs_nd)),
                     'spearman_share_ch4o2_6_vs_cell_median': float(spearmanr(cs_nd.share6, cs_nd.med)[0]),
                     'spearman_share_ch4o2_6_vs_cell_max': float(spearmanr(cs_nd.share6, cs_nd.mx)[0])},
    'n_cells_share_not_one_third': int((~np.isclose(cs.share6, 1 / 3)).sum()),
    'within_cell_check': {
        'population': 'cells with at least one CH4:O2 = 2 row and one CH4:O2 = 6 row',
        'n_cells': int(len(both)),
        'share_cells_ratio6_median_below_ratio2_median': float((both[6.0] < both[2.0]).mean()),
        'median_over_cells_of_ratio2_minus_ratio6_median_yield_pct': float((both[2.0] - both[6.0]).median())},
    'note': ('A complete 27-condition cell holds exactly 1/3 CH4:O2 = 6 rows. The share moves only when '
             'rows were excluded by the mass-balance check (or duplicated in a doubled catalyst). This is '
             'an across-cell association. It mixes catalysts, so it shows that the cell median tracks '
             'the surviving condition mix, not that exclusion alone moves it.')}

idx_top = pd.Series(yc).groupby(gc).idxmax().reindex(range(n_cat)).values   # first top row, file order
peak_ratio = ratio[idx_top]
at_max = yc == true_max_c[gc]
n_ratios_at_max = pd.Series(ratio[at_max]).groupby(gc[at_max]).nunique()
has2_at_max = pd.Series(ratio[at_max] == 2.0).groupby(gc[at_max]).any()
PEAK = {
    'population': 'all_917; single highest row per catalyst (first in file order if tied); no model',
    'n_peak_at_ch4o2_2': int((peak_ratio == 2).sum()), 'n_peak_at_ch4o2_4': int((peak_ratio == 4).sum()),
    'n_peak_at_ch4o2_6': int((peak_ratio == 6).sum()),
    'share_peak_at_ch4o2_2': float((peak_ratio == 2).mean()),
    'n_catalysts_max_tied_across_ratios': int((n_ratios_at_max > 1).sum()),
    'n_catalysts_any_max_row_at_ch4o2_2': int(has2_at_max.sum()),
    'ND_889_share_peak_at_ch4o2_2': float((peak_ratio[~DBL] == 2).mean()),
    'note': 'CH4:O2 = 2 is the lowest (most O2-rich) ratio tested, the edge of the 2/4/6 grid.'}

# single highest row influence on the label (Data.load rows; descriptive)
keep = rows_d >= 2
lab_drop = lab.drop(index=lab.groupby('cat_id')[TARGET].idxmax().values)


def label_q_from(frame, q):
    return (frame.groupby(['cat_id', 'Temperature_C'])[TARGET].quantile(q)
            .groupby(level=0).max().reindex(range(n_cat)).values)


INFL = {'population': f'catalysts with >= 2 rows ({int(keep.sum())} of {n_cat}); no model',
        'n_catalysts': int(keep.sum()),
        'what': ("drop each catalyst's single highest row, rebuild the label, compare. "
                 "Absolute change in yield %; relative = change / label (label > 0)."), 'labels': {}}
for q in Q5:
    full = label_q_from(lab, q)
    assert np.array_equal(full, LQ[q])
    ch = np.abs(full - label_q_from(lab_drop, q))[keep]
    pos = full[keep] > 0
    INFL['labels'][f'q{q:.2f}'] = {
        'median_abs_change_yield_pct': float(np.median(ch)), 'mean_abs_change_yield_pct': float(ch.mean()),
        'median_relative_change': float(np.median(ch[pos] / full[keep][pos])),
        'share_catalysts_unchanged': float((ch == 0).mean())}
log("condition-mix facts done")

# ------------------------------------------------------------------------------- audit comparison
AUD2, NSEED = {}, {}
if not DEV:
    two = [SEEDS.index(0), SEEDS.index(1)]
    m2 = {k: np.mean([PRED[k][i] for i in two], axis=0)
          for k in ['q1.00', 'q0.50', 'q0.75', 'mean_of_five', 'global_min']}
    for k in ['q0.50', 'q0.75', 'mean_of_five', 'global_min']:
        b = paired_catalyst_bootstrap(TRUE_MAX, m2[k], m2['q1.00'], n_boot=N_BOOT, seed=BOOT_SEED)
        AUD2[k] = {'delta_boot_mean': b['delta_boot_mean'], 'ci95_low': b['ci95_low'],
                   'ci95_high': b['ci95_high'], 'reading': 'detected' if b['clears_zero'] else 'no detectable change'}
    # point delta (no bootstrap) on predictions averaged over the first k seeds of SEEDS
    NSEED = {'what': ('all_917; Spearman(seed-averaged prediction over the first k seeds of '
                      f'{SEEDS}) minus the same for q1.00; point delta, descriptive, no test'),
             **{k: {str(n): float(spearmanr(TRUE_MAX, np.mean(PRED[k][:n], axis=0))[0]
                              - spearmanr(TRUE_MAX, np.mean(PRED['q1.00'][:n], axis=0))[0])
                    for n in range(1, len(SEEDS) + 1)}
                for k in ['q0.50', 'q0.75', 'mean_of_five', 'five_row_fair', 'five_row_unfair']}}
SEED_AVG_GAIN = {'what': 'all_917; seed-averaged Spearman minus per-seed mean Spearman, per arm; descriptive',
                 **{k: float(RES[k]['all_917']['seed_averaged_predictions']['spearman']
                             - RES[k]['all_917']['per_seed_mean']['spearman']) for k in ARMS}}
P8_ENR = {k: v['enrichment_mean'] for k, v in P8.items() if k.startswith('q')}


def _a(audit, mine, what):
    return {'audit_value_uncommitted': audit, 'this_script': mine, 'this_script_quantity': what,
            'abs_diff': None if mine is None else float(abs(mine - audit))}


r_all = lambda k, est, m: RES[k]['all_917'][est][m]
b_all = lambda k, ref='q1.00': BOOT[f'vs_{ref}'][k]['all_917']['spearman']
qrange = [r_all(k, 'per_seed_mean', 'enrichment') for k in ['q0.50', 'q0.75', 'q1.00']]
AUDIT = {
    'note': ('Approximate values from an uncommitted ad hoc analysis. Its seeds and estimators are not '
             'recorded. Each entry names the quantity this script compares it to.'),
    'five_row_unfair_spearman': _a(0.7584, r_all('five_row_unfair', 'per_seed_mean', 'spearman'),
                                   'all_917, per-seed mean Spearman'),
    'five_row_unfair_enrichment': _a(4.17, r_all('five_row_unfair', 'per_seed_mean', 'enrichment'),
                                     'all_917, per-seed mean enrichment@10%'),
    'mean_of_five_spearman': _a(0.7672, r_all('mean_of_five', 'per_seed_mean', 'spearman'),
                                'all_917, per-seed mean Spearman'),
    'mean_of_five_enrichment': _a(4.33, r_all('mean_of_five', 'per_seed_mean', 'enrichment'),
                                  'all_917, per-seed mean enrichment@10%'),
    'five_row_mcs100_spearman': _a(0.768, r_all('five_row_fair', 'per_seed_mean', 'spearman'),
                                   'all_917, per-seed mean Spearman, five_row_fair'),
    'global_min_delta_vs_max_seedmean': _a(
        -0.0907, r_all('global_min', 'per_seed_mean', 'spearman') - r_all('q1.00', 'per_seed_mean', 'spearman'),
        'all_917, difference of per-seed mean Spearman, global_min - q1.00'),
    'global_min_delta_vs_max_bootstrap': _a(-0.0907, b_all('global_min')['delta_boot_mean'],
                                            'all_917, bootstrap-mean delta Spearman, 5-seed-averaged predictions'),
    'enrichment_range_q050_to_q100_low': {
        **_a(4.15, min(qrange), 'all_917, per-seed mean enrichment, min over q0.50/q0.75/q1.00 (this sweep)'),
        'phase8_grid_min': min(P8_ENR.values()), 'phase8_grid_argmin': min(P8_ENR, key=P8_ENR.get)},
    'enrichment_range_q050_to_q100_high': {
        **_a(4.43, max(qrange), 'all_917, per-seed mean enrichment, max over q0.50/q0.75/q1.00 (this sweep)'),
        'phase8_grid_max': max(P8_ENR.values()), 'phase8_grid_argmax': max(P8_ENR, key=P8_ENR.get)},
    'enrichment_range_explanation': (
        f"The audit range matches phase8_target_robustness.json across its 9 quantiles q0.50-q1.00 "
        f"(per-seed mean enrichment {min(P8_ENR.values()):.3f}x at {min(P8_ENR, key=P8_ENR.get)} to "
        f"{max(P8_ENR.values()):.3f}x at {max(P8_ENR, key=P8_ENR.get)}). This sweep keeps only q0.50, q0.75 "
        f"and q1.00 from that range, so its range is narrower. Same runs, different subset."),
    'cell_ratio6_share_vs_cell_median_spearman': _a(-0.249, CELL_MIX['spearman_share_ch4o2_6_vs_cell_median'],
                                                    'all observed cells, Spearman'),
    'catalysts_peak_at_ch4o2_2': _a(775, PEAK['n_peak_at_ch4o2_2'], 'all_917, count'),
}
if not DEV:
    for k, (aud, lo, hi) in {'mean_of_five': (0.0081, 0.0016, 0.0153), 'q0.75': (0.0083, None, None),
                             'q0.50': (0.0079, -0.0001, 0.0162), 'global_min': (-0.0907, None, None)}.items():
        AUDIT[f'two_seed_bootstrap_{k}_vs_q1.00'] = {
            **_a(aud, AUD2[k]['delta_boot_mean'],
                 'all_917, bootstrap-mean delta Spearman on predictions averaged over seeds 0 and 1 only'),
            'audit_ci95': [lo, hi], 'this_script_ci95_two_seeds': [AUD2[k]['ci95_low'], AUD2[k]['ci95_high']],
            'this_script_five_seeds': {'delta_boot_mean': b_all(k)['delta_boot_mean'],
                                       'ci95': [b_all(k)['ci95_low'], b_all(k)['ci95_high']]}}
    q5 = NSEED['q0.50']
    two_diff = max(AUDIT[f'two_seed_bootstrap_{k}_vs_q1.00']['abs_diff'] for k in ['mean_of_five', 'q0.75', 'q0.50'])
    gm_two = AUDIT['two_seed_bootstrap_global_min_vs_q1.00']
    AUDIT['two_seed_explanation'] = (
        f"Averaging seeds 0 and 1 only reproduces the audit's 2-seed bootstrap deltas (all_917 Spearman, arm - "
        f"q1.00) for mean_of_five, q0.75 and q0.50 to within {two_diff:.4f}. For global_min it gives "
        f"{gm_two['this_script']:+.4f} vs the audit's {gm_two['audit_value_uncommitted']:+.4f} "
        f"(off by {gm_two['abs_diff']:.4f}; see global_min_explanation). "
        f"The remainder is not explained here; the audit's seeds and resampling are not recorded. "
        f"The q0.50 - q1.00 point delta changes with the number of seeds averaged: 1 seed {q5['1']:+.4f}, "
        f"2 seeds {q5['2']:+.4f}, {len(SEEDS)} seeds {q5[str(len(SEEDS))]:+.4f}. Seed-averaging over all seeds "
        f"raises q1.00 by {SEED_AVG_GAIN['q1.00']:+.4f} and q0.50 by {SEED_AVG_GAIN['q0.50']:+.4f}. The committed "
        f"result is the {len(SEEDS)}-seed one. 2-seed q0.50 CI lower bound: audit -0.0001, this script "
        f"{AUD2['q0.50']['ci95_low']:+.4f}.")
    gm = {'two_seed_bootstrap_delta': AUD2['global_min']['delta_boot_mean'],
          'five_seed_bootstrap_delta': b_all('global_min')['delta_boot_mean'],
          'difference_of_per_seed_means': r_all('global_min', 'per_seed_mean', 'spearman')
          - r_all('q1.00', 'per_seed_mean', 'spearman')}
    closest = min(gm, key=lambda k: abs(gm[k] + 0.0907))
    AUDIT['global_min_explanation'] = (
        f"Candidates for the audit's -0.0907 (all_917, global_min - q1.00 Spearman): "
        + ", ".join(f"{k} {v:+.4f}" for k, v in gm.items())
        + f". Closest: {closest}. The {len(SEEDS)}-seed bootstrap CI "
        f"[{b_all('global_min')['ci95_low']:+.4f}, {b_all('global_min')['ci95_high']:+.4f}] is "
        + ('entirely below zero: detected loss.' if b_all('global_min')['ci95_high'] < 0 else
           'not entirely below zero.'))

# near-identical numbers from different estimators (all_917 Spearman)
NEAR = []
pool = [(k, est, r_all(k, est, 'spearman')) for k in ARMS for est in ['per_seed_mean', 'seed_averaged_predictions']]
for i, (k1, e1, v1) in enumerate(pool):
    for k2, e2, v2 in pool[i + 1:]:
        if e1 != e2 and k1 != k2 and abs(v1 - v2) < 0.0005:
            NEAR.append(f"{k1} {e1} {v1:.4f} vs {k2} {e2} {v2:.4f}: different label AND different estimator")

# ------------------------------------------------------------------------------- verdict
mb = BOOT[f'vs_{MAX_LABEL}'][MED_LABEL]
losses = [f"{pn} {met}" for pn in POP_NAMES for met in BOOT_METRICS if mb[pn][met]['ci95_high'] < 0]
R4 = len(losses) == 0
sp, en = mb['all_917']['spearman'], mb['all_917']['enrichment']
infl = INFL['labels']


def _ci(b, fmt='+.4f', unit=''):
    return (f"{_f(b['delta_boot_mean'], fmt)}{unit} [{_f(b['ci95_low'], fmt)}, {_f(b['ci95_high'], fmt)}], "
            f"{b['reading']}")


def _metric_txt(b, fmt, unit=''):
    """Bootstrap-mean delta and CI, plus which label the point delta favours and the largest loss not excluded."""
    fav = MED_LABEL if b['delta_point'] > 0 else (MAX_LABEL if b['delta_point'] < 0 else 'neither')
    t = f"{_ci(b, fmt, unit)}; point delta {b['delta_point']:{fmt}}{unit} favours {fav}"
    if not b['clears_zero'] and b['ci95_low'] < 0:
        t += f"; a loss of up to {-b['ci95_low']:{fmt.lstrip('+')}}{unit} is not excluded"
    return t


single_row_txt = (f"in the {INFL['n_catalysts']} of {n_cat} catalysts with >= 2 rows, dropping each "
                  f"catalyst's single highest row moves the q0.50 label by a median "
                  f"{infl['q0.50']['median_abs_change_yield_pct']:.2f} yield-% and the observed max by "
                  f"{infl['q1.00']['median_abs_change_yield_pct']:.2f}")
if R4:
    rank_txt = (f"ranks at least as well as the observed max in the sense of rule R4: no 95% CI of q0.50 - q1.00 "
                f"lies entirely below zero (2 metrics x 3 populations). This is a no-detectable-loss claim, not a "
                f"superiority claim. all_917, seed-averaged out-of-fold predictions, bootstrap-mean delta "
                f"[95% CI]: Spearman {_metric_txt(sp, '+.4f')}. Enrichment@10% {_metric_txt(en, '+.2f', 'x')}.")
else:
    rank_txt = (f"does NOT rank at least as well as the observed max: detected loss on {', '.join(losses)} "
                f"(all_917 Spearman {_ci(sp)})")
SENS = {}
for name in ARMS:
    SENS[name] = {}
    for pn in POP_NAMES:
        SENS[name][pn] = {}
        for met in BOOT_METRICS:
            row = {}
            for ref in [MAX_LABEL, MED_LABEL]:
                if name == ref:
                    row[f'vs_{ref}'] = 'reference'
                    continue
                b = BOOT[f'vs_{ref}'][name][pn][met]
                row[f'vs_{ref}'] = {'delta_boot_mean': b['delta_boot_mean'], 'delta_point': b['delta_point'],
                                    'ci95': [b['ci95_low'], b['ci95_high']], 'reading': b['reading']}
            SENS[name][pn][met] = row

# R6: "the evidence permits either" only if all six q0.50 - q1.00 CIs include zero
six = {f"{pn} {met}": mb[pn][met] for pn in POP_NAMES for met in BOOT_METRICS}
R6 = not any(b['clears_zero'] for b in six.values())
six_txt = "; ".join(f"{k} {_ci(b, '+.4f' if k.endswith('spearman') else '+.2f')}" for k, b in six.items())
detected_six = [k for k, b in six.items() if b['clears_zero']]

# reading limits: no superiority, estimator dependence, multiplicity
better_than_max, worse_than_max = [], []
for name in ARMS:
    if name == MAX_LABEL:
        continue
    for pn in POP_NAMES:
        for met in BOOT_METRICS:
            b = BOOT[f'vs_{MAX_LABEL}'][name][pn][met]
            f_ = '+.4f' if met == 'spearman' else '+.2f'
            if b['ci95_low'] > 0:
                better_than_max.append(f"{name} {pn} {met} {_ci(b, f_)[:-len(b['reading']) - 2]}")
            if b['ci95_high'] < 0:
                worse_than_max.append(f"{name} {pn} {met} {_ci(b, f_)[:-len(b['reading']) - 2]}")
med_better = [k for k, b in six.items() if b['ci95_low'] > 0]

BOUNDARY, n_ci, n_det = [], 0, 0
for ref in [MAX_LABEL, MED_LABEL]:
    for name, byp in BOOT[f'vs_{ref}'].items():
        if ref == MED_LABEL and name == MAX_LABEL:
            continue                      # mirror of q0.50 - q1.00; count each comparison once
        for pn, bym in byp.items():
            for met, b in bym.items():
                n_ci += 1; n_det += int(b['clears_zero'])
                near = min(abs(b['ci95_low']), abs(b['ci95_high']))
                if near < W2_FRAC * (b['ci95_high'] - b['ci95_low']):
                    BOUNDARY.append({'comparison': f"{name} - {ref}", 'population': pn, 'metric': met,
                                     'ci95': [b['ci95_low'], b['ci95_high']], 'reading': b['reading']})
for k, b in AUD2.items():
    n_ci += 1; n_det += int(b['reading'] == 'detected')
_enr = {k: (r_all(k, 'per_seed_mean', 'enrichment'), r_all(k, 'seed_averaged_predictions', 'enrichment'))
        for k in [MED_LABEL, MAX_LABEL]}
READING_LIMITS = {
    'not_superiority': (
        f"q0.50 is detectably better than q1.00 on: {', '.join(med_better) if med_better else 'none of the 6 comparisons'}. "
        f"Labels detectably better than q1.00 (CI entirely above zero; bootstrap-mean delta [95% CI]): "
        f"{'; '.join(better_than_max) if better_than_max else 'none'}. Documents must not turn the adopted-label "
        f"verdict into a superiority claim."),
    'detectably_worse_than_q1.00': worse_than_max,
    'enrichment_estimator_note': (
        f"all_917 enrichment@10%, q0.50 vs q1.00, depends on the estimator: per-seed mean "
        f"{_enr[MED_LABEL][0]:.3f}x vs {_enr[MAX_LABEL][0]:.3f}x ({_enr[MED_LABEL][0] - _enr[MAX_LABEL][0]:+.3f}x"
        + ("; equal to phase8's stored enrichment_mean values, checked by R1" if not DEV else "; dev, seed 0 only")
        + f"); seed-averaged predictions "
        f"{_enr[MED_LABEL][1]:.3f}x vs {_enr[MAX_LABEL][1]:.3f}x (point delta {en['delta_point']:+.3f}x); "
        f"bootstrap-mean delta {en['delta_boot_mean']:+.3f}x. Quote one estimator at a time and name it."),
    'multiplicity': (
        f"This file holds {n_ci} distinct paired-bootstrap 95% CIs (q1.00 - q0.50 counted once, as q0.50 - "
        f"q1.00; the 2-seed audit CIs included); {n_det} read 'detected'. At 95% per CI some "
        f"detections are expected by chance. {len(BOUNDARY)} readings sit near the zero boundary (rule W2, "
        f"listed in boundary_readings); do not over-read them."
        + (f" The q0.50 - q1.00 all_917 Spearman reading is '{AUD2[MED_LABEL]['reading']}' with seeds 0 and 1 "
           f"averaged and '{sp['reading']}' with all {len(SEEDS)} seeds." if AUD2 else "")),
    'boundary_rule': f"W2: CI bound nearer zero is closer to zero than {W2_FRAC:.0%} of the CI width",
    'boundary_readings': BOUNDARY,
}

VERDICT = {
    'adopted_label': ADOPTED_TEXT,
    'reason': f"not set by a single high reading ({single_row_txt}); {rank_txt}",
    'rule_R4_passes': bool(R4),
    'rule_R6_passes': bool(R6),
    'reading_limits': READING_LIMITS,
    'sensitivity_table': {
        'what': ('every label minus q1.00 (observed max) and minus q0.50 (adopted), per population and '
                 'metric; paired catalyst bootstrap on seed-averaged out-of-fold predictions, '
                 f'{N_BOOT} resamples, seed {BOOT_SEED}; delta_boot_mean = bootstrap mean (phase8 '
                 'convention), delta_point = full-population difference'),
        'rows': SENS},
    'cautions': [
        (f"{PEAK['share_peak_at_ch4o2_2']:.1%} of catalysts ({PEAK['n_peak_at_ch4o2_2']}/{n_cat}) have their "
         f"single highest row at CH4:O2 = 2, the edge of the tested grid. The true optimum may lie outside "
         f"it, so every label here is a within-grid quantity."),
        (f"The cell median shifts with which conditions survived the mass-balance check: across "
         f"{CELL_MIX['n_cells']} observed cells, Spearman(share of CH4:O2 = 6 rows, cell median) = "
         f"{CELL_MIX['spearman_share_ch4o2_6_vs_cell_median']:+.3f}. This is an across-cell association that "
         f"mixes catalysts. Within the {CELL_MIX['within_cell_check']['n_cells']} cells holding both ratios, "
         f"the CH4:O2 = 6 median lies below the CH4:O2 = 2 median in "
         f"{CELL_MIX['within_cell_check']['share_cells_ratio6_median_below_ratio2_median']:.0%} of them, "
         f"so a cell that lost CH4:O2 = 2 rows tends to have a lower median. The cell max shows "
         + ('about the same association' if abs(CELL_MIX['spearman_share_ch4o2_6_vs_cell_max'])
            >= abs(CELL_MIX['spearman_share_ch4o2_6_vs_cell_median']) - CAUTION_TOL else 'a weaker association')
         + f" ({CELL_MIX['spearman_share_ch4o2_6_vs_cell_max']:+.3f})"
         + (", so this caution applies to the observed max as well and is not a reason to prefer the max "
            "over the median."
            if abs(CELL_MIX['spearman_share_ch4o2_6_vs_cell_max'])
            >= abs(CELL_MIX['spearman_share_ch4o2_6_vs_cell_median']) - CAUTION_TOL else "."))],
    'supersedes': ("phase8_target_robustness.json verdict 'KEEP the observed maximum' -- the PI adopted the "
                   "median; " + ("the evidence permits either" if R6 else
                                 "the evidence does not permit either: q0.50 - q1.00 detected on "
                                 + ", ".join(detected_six))),
    'supersedes_basis': (f"rule R6; q0.50 - q1.00, seed-averaged out-of-fold predictions, bootstrap-mean delta "
                         f"[95% CI]: {six_txt}. "
                         + ("None of the six CIs excludes zero." if R6 else
                            f"Detected on: {', '.join(detected_six)}.")),
}

RUNTIME = (time.time() - t0) / 60
OUT = {
    'about': {
        'script': 'phase8b_label_sweep.py',
        'what': 'Training-label sweep for formulation B; every arm scored against TRUE_MAX (observed max).',
        'protocol': ('Data.load(); one row per catalyst; 66 composition columns (prep_enc + 65 element '
                     'loadings; 34 vary across the 917 catalysts); catalyst_folds 5-fold; TUNED LightGBM '
                     '(grouped_tuning.json), n_jobs=2; train-fold StandardScaler'),
        'n_feature_columns': int(Xc.shape[1]),
        'n_feature_columns_varying_across_catalysts': N_VARYING,
        'feature_columns_constant_in_lab_data': [c for c, v in zip(el_cols, Xc.std(0)) if v == 0],
        'rows_per_catalyst_data_load': {'min': int(ROWS_PER_CAT.min()), 'median': float(np.median(ROWS_PER_CAT)),
                                        'max': int(ROWS_PER_CAT.max()),
                                        'n_catalysts_above_135': int((ROWS_PER_CAT > 135).sum())},
        'tuned_overrides': TUNED,
        'lightgbm_n_jobs': N_JOBS,
        'n_jobs_note': ('phase8 used the lgb_params default n_jobs=-1; R1 reproduces phase8 exactly with '
                        'n_jobs=2, so it has no effect here'),
        'wording_thresholds': {'caution_cell_max_same_association_tolerance': CAUTION_TOL,
                               'W2_boundary_fraction_of_ci_width': W2_FRAC},
        'seeds': SEEDS, 'n_boot': N_BOOT, 'boot_seed': BOOT_SEED, 'dev_run': DEV,
        'populations': {'all_917': 'all catalysts', 'EE_771': f'>= {MIN_CELL} rows in some cell',
                        'ND_889': 'without the 28 doubled catalysts'},
        'estimators': {
            'per_seed_mean': 'metric on each seed\'s out-of-fold predictions, then mean over seeds',
            'seed_averaged_predictions': 'out-of-fold predictions averaged over seeds, then the metric',
            'bootstrap': 'paired catalyst bootstrap on seed-averaged predictions (ocm_eval.paired_catalyst_bootstrap)'},
        'truth': 'TRUE_MAX = observed max over all the catalyst\'s rows, for every arm',
        'median_label_vs_groupby_median_max_abs_diff': median_label_check,
        'runtime_minutes': RUNTIME},
    'label_definitions': {
        'q0.00': 'cell minimum, then max over temperatures', 'q0.25': 'cell lower quartile, then max',
        'q0.50': 'ADOPTED: cell median, then max over temperatures', 'q0.75': 'cell upper quartile, then max',
        'q1.00': 'observed max', 'nearest_rank_upper_quartile': 'ceil(0.25 n)-th largest row per cell, then max',
        'global_min': 'min over all rows', 'median_of_all_raw_rows': 'median over all rows, no cells',
        'mean_of_five': 'mean of q0.00, q0.25, q0.50, q0.75, q1.00',
        'five_row_unfair': 'five identical-feature rows per catalyst (labels q0.00..q1.00), TUNED min_child_samples=20 rows',
        'five_row_fair': 'as five_row_unfair, min_child_samples=100 rows (20 catalysts)'},
    'label_summary_no_model': {k: {'label_mean_all_917_yield_pct': float(np.mean(v)),
                                   'spearman_label_vs_true_max_all_917': float(spearmanr(v, TRUE_MAX)[0])}
                               for k, v in LABELS.items()},
    'results': RES,
    'bootstrap': BOOT,
    'phase8_reproduction_R1': PHASE8_CHECK,
    'phase9_reproduction_R2': PHASE9_CHECK,
    'five_row_checks_R5': FIVE_ROW_CHECK,
    'seed_spread': SEED_SPREAD,
    'condition_mix_facts': {'cell_ratio6_share': CELL_MIX, 'peak_ratio': PEAK,
                            'single_highest_row_influence': INFL},
    'seed_averaging_gain': SEED_AVG_GAIN,
    'delta_vs_n_seeds_averaged': NSEED if NSEED else {'skipped_in_dev': True},
    'audit_comparison': AUDIT,
    'near_identical_numbers': NEAR,
    'verdict': VERDICT,
}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if DEV:
    log("dev run: no file written")
else:
    json.dump(OUT, open(OUT_JSON, 'w'), indent=1, default=_json_default)
    log(f"wrote {OUT_JSON}")

# ------------------------------------------------------------------------------- results
print("\n========== RESULTS (every arm scored against TRUE_MAX, the observed max) ==========")
print("Spearman: 'seed-mean' = per-seed mean; 'seed-avg' = on seed-averaged predictions.")
print("Deltas: bootstrap-mean delta of seed-averaged Spearman, 95% paired catalyst bootstrap CI.")
for pn in POP_NAMES:
    print(f"\n-- population {pn} (n = {int(POPS[pn].sum())}) --")
    print(f"{'label':28s} {'seed-mean':>9s} {'seed-avg':>9s} {'enrich':>7s} {'p@20':>5s}  "
          f"{'d vs q1.00 [95% CI]':>28s}  {'d vs q0.50 [95% CI]':>28s}")
    for name in ARMS:
        r = RES[name][pn]; sa = r['seed_averaged_predictions']
        cols = []
        for ref in [MAX_LABEL, MED_LABEL]:
            if name == ref:
                cols.append(f"{'reference':>28s}")
            else:
                b = BOOT[f'vs_{ref}'][name][pn]['spearman']
                cols.append(f"{_f(b['delta_boot_mean'])} [{_f(b['ci95_low'])}, {_f(b['ci95_high'])}]"
                            f"{'*' if b['clears_zero'] else ' '}".rjust(28))
        print(f"{name:28s} {r['per_seed_mean']['spearman']:9.4f} {sa['spearman']:9.4f} "
              f"{sa['enrichment']:6.2f}x {sa['precision_at20']:5.2f}  {cols[0]}  {cols[1]}")
print("  * = detected (CI excludes zero). Otherwise: no detectable change.")
print("  enrich and p@20 are on seed-averaged predictions.")

print("\nEnrichment@10% deltas, all_917 (bootstrap mean, seed-averaged predictions, 95% CI):")
for name in ARMS:
    out = []
    for ref in [MAX_LABEL, MED_LABEL]:
        if name != ref:
            b = BOOT[f'vs_{ref}'][name]['all_917']['enrichment']
            out.append(f"vs {ref} {_f(b['delta_boot_mean'], '+.2f')} [{_f(b['ci95_low'], '+.2f')}, "
                       f"{_f(b['ci95_high'], '+.2f')}] "
                       f"{b['reading']}")
    print(f"  {name:28s} " + "; ".join(out))

print(f"\nR1 phase8 reproduction: {'pass' if PHASE8_CHECK['rule_R1_passes'] else 'FAIL'}"
      + (" (dev: seed 0 only)" if DEV else " (exact)"))
if not DEV:
    print(f"R2 phase9 reproduction: {'pass' if PHASE9_CHECK['rule_R2_passes'] else 'FAIL'} (exact)")
print(f"R5 five-row identity: {'pass' if FIVE_ROW_CHECK['rule_R5_passes'] else 'FAIL'} "
      f"({FIVE_ROW_CHECK['n_catalyst_predictions_checked']} catalyst predictions, max spread "
      f"{FIVE_ROW_CHECK['max_abs_spread_across_five_rows']}; holds by construction, checks stacking only)")
for nm in ['five_row_unfair', 'five_row_fair']:
    c = FIVE_ROW_CHECK[f'{nm}_vs_mean_of_five_seed_averaged_predictions']
    print(f"  {nm} vs mean_of_five predictions (all_917, seed-averaged): Pearson {c['pearson']:.4f}, "
          f"max abs diff {c['max_abs_diff_yield_pct']:.2f} yield-%")

print(f"  five_row_unfair: {FIVE_ROW_CHECK['five_row_unfair_reading']}")
print("\nSeed-averaging gain (all_917, seed-averaged minus per-seed mean Spearman): " + ", ".join(
    f"{k} {v:+.4f}" for k, v in SEED_AVG_GAIN.items() if k != 'what'))
if NSEED:
    print("q0.50 - q1.00 point delta by seeds averaged (all_917, no test): " + ", ".join(
        f"{n} seed(s) {v:+.4f}" for n, v in NSEED['q0.50'].items()))
    print(f"Audit: {AUDIT['two_seed_explanation']}")
    print(f"Audit: {AUDIT['global_min_explanation']}")
print(f"Audit: {AUDIT['enrichment_range_explanation']}")

print(f"\nCell mix (all {CELL_MIX['n_cells']} observed cells, no model): Spearman(share CH4:O2 = 6, "
      f"cell median) {CELL_MIX['spearman_share_ch4o2_6_vs_cell_median']:+.3f}; with cell max "
      f"{CELL_MIX['spearman_share_ch4o2_6_vs_cell_max']:+.3f}.")
print(f"Peak at CH4:O2 = 2 (all_917, single highest row): {PEAK['n_peak_at_ch4o2_2']}/{n_cat} = "
      f"{PEAK['share_peak_at_ch4o2_2']:.1%}; ties across ratios at the max: "
      f"{PEAK['n_catalysts_max_tied_across_ratios']}.")
for k in ['q0.50', 'q1.00']:
    v = INFL['labels'][k]
    print(f"Drop single highest row ({INFL['n_catalysts']} of {n_cat} catalysts with >= 2 rows), label {k}: "
          f"median change {v['median_abs_change_yield_pct']:.3f} yield-%, unchanged for "
          f"{v['share_catalysts_unchanged']:.0%} of them.")
if NEAR:
    print("\nNear-identical numbers that are DIFFERENT quantities (all_917 Spearman):")
    for s in NEAR:
        print(f"  {s}")

print(f"\nVERDICT: adopted label = {VERDICT['adopted_label']}")
print(f"  reason: {VERDICT['reason']}")
for c in VERDICT['cautions']:
    print(f"  caution: {c}")
print(f"  supersedes (R6 {'pass' if R6 else 'FAIL'}): {VERDICT['supersedes']}")
print(f"  basis: {VERDICT['supersedes_basis']}")
for k in ['not_superiority', 'enrichment_estimator_note', 'multiplicity']:
    print(f"  limit: {READING_LIMITS[k]}")
print("  boundary readings (W2): " + "; ".join(
    f"{x['comparison']} {x['population']} {x['metric']} [{_f(x['ci95'][0])}, {_f(x['ci95'][1])}] {x['reading']}"
    for x in BOUNDARY))
log(f"done in {RUNTIME:.1f} min")
