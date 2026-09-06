"""
phase10_ground_truth_invariance.py -- does the conditions-vs-time-on-stream question change our answer?

MOTIVATION (SESSION_CONTEXT.md section 7 item 2). We do not know whether the ~27 measurements in each
(catalyst, temperature) cell are 27 DISTINCT REACTION CONDITIONS or 27 SUCCESSIVE TIME-ON-STREAM
SAMPLES at one condition. Nothing in the CSV can settle it: every one of the 4,399 cells is stored
sorted descending by yield (0 violations in 84,675 within-cell adjacent comparisons), so row order
encodes rank, not acquisition sequence, and no position/decay/periodicity test is possible. Only
Prof. Taniike can answer it.

This script does NOT try to answer it. It asks the question that is answerable from the file:
DOES THE ANSWER CHANGE WHAT WE RECOMMEND?

  - Under the distinct-conditions reading, a catalyst's observed maximum is an achievable operating
    point, and the max is the right screening target.
  - Under the time-on-stream reading, the maximum is a fresh-catalyst transient, and the right target
    is something nearer the catalyst's sustained performance -- a LOW within-cell quantile.

Why the existing phases do not settle this:
  - phase8_target_robustness.py varies the TRAINING target but scores every variant against TRUE_MAX
    (line 130). One evaluation column. If the max is the wrong ground truth, that only shows our
    labels agree with each other.
  - phase5_target_audit.py has the mirror-image flaw: its D_sensitivity moves training and evaluation
    together, so p90 scoring above max says p90 is an EASIER target, not a better one.
  - Neither varies the two independently. That cell of the design is empty, and it is the one that
    speaks to the gating question. Also, phase8's sweep stops at q=0.50: no label below the median
    has ever been tested, and that is exactly the region the time-on-stream reading points at.

WORDING CONSTRAINT (do not weaken). A low within-cell quantile equals "sustained performance" only
under monotone deactivation. Without assuming decay kinetics it is the catalyst's OBSERVED FLOOR at
its best temperature. Ranking by floor versus by ceiling is the contrast that matters either way, so
this analysis is valid without assuming monotonicity -- but every claim must be worded as
floor-vs-ceiling, never as fresh-vs-aged.

Protocol: formulation B (one row per catalyst, composition only, temperature dropped), tuned LGBM,
catalyst-grouped 5-fold CV, per-split train-only scaler, identical folds across every variant.
Output: phase10_ground_truth_invariance.json
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

SEEDS = [0, 1, 2, 7, 13]
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
BUDGET = 20
TOP_FRAC = 0.10


def target_quantile(q):
    """Max over a catalyst's temperature cells of the within-cell q-quantile (phase8's construction).
    q=1.00 is the observed maximum (the ceiling); q=0.05 is the floor at the best temperature."""
    s = cellg.quantile(q).reset_index()
    return s.groupby('cat_id')[TARGET].max().values


TARGETS = {f'q{q:.2f}': target_quantile(q) for q in QS}
CEILING, FLOOR = 'q1.00', f'q{QS[0]:.2f}'
log(f"built {len(TARGETS)} targets over {n_cat} catalysts")


def topk(v, k):
    return set(np.argsort(-np.asarray(v))[:k])


def overlap(a, b, k):
    return len(topk(a, k) & topk(b, k)) / k


# ---------------------------------------------------------------- A. data-level agreement (no model)
# Computed BEFORE the modelling sections and used to motivate the design; the pre-registered rule
# below applies to sections B-D, which had not been run when the rule was fixed.
K10 = max(1, int(round(TOP_FRAC * n_cat)))
A = {'note': 'model-free: do the candidate ground truths rank the 917 catalysts the same way?',
     'n_catalysts': int(n_cat), 'k_top10pct': int(K10), 'vs_ceiling': {}}
for name, v in TARGETS.items():
    A['vs_ceiling'][name] = {
        'spearman_vs_ceiling': float(spearmanr(TARGETS[CEILING], v)[0]),
        'top20_overlap_vs_ceiling': float(overlap(TARGETS[CEILING], v, BUDGET)),
        'top10pct_overlap_vs_ceiling': float(overlap(TARGETS[CEILING], v, K10)),
        'mean_label': float(np.mean(v))}
A['floor_vs_ceiling_spearman'] = float(spearmanr(TARGETS[FLOOR], TARGETS[CEILING])[0])
A['floor_vs_ceiling_top20_overlap'] = float(overlap(TARGETS[FLOOR], TARGETS[CEILING], BUDGET))
log(f"A: floor-vs-ceiling spearman={A['floor_vs_ceiling_spearman']:.4f} "
    f"top20 overlap={A['floor_vs_ceiling_top20_overlap']:.2f}")

# ---------------------------------------------------------------- pre-registered decision rule
RULE = {
    'registered_before_running': ['B_train_eval_matrix', 'C_regret', 'D_shortlist_stability'],
    'informed_by': 'A_data_level_agreement (model-free, run first)',
    'claim_under_test': ('training on the observed maximum is SAFE under both readings of the 27 '
                         'within-cell slots'),
    'criteria': [
        '(i) enrichment@10% of the max-trained model against EVERY eval target q in [0.05,1.00] '
        'stays above 2.0x',
        '(ii) Spearman regret -- score(train=q1.00, eval=q) minus the best score achievable by any '
        'training target for that eval target -- is >= -0.05 for every eval target q'],
    'verdict_if_failed': ('the target choice is reading-dependent; report which eval targets fail '
                          'and treat the question to Prof. Taniike as blocking for those')}


# ---------------------------------------------------------------- grouped CV (identical to phase8)
def fold_assignment(seed, n=n_cat, k=5):
    r = np.random.default_rng(seed); perm = r.permutation(n); f = np.empty(n, int)
    for i, ch in enumerate(np.array_split(perm, k)): f[ch] = i
    return f


def oof_predictions(y_train, seed):
    """Out-of-fold predictions under catalyst-grouped CV. Trained on y_train only."""
    f = fold_assignment(seed); yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), y_train[tr])
        yp[va] = m.predict(sc.transform(Xc[va]))
    return yp


MEAN_PRED, PER_SEED_PRED = {}, {}
for name, ytr in TARGETS.items():
    preds = [oof_predictions(ytr, s) for s in SEEDS]
    PER_SEED_PRED[name] = preds
    MEAN_PRED[name] = np.mean(preds, axis=0)
    log(f"trained on {name}")

# ---------------------------------------------------------------- B. full train x eval matrix
B = {}
for tr_name in TARGETS:
    B[tr_name] = {}
    for ev_name, yev in TARGETS.items():
        m_avg = cat_metrics(np.arange(n_cat), yev, MEAN_PRED[tr_name],
                            top_frac=TOP_FRAC, budget=BUDGET)
        per_seed = [cat_metrics(np.arange(n_cat), yev, p, top_frac=TOP_FRAC, budget=BUDGET)
                    for p in PER_SEED_PRED[tr_name]]
        B[tr_name][ev_name] = {
            # seed-averaged predictions, then scored -- the phase9 convention
            'spearman_seedavg': float(m_avg['spearman_max']),
            'enrichment_seedavg': float(m_avg['enrichment_top10pct']),
            'precision_at20_seedavg': float(m_avg['precision_at20_vs_top10pct']),
            # mean over per-seed scores -- a DIFFERENT quantity, kept labelled apart (section 8 rule)
            'spearman_perseedmean': float(np.mean([x['spearman_max'] for x in per_seed])),
            'spearman_perseed_std': float(np.std([x['spearman_max'] for x in per_seed], ddof=1))}

# ---------------------------------------------------------------- C. regret of training on the max
# Regret is measured against the BEST AVAILABLE training target for each evaluation ground truth,
# not merely the matched one: the matrix diagonal is not always its column's maximum (training on a
# smoothed intermediate label often generalises better than training on the exact eval target), so
# a matched-target comparison would understate the cost. Both are stored; the verdict uses the
# stricter vs-best-available figure.
C = {'definition': ('regret(q) = spearman(train=q1.00, eval=q) - max_p spearman(train=p, eval=q); '
                    'negative means training on the ceiling costs us when the truth is q'),
     'per_eval_target': {}}
for ev in TARGETS:
    got = B[CEILING][ev]['spearman_seedavg']
    matched = B[ev][ev]['spearman_seedavg']
    best_tr = max(TARGETS, key=lambda p: B[p][ev]['spearman_seedavg'])
    best = B[best_tr][ev]['spearman_seedavg']
    C['per_eval_target'][ev] = {
        'spearman_trained_on_ceiling': got,
        'spearman_trained_on_matched_target': matched,
        'best_training_target': best_tr,
        'spearman_trained_on_best_available': best,
        'regret_vs_best_available': float(got - best),
        'regret_vs_matched_target': float(got - matched),
        'enrichment_trained_on_ceiling': B[CEILING][ev]['enrichment_seedavg'],
        'enrichment_trained_on_matched_target': B[ev][ev]['enrichment_seedavg']}
C['worst_regret'] = float(min(v['regret_vs_best_available'] for v in C['per_eval_target'].values()))
C['worst_regret_at'] = min(C['per_eval_target'],
                           key=lambda k: C['per_eval_target'][k]['regret_vs_best_available'])
C['worst_regret_vs_matched_target'] = float(
    min(v['regret_vs_matched_target'] for v in C['per_eval_target'].values()))
C['min_enrichment_trained_on_ceiling'] = float(
    min(v['enrichment_trained_on_ceiling'] for v in C['per_eval_target'].values()))

# ---------------------------------------------------------------- D. shortlist stability
D = {'note': ('do the MODEL shortlists agree across training targets? this is the decision the '
              'campaign actually makes'),
     'budget': BUDGET, 'pairwise_top20_overlap_of_model_shortlists': {}}
for a in TARGETS:
    D['pairwise_top20_overlap_of_model_shortlists'][a] = {
        b: float(overlap(MEAN_PRED[a], MEAN_PRED[b], BUDGET)) for b in TARGETS}
D['ceiling_vs_floor_model_shortlist_overlap'] = float(
    overlap(MEAN_PRED[CEILING], MEAN_PRED[FLOOR], BUDGET))
D['data_level_ceiling_vs_floor_overlap_for_reference'] = A['floor_vs_ceiling_top20_overlap']

# ---------------------------------------------------------------- verdict
crit_i = C['min_enrichment_trained_on_ceiling'] > 2.0
crit_ii = C['worst_regret'] >= -0.05
failing = [k for k, v in C['per_eval_target'].items()
           if v['regret_vs_best_available'] < -0.05 or v['enrichment_trained_on_ceiling'] <= 2.0]
VERDICT = {
    'criterion_i_enrichment_above_2x': bool(crit_i),
    'criterion_i_observed_minimum_enrichment': C['min_enrichment_trained_on_ceiling'],
    'criterion_ii_regret_above_minus_0.05': bool(crit_ii),
    'criterion_ii_observed_worst_regret': C['worst_regret'],
    'CLAIM_SUPPORTED': bool(crit_i and crit_ii),
    'failing_eval_targets': failing}
VERDICT['reading'] = (
    'Training on the observed maximum is robust to the gating question: a max-trained model still '
    'ranks catalysts well and still enriches, even when the ground truth is the catalyst floor.'
    if VERDICT['CLAIM_SUPPORTED'] else
    'The target choice is reading-dependent. The question to Prof. Taniike is BLOCKING for the '
    'eval targets listed in failing_eval_targets, not a detail to assume past.')

json.dump({'meta': {'protocol': ('formulation B, composition-only, tuned LGBM, catalyst-grouped '
                                 '5-fold CV, identical folds across variants'),
                    'seeds': SEEDS, 'quantile_grid': QS, 'budget': BUDGET,
                    'ceiling_target': CEILING, 'floor_target': FLOOR,
                    'what_this_does_NOT_do': ('it does not determine whether the 27 within-cell slots '
                                              'are distinct conditions or time-on-stream samples; '
                                              'the CSV cannot answer that (cells are yield-sorted)'),
                    'wording_constraint': ('a low within-cell quantile is the catalyst OBSERVED FLOOR; '
                                           'calling it "sustained performance" assumes monotone '
                                           'deactivation, which is not established'),
                    'preregistered_rule': RULE},
           'A_data_level_agreement': A,
           'B_train_eval_matrix': B,
           'C_regret_of_training_on_ceiling': C,
           'D_shortlist_stability': D,
           'verdict': VERDICT},
          open('phase10_ground_truth_invariance.json', 'w'), indent=1)

# ---------------------------------------------------------------- report
print()
log('A. MODEL-FREE: do the candidate ground truths agree?')
print('   %-8s %10s %12s %12s' % ('target', 'rho_vs_max', 'top20_ovl', 'top10%_ovl'))
for k, v in A['vs_ceiling'].items():
    print('   %-8s %10.4f %12.2f %12.2f' % (k, v['spearman_vs_ceiling'],
          v['top20_overlap_vs_ceiling'], v['top10pct_overlap_vs_ceiling']))

log('B. TRAIN x EVAL Spearman (rows = training target, cols = evaluation ground truth)')
hdr = '   %-8s' % 'train\\eval' + ''.join('%9s' % k for k in TARGETS)
print(hdr)
for a in TARGETS:
    print('   %-8s' % a + ''.join('%9.4f' % B[a][b]['spearman_seedavg'] for b in TARGETS))

log('C. REGRET of having trained on the ceiling, if the truth is the floor')
print('   %-8s %12s %10s %12s %10s %11s' % ('eval', 'trained_max', 'best_tr',
                                              'best_avail', 'regret', 'enrich_max'))
for k, v in C['per_eval_target'].items():
    print('   %-8s %12.4f %10s %12.4f %+10.4f %11.2f' % (k, v['spearman_trained_on_ceiling'],
          v['best_training_target'], v['spearman_trained_on_best_available'],
          v['regret_vs_best_available'], v['enrichment_trained_on_ceiling']))
print('   worst regret %+.4f at %s ; min enrichment %.2fx'
      % (C['worst_regret'], C['worst_regret_at'], C['min_enrichment_trained_on_ceiling']))

log('D. SHORTLIST STABILITY (top-20 overlap of model shortlists)')
print('   ceiling-trained vs floor-trained shortlist overlap: %.2f'
      % D['ceiling_vs_floor_model_shortlist_overlap'])
print('   (model-free ceiling-vs-floor label overlap for reference: %.2f)'
      % D['data_level_ceiling_vs_floor_overlap_for_reference'])

log('VERDICT: CLAIM_SUPPORTED = %s' % VERDICT['CLAIM_SUPPORTED'])
print('   criterion (i) min enrichment %.2fx > 2.0x : %s'
      % (VERDICT['criterion_i_observed_minimum_enrichment'], crit_i))
print('   criterion (ii) worst regret %+.4f >= -0.05 : %s'
      % (VERDICT['criterion_ii_observed_worst_regret'], crit_ii))
if failing: print('   failing eval targets:', failing)
print('   ' + VERDICT['reading'])
log('done -> phase10_ground_truth_invariance.json')
