"""
phase13_condition_value.py -- where do the gas-flow conditions add deployable value?

Four parts, (a)-(d). Every rule is stated here, before any result is computed. Arm C and Cp models are
not retrained: their out-of-fold (OOF) predictions come from phase11_predictions.npz, whose alignment
with the data is asserted on load. The only models trained here are the small ones named in (b) and (c).

DATA AND CONVENTIONS
  Slot = (catalyst, temperature, gas-flow condition): 5 temperatures x 27 conditions = 135 per catalyst.
  Slot yield = mean of the slot's rows. This averages the 1,906 duplicated slots of the 28 doubled
  catalysts (two runs with a run-level offset, phase12).
  Unmeasured slots. A slot with no row (excluded by the lab's mass-balance check, or never run) is
  MISSING in every headline result, never yield 0. phase12 rule R4 found exclusion "not outcome-coupled
  within catalyst by this rule". A zero-imputation variant appears only as a labelled sensitivity check
  that cites that verdict. (a)'s headline is different in kind: it scores a recommended RUN, and a run
  that gave no usable measurement delivered nothing, so that pick counts as 0. It does not impute the
  slot's yield.
  Truth for (a) and (b) = max over the catalyst's measured slot yields. It equals the observed row max
  except for doubled catalysts whose best slot was measured twice; the count is stored. The composition
  models in (b) and the phase11 B baselines are scored against the observed row max, as in phase11.
  Populations: all_917; EE_771 = equal effort (>= 20 rows in some (catalyst, temperature) cell);
  ND_889 = without the 28 doubled catalysts.
  Folds: catalyst_folds(seed, 917), seeds 0, 1, 2, 7, 13 -- the same folds as phase11, so every OOF
  prediction reused here is paired with the folds used here.
  LightGBM: TUNED from grouped_tuning.json (300 trees; its subsample is inert in LightGBM 4.7 and is
  left as is), n_jobs=2, StandardScaler fit on the training fold only.
  Bootstraps. Metric-vs-truth differences: ocm_eval.paired_catalyst_bootstrap on seed-averaged OOF
  predictions, 3000 resamples, seed 7. Differences of per-catalyst quantities (regret, SSE): catalysts
  resampled with replacement, 3000 resamples, RNG seed 7; mean and 95% percentile CI. A CI that includes
  zero reads "no detectable change". Seed-to-seed spread is never used as a test.
  Labels. per_seed = one value per seed. per_seed_mean = their mean. seed_averaged = per-catalyst
  values (or predictions) averaged over seeds first, then summarised. seed_mean_of_pooled_oof = OOF
  scores pooled over the 5 folds of a seed, metric per seed, then mean over seeds.

(a) CONDITION RECOMMENDATION. In seed s every catalyst c is held out once, so the fold model that
  produced grid_C_per_seed[s, c] never saw c. Each strategy ranks c's 135 slots, best first; ties go to
  the lower slot id.
    model_C_per_seed        grid_C_per_seed[s, c].
    model_Cp_seed_averaged  grid_Cp[c], seed-averaged OOF; the same ranking in every seed (label kept).
    global_rule             mean slot yield over the training catalysts of c's fold in seed s (mean over
                            the training catalysts that measured the slot).
    family_rule             the same over training catalysts with c's Support (last token of Name); the
                            global rule if the family has < 5 training catalysts (count stored). A slot no
                            family training catalyst measured is ranked after all others, in global-rule
                            order (count stored).
    model_C_seed_averaged   ADDED CHECK, not in the comparisons: grid_C seed-averaged, so model C and Cp
                            can be compared on the same footing.
  Scoring, k = 1, 3, 5 picks. realised(c) = best yield among the top-k picks.
    headline    an unmeasured pick is a failed run and delivers 0; realised = 0 if all k picks fail.
    optimistic  (label kept) unmeasured picks are skipped; the next-ranked measured slots are taken.
  regret(c) = truth(c) - realised(c) >= 0, averaged over the 5 seeds per catalyst.
  Reported per strategy, scoring, k and population: mean and median of the seed-averaged regret,
  per-seed means and medians, and the share of top-k picks that are unmeasured.
  Comparisons (regret_a - regret_b > 0 means b is better): family - model_C_per_seed,
  global - model_C_per_seed, family - model_Cp_seed_averaged. Per-seed mean edges are stored.
  RULE (a). PASS if, at k = 3, headline scoring, all_917: the 95% CI of mean(regret_family -
  regret_model_C_per_seed) lies above zero AND the per-seed mean of that edge is > 0 in all 5 seeds.
  Otherwise FAIL. The same evaluation at k = 1 and k = 5 is secondary, as is the optimistic scoring.

(b) DESIGNED REDUCED PROTOCOL. In seed s, fold k, K slots are chosen from the TRAINING catalysts of
  that fold only, then the fold's held-out catalysts are scored. The two sets are asserted disjoint.
    greedy          add the slot that maximises Spearman(max over the chosen slots the catalyst
                    measured, truth) over training catalysts; catalysts with no measured chosen slot are
                    left out of the objective. Ties: lower slot id. Nested in K (one run to K = 20).
    most_often_max  the K slots that most often hold a training catalyst's argmax slot; count ties go
                    to the higher training mean slot yield, then the lower slot id.
    random          200 draws of K distinct slots per (seed, fold, K); RNG seeded by (7, seed, fold, K).
  K = 3, 5, 10, 20. Held-out score(c) = max over the chosen slots that were measured for c.
    headline (missing)          scored on held-out catalysts with >= 1 measured chosen slot. Coverage =
                                share of held-out catalysts scored.
    zero_imputed_same_covered   identical to headline by construction (asserted, not stored twice).
    zero_imputed_all            SENSITIVITY CHECK citing phase12 R4: uncovered catalysts score 0; all
                                held-out catalysts scored.
  Metrics: Spearman(score, truth); precision_top10pct = share of the top 10% by score that are in the
  top 10% by truth, both within the scored set. Aggregation: seed_mean_of_pooled_oof. For random, draw
  d of every fold is pooled into one OOF vector per seed, giving 200 seed-mean values per K.
  CAVEAT. Truth is a max over the same single runs the chosen slots are drawn from, so a chosen slot
  that holds a catalyst's max scores it exactly. These numbers are optimistic about prospective use.
  RULE (b). At each K, separately for greedy and most_often_max: the designed protocol is "better than
  random at K" if its seed-mean Spearman (headline scoring, all_917) exceeds the 97.5th percentile of
  the 200 random seed-mean Spearman values at the same K, computed with the same covered-catalyst
  definition. Otherwise "not better than random at K". The 97.5th percentile of the 1,000 per-seed
  random values is stored as a stricter secondary reference; it does not enter the rule.
  COMPOSITION MODEL ON K-SLOT LABELS (descriptive, no rule). Per seed and fold, each training catalyst
  is labelled by max over the fold's greedy K slots (training catalysts with none measured are dropped).
  Diagnostic (no rule): greedy's training objective is also evaluated on most_often_max's K slots, to
  show whether greedy forward selection is a good optimiser of its own objective.
  Formulation-B LightGBM on the 65 composition columns predicts the held-out catalysts. Scored against
  the observed row max on all_917, EE_771, ND_889, per seed and on seed-averaged predictions. Paired
  catalyst bootstrap (Spearman; enrichment on all_917) vs phase11 cat_B_median and cat_B_max. Added
  control: the same pipeline with label = max over all measured slot yields, which separates the
  effect of the reduction from the slot-mean convention. Pipeline check: B_median and B_max are
  retrained here and compared with the npz.

(c) COMPOSITION x CONDITION. Within-cell shape of held-out catalysts, defined the same way for every
  model. Cell = (catalyst, temperature). Eligible cell: >= 8 measured slots. For each measured slot r of
  an eligible cell (slot yields): e_r = (y_r - mean y over the cell's OTHER measured slots) - (p_r - mean
  p over the same slots). within-shape RMSE = sqrt(mean e_r^2) over the population's eligible slots.
  Models (catalyst-grouped, 5 seeds, seed-averaged predictions):
    (1) shared_effect  effect(T, condition) = mean over the training catalysts' eligible cells of
                       (slot yield - cell mean); p = effect. Eligible cells only, so tiny cells (whose
                       deviations are forced towards 0) do not shrink the effect.
    (2) lgbm_C         rows_C collapsed to slot means.
    (3) lgbm_Cp        rows_Cp collapsed to slot means.
    (4) amplitude      per training eligible cell, a_c = least-squares slope of (slot yield - cell mean)
                       on the fold's effect centred over the cell's measured slots; LightGBM (TUNED) on
                       the 65 composition columns + Temperature_C -> a_c; held-out p = a_hat * effect.
    (5) per_cell_linear_LOO  REFERENCE BOUND, not a model for unseen catalysts: per-cell OLS on
                       intercept + total flow + Ar fraction + CH4:O2 (numeric), fitted leaving slot r out
                       and evaluated at r. With an intercept the fitted mean over the other slots equals
                       their mean yield, so e_r = y_r - p_r(fit without r).
    flat (p = 0)       no-shape reference.
  Populations: all_917, EE_771, ND_889 (eligible cells), and complete 27-slot cells of all_917.
  Bootstrap vs lgbm_C: per-catalyst SSE and slot counts, catalysts resampled; RMSE(model) - RMSE(lgbm_C)
  (used by the rule) and the mean per-catalyst SSE difference.
  RULE (c). Further response-surface modelling is worth pursuing only if a model among (1), (3), (4)
  has, on all_917: RMSE(lgbm_C) - RMSE(model) >= 0.05 AND the 95% CI of RMSE(model) - RMSE(lgbm_C)
  entirely below zero. Otherwise: "stop: headroom too small", with the numbers.
  Added checks, not the metric and outside the rule: (i) an in-sample row-level variance split, to
  compare with the audit's "4.6 of 19.9 points"; (ii) the audit's convention (rows instead of slot
  means, cells by row count, and per-cell slopes on the raw flows), to explain differences from it.

(d) ARGMAX TABLE. Descriptive; no rule. Rows, not slot means. Each share is labelled by its statistic.
    S1  the catalyst's single highest row.
    S2  per level of the axis, the mean of the catalyst's (up to) 3 highest rows at that level; the
        level with the highest value.
    S3  per level, the mean of the rows in the best-temperature cell (the cell with the highest max);
        the level with the highest value. Not defined for the temperature axis.
  Ties between levels split the catalyst equally (fractional counts; the number of tied catalysts is
  stored). Ba = Ba anywhere in the Name tokens (a Ba promoter or the BaO support); the promoter-only
  definition is stored as well. Best T for the strata = the S1 temperature (the cell with the highest
  max). Top decile = the 92 catalysts with the highest observed row max.

Output:  phase13_condition_value.json
Usage:   python3 phase13_condition_value.py [--dev]
         --dev = seed 0 only (all 5 folds, since pooled OOF metrics need them), 20 random draws,
         300 resamples, no file written.
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from ocm_eval import (load_with_conditions, lgb_params, cat_metrics, TARGET, COND_COLS,
                      catalyst_folds, equal_effort_mask, design_axes, slot_keys, condition_grid,
                      doubled_catalyst_mask, paired_catalyst_bootstrap, MIN_CELL)

DEV = '--dev' in sys.argv
SEEDS_ALL = [0, 1, 2, 7, 13]
SEEDS = [0] if DEV else SEEDS_ALL
NSEED = len(SEEDS)
N_BOOT = 300 if DEV else 3000
N_RANDOM = 20 if DEV else 200
BOOT_SEED = 7
N_JOBS = 2
KS_REC = [1, 3, 5]
KS_PROT = [3, 5, 10, 20]
MIN_FAMILY = 5
MIN_SHAPE = 8
TEMPS = [700, 750, 800, 850, 900]
OUT_JSON = 'phase13_condition_value.json'

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# =============================================================================== data
df, feat_all, groups, n_cat = load_with_conditions()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
y = df[TARGET].values.astype(float)
N = len(y)
EL = [c for c in feat_all if c not in COND_COLS + ['Temperature_C']]
assert N == 89074 and n_cat == 917 and len(EL) == 65
ax = design_axes(df)
GRID = condition_grid(df)
NS = len(GRID)
key2slot = {k: int(s) for k, s in zip(zip(GRID.Temperature_C.astype(float), GRID.total_flow,
                                          GRID.ar_fraction, GRID.ch4_o2), GRID.slot)}
ROW_SLOT = np.array([key2slot[k] for k in slot_keys(df)])
SL_T = GRID.Temperature_C.values.astype(float)
SL_TOT, SL_AR, SL_R = (GRID[c].values.astype(float) for c in ['total_flow', 'ar_fraction', 'ch4_o2'])
assert NS == 135 and (GRID.slot.values == np.arange(NS)).all() and (SL_T == np.repeat(TEMPS, 27)).all()
for t in range(1, 5):          # the same 27 conditions in the same order at every temperature
    assert (SL_TOT[27 * t:27 * t + 27] == SL_TOT[:27]).all() and (SL_AR[27 * t:27 * t + 27] == SL_AR[:27]).all()
    assert (SL_R[27 * t:27 * t + 27] == SL_R[:27]).all()

Z = np.load('phase11_predictions.npz')
assert (Z['row_catalyst'] == groups).all(), 'npz row_catalyst does not match groups'
assert (Z['row_slot'] == ROW_SLOT).all(), 'npz row_slot does not match slot_keys'
assert Z['seeds'].tolist() == SEEDS_ALL and Z['grid_C_per_seed'].shape == (5, n_cat, NS)
for c in ['Temperature_C', 'total_flow', 'ar_fraction', 'ch4_o2'] + COND_COLS:
    assert np.allclose(Z[f'slot_{c}'], GRID[c].values.astype(np.float32)), c
for a in ['A', 'C', 'Cp']:
    assert Z[f'rows_{a}'].shape == (N,) and Z[f'grid_{a}'].shape == (n_cat, NS)

COMP = df.groupby(groups)[EL].first().reindex(range(n_cat)).values.astype(float)
NAMES = df.groupby(groups).Name.first().reindex(range(n_cat)).values
TOK = [n.split('-') for n in NAMES]
assert all(len(t) == 4 for t in TOK)
SUP = np.array([t[-1] for t in TOK])
FAMS = sorted(set(SUP))
BA_ANY = np.array([('Ba' in t[:3]) or t[3] == 'BaO' for t in TOK])
BA_PROMOTER = np.array(['Ba' in t[:3] for t in TOK])

POPS = {'all_917': np.ones(n_cat, bool),
        'EE_771': equal_effort_mask(groups, df.Temperature_C.values, n_cat, MIN_CELL),
        'ND_889': ~doubled_catalyst_mask(df, groups, n_cat)}
DBL = ~POPS['ND_889']
assert POPS['EE_771'].sum() == 771 and POPS['ND_889'].sum() == 889

KEY = groups * NS + ROW_SLOT
CNT = np.bincount(KEY, minlength=n_cat * NS)
SY = np.where(CNT > 0, np.bincount(KEY, y, minlength=n_cat * NS) / np.maximum(CNT, 1), np.nan).reshape(n_cat, NS)
MEAS = CNT.reshape(n_cat, NS) > 0
assert int((CNT >= 2).sum()) == 1906 and CNT.max() == 2 and MEAS.any(1).all()
assert np.nanmin(SY) >= 0                       # zero-imputation of a max can only lower it
TRUTH = np.nanmax(SY, 1)                        # truth for (a), (b): max over measured slot yields
TRUE_MAX = pd.Series(y).groupby(groups).max().reindex(range(n_cat)).values   # observed row max
assert np.allclose(TRUE_MAX, Z['true_max'], atol=1e-4)
TRUTH_DIFF = ~np.isclose(TRUTH, TRUE_MAX)
assert DBL[TRUTH_DIFF].all()
Y_MEDIAN = (pd.DataFrame({'c': groups, 't': df.Temperature_C.values, 'y': y})
            .groupby(['c', 't']).y.median().groupby(level=0).max().reindex(range(n_cat)).values)
log(f"{N:,} rows, {n_cat} catalysts; EE {POPS['EE_771'].sum()}, ND {POPS['ND_889'].sum()}; npz aligned; "
    f"truth (max of slot means) differs from observed row max for {TRUTH_DIFF.sum()} catalysts (all doubled)")


def slot_label(j):
    return f"{int(SL_T[j])}C tot{SL_TOT[j]:g} Ar{SL_AR[j]:.2f} CH4:O2={SL_R[j]:g}"


def resample_matrix(n):
    """Catalyst resampling indices, 3000 x n, RNG seed 7 (same draws for every comparison of size n)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, (N_BOOT, n))


def reading(lo, hi, pos='a above b', neg='a below b'):
    return pos if lo > 0 else (neg if hi < 0 else 'no detectable change')


def boot_mean_diff(d, mask):
    """Mean over catalysts in mask of a per-catalyst difference d; catalyst bootstrap CI."""
    idx = np.where(mask)[0]
    bm = d[idx[resample_matrix(len(idx))]].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {'n_catalysts': int(len(idx)), 'mean_diff': float(d[idx].mean()), 'boot_mean': float(bm.mean()),
            'ci95_low': float(lo), 'ci95_high': float(hi), 'clears_zero': bool(lo > 0 or hi < 0),
            'reading': reading(lo, hi, 'above zero', 'below zero'), 'n_boot': N_BOOT, 'rng_seed': BOOT_SEED}


def prec_top10(t, p):
    """Same as ocm_eval.cat_metrics precision_top10pct (stable ties, first occurrence first)."""
    K = max(1, int(round(0.10 * len(t))))
    return len(np.intersect1d(np.argsort(-t, kind='stable')[:K], np.argsort(-p, kind='stable')[:K])) / K


def screen(truth, pred, mask):
    """Catalyst screening metrics on the catalysts in mask (ocm_eval.cat_metrics)."""
    m = cat_metrics(np.arange(int(mask.sum())), truth[mask], pred[mask])
    return {'n_catalysts': int(mask.sum()), 'spearman': float(m['spearman_max']),
            'precision_top10pct': float(m['precision_top10pct']),
            'enrichment_top10pct': float(m['enrichment_top10pct'])}


def fit_predict(Xtr, ytr, Xva, seed):
    """TUNED LightGBM, n_jobs=2, train-fold StandardScaler (phase11 run_B pipeline)."""
    sc = StandardScaler().fit(Xtr)
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED, n_jobs=N_JOBS)).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xva))


RES = {'about': {
    'script': 'phase13_condition_value.py', 'dev_run': DEV, 'seeds': SEEDS, 'n_boot': N_BOOT,
    'boot_rng_seed': BOOT_SEED, 'n_random_draws_per_fold': N_RANDOM, 'lightgbm_version': lgb.__version__,
    'tuned_params': TUNED, 'n_jobs': N_JOBS,
    'populations': {k: int(v.sum()) for k, v in POPS.items()},
    'slot_yield': 'mean of the slot\'s rows (averages the 1,906 duplicated slots)',
    'truth_a_b': 'max over the catalyst\'s measured slot yields',
    'truth_composition_models': 'observed row max (as phase11)',
    'n_catalysts_truth_differs_from_observed_row_max': int(TRUTH_DIFF.sum()),
    'n_catalysts_truth_differs_note': ('all are doubled catalysts whose best slot was measured twice; the '
                                       'slot mean is below the higher run'),
    'unmeasured_slots': ('MISSING in headline results; zero-imputation only as a labelled sensitivity check '
                         'citing phase12 exclusion_test.rule_R4 ("not outcome-coupled within catalyst by '
                         'this rule")'),
    'labels': {'per_seed': 'one value per seed', 'per_seed_mean': 'mean of the per-seed values',
               'seed_averaged': 'per-catalyst values or predictions averaged over seeds, then summarised',
               'seed_mean_of_pooled_oof': ('OOF scores pooled over the 5 folds of a seed, metric per seed, '
                                           'then mean over seeds')},
    'test_rule': ('differences judged only by catalyst bootstrap CIs (3000 resamples, seed 7); seed-to-seed '
                  'spread is never a test; a CI including zero reads "no detectable change"')}}

# ---- consistency with phase11 (from the npz; exact reproductions)
P11 = json.load(open('phase11_condition_baseline.json'))
S_SLOTS = Z['fixed_slot_set']
in_S = np.isin(ROW_SLOT, S_SLOTS)
TRUTH_S = np.full(n_cat, -np.inf); np.maximum.at(TRUTH_S, groups[in_S], y[in_S])
S_COMPLETE = MEAS[:, S_SLOTS].all(1)
TRUTH_S[~S_COMPLETE] = np.nan
b_fair = paired_catalyst_bootstrap(TRUTH_S, Z['grid_C'][:, S_SLOTS].max(1), Z['grid_A'][:, S_SLOTS].max(1),
                                   mask=S_COMPLETE, n_boot=3000, seed=7)
p11_fair = P11['screening_bootstrap_seed_averaged']['fixed_slot_set_max']['all_917_S_complete']['C_minus_A']['spearman']
CONS = {'fair_screening_C_minus_A_spearman_465_complete': {
    'this_script_from_npz': {k: b_fair[k] for k in ('n_catalysts', 'delta_point', 'ci95_low', 'ci95_high')},
    'phase11_json': {k: p11_fair[k] for k in ('n_catalysts', 'delta_point', 'ci95_low', 'ci95_high')},
    'note': 'npz stores float32; phase11 bootstrapped float64 predictions'}}
for a in ['B_max', 'B_median']:
    for pn in ['all_917', 'EE_771']:
        mine = screen(TRUE_MAX, Z[f'cat_{a}'].astype(float), POPS[pn])['spearman']
        ref = P11['screening']['measured_rows_max'][a][pn]['seed_averaged_predictions']['spearman']
        CONS[f'{a}_{pn}_spearman_seed_averaged'] = {'this_script_from_npz': mine, 'phase11_json': ref}
        assert abs(mine - ref) < 1e-3, (a, pn, mine, ref)
assert b_fair['n_catalysts'] == 465 and abs(b_fair['delta_point'] - p11_fair['delta_point']) < 1e-3
CONS['row_rmse_per_seed_mean_from_phase11_json'] = {
    a: P11['row_rmse'][a]['all_917']['per_seed_mean']['rmse'] for a in ['A', 'C']}
RES['phase11_consistency'] = CONS
log(f"phase11 consistency: fair screening C-A (465 complete, seed-averaged) {b_fair['delta_point']:+.4f} "
    f"[{b_fair['ci95_low']:+.4f}, {b_fair['ci95_high']:+.4f}] vs phase11 {p11_fair['delta_point']:+.4f} "
    f"[{p11_fair['ci95_low']:+.4f}, {p11_fair['ci95_high']:+.4f}]; B Spearmans reproduced")

# =============================================================================== (a) recommendation
log("(a) condition recommendation")
STRATS = ['model_C_per_seed', 'model_Cp_seed_averaged', 'global_rule', 'family_rule', 'model_C_seed_averaged']
SCORINGS = ['headline', 'optimistic']
GC_SEED = Z['grid_C_per_seed'].astype(float)
GCP_AVG = Z['grid_Cp'].astype(float)
GC_AVG = Z['grid_C'].astype(float)
REG = {st: {sc: {k: np.zeros((NSEED, n_cat)) for k in KS_REC} for sc in SCORINGS} for st in STRATS}
UNM = {st: {k: np.zeros((NSEED, n_cat)) for k in KS_REC} for st in STRATS}
FAM = {'n_family_fold_pairs': 0, 'n_family_fold_pairs_fallback_to_global': 0,
       'n_catalyst_seed_pairs_fallback_to_global': 0, 'n_catalyst_seed_slot_triples_family_unmeasured': 0,
       'min_family_training_catalysts': 10 ** 9}


def rule_scores(seed):
    """Global and family rule slot scores for every catalyst, from its fold's training catalysts."""
    fc = catalyst_folds(seed, n_cat)
    G = np.empty((n_cat, NS)); F = np.empty((n_cat, NS))
    for k in range(5):
        tr, va = fc != k, fc == k
        assert not (tr & va).any()
        gm = np.nanmean(SY[tr], 0)
        assert np.isfinite(gm).all()
        G[va] = gm
        for f in FAMS:
            vf = va & (SUP == f)
            if not vf.any():
                continue
            trf = tr & (SUP == f)
            FAM['n_family_fold_pairs'] += 1
            FAM['min_family_training_catalysts'] = min(FAM['min_family_training_catalysts'], int(trf.sum()))
            if trf.sum() < MIN_FAMILY:
                F[vf] = gm
                FAM['n_family_fold_pairs_fallback_to_global'] += 1
                FAM['n_catalyst_seed_pairs_fallback_to_global'] += int(vf.sum())
                continue
            fm = np.nanmean(SY[trf], 0)
            miss = ~np.isfinite(fm)
            FAM['n_catalyst_seed_slot_triples_family_unmeasured'] += int(miss.sum() * vf.sum())
            F[vf] = np.where(miss, gm - 1e6, fm)       # unmeasured-in-family slots last, in global order
    return G, F


def realise(score):
    """Per catalyst: headline realised, optimistic realised and n unmeasured among top-k, for each k."""
    order = np.argsort(-score, axis=1, kind='stable')
    V = np.take_along_axis(SY, order, 1)
    M = ~np.isnan(V)
    cm = np.cumsum(M, 1)
    out = {}
    for k in KS_REC:
        head = np.nan_to_num(V[:, :k], nan=0.0).max(1)
        opt = np.where(M & (cm <= k), V, -np.inf).max(1)
        out[k] = (head, opt, (~M[:, :k]).sum(1))
    return out


for i, s in enumerate(SEEDS):
    si = SEEDS_ALL.index(s)
    G, F = rule_scores(s)
    scores = {'model_C_per_seed': GC_SEED[si], 'model_Cp_seed_averaged': GCP_AVG, 'global_rule': G,
              'family_rule': F, 'model_C_seed_averaged': GC_AVG}
    for st, sc in scores.items():
        for k, (h, o, u) in realise(sc).items():
            REG[st]['headline'][k][i] = TRUTH - h
            REG[st]['optimistic'][k][i] = TRUTH - o
            UNM[st][k][i] = u
for st in STRATS:
    for sc in SCORINGS:
        for k in KS_REC:
            assert REG[st][sc][k].min() >= -1e-9
            if sc == 'optimistic':                       # optimistic realised >= headline realised
                assert (REG[st]['optimistic'][k] <= REG[st]['headline'][k] + 1e-9).all()


def reg_summary(R, m):
    avg = R.mean(0)
    return {'n_catalysts': int(m.sum()),
            'mean_regret_seed_averaged': float(avg[m].mean()),
            'median_regret_seed_averaged': float(np.median(avg[m])),
            'per_seed_mean_regret': [float(R[i, m].mean()) for i in range(NSEED)],
            'per_seed_median_regret': [float(np.median(R[i, m])) for i in range(NSEED)]}


SUM_A = {sc: {f'k{k}': {pn: {st: reg_summary(REG[st][sc][k], m) for st in STRATS}
                        for pn, m in POPS.items()} for k in KS_REC} for sc in SCORINGS}
UNM_A = {f'k{k}': {pn: {st: float(UNM[st][k][:, m].sum() / (k * NSEED * m.sum())) for st in STRATS}
                   for pn, m in POPS.items()} for k in KS_REC}
MEAN_TRUTH = {pn: float(TRUTH[m].mean()) for pn, m in POPS.items()}

PAIRS_A = [('family_rule', 'model_C_per_seed'), ('global_rule', 'model_C_per_seed'),
           ('family_rule', 'model_Cp_seed_averaged')]
CMP_A = {}
for sc in SCORINGS:
    CMP_A[sc] = {}
    for k in KS_REC:
        CMP_A[sc][f'k{k}'] = {}
        for pn, m in POPS.items():
            CMP_A[sc][f'k{k}'][pn] = {}
            for a, b in PAIRS_A:
                d = REG[a][sc][k].mean(0) - REG[b][sc][k].mean(0)
                r = boot_mean_diff(d, m)
                r['what'] = f'mean over catalysts of seed-averaged regret_{a} - regret_{b}; > 0 means {b} better'
                r['per_seed_mean_edge'] = [float((REG[a][sc][k][i] - REG[b][sc][k][i])[m].mean())
                                           for i in range(NSEED)]
                CMP_A[sc][f'k{k}'][pn][f'{a}_minus_{b}'] = r


def eval_rule_a(k, sc='headline', pn='all_917'):
    b = CMP_A[sc][f'k{k}'][pn]['family_rule_minus_model_C_per_seed']
    ci_ok = b['ci95_low'] > 0
    n_pos = sum(e > 0 for e in b['per_seed_mean_edge'])
    seeds_ok = len(b['per_seed_mean_edge']) == 5 and n_pos == 5
    return {'k': k, 'scoring': sc, 'population': pn, 'mean_edge': b['mean_diff'],
            'ci95': [b['ci95_low'], b['ci95_high']], 'ci_above_zero': bool(ci_ok),
            'n_seeds_edge_positive': int(n_pos), 'n_seeds': len(b['per_seed_mean_edge']),
            'verdict': 'PASS' if (ci_ok and seeds_ok) else 'FAIL'}


RULE_A = {'rule': ('PASS if, at k = 3, headline scoring, all_917: the 95% CI of mean(regret_family - '
                   'regret_model_C_per_seed) lies above zero AND the per-seed mean edge is > 0 in all 5 seeds'),
          'primary_k3_headline_all_917': eval_rule_a(3),
          'secondary_same_evaluation': [eval_rule_a(1), eval_rule_a(5)] +
                                       [eval_rule_a(k, 'optimistic') for k in KS_REC]}
RECOVERY = {}
for sc in SCORINGS:
    for k in KS_REC:
        g_, f_, c_ = (SUM_A[sc][f'k{k}']['all_917'][st]['mean_regret_seed_averaged']
                      for st in ('global_rule', 'family_rule', 'model_C_per_seed'))
        RECOVERY[f'{sc}_k{k}'] = float((g_ - f_) / (g_ - c_)) if g_ != c_ else None
RES['a_condition_recommendation'] = {
    'truth': 'max over measured slot yields', 'mean_truth': MEAN_TRUTH,
    'family_rule_fallback_and_fill_counts': FAM,
    'regret': SUM_A, 'share_of_top_k_picks_unmeasured': UNM_A,
    'share_unmeasured_note': 'pooled over seeds and catalysts of the population; the headline picks',
    'comparisons_catalyst_bootstrap': CMP_A,
    'family_rule_share_of_model_C_gain_over_global': {
        'what': ('(mean regret global - family) / (mean regret global - model_C_per_seed), all_917, '
                 'seed-averaged regret; descriptive ratio, no CI'), **RECOVERY},
    'rule_a': RULE_A}
r3 = RULE_A['primary_k3_headline_all_917']
log(f"(a) rule: k=3 headline all_917 family - model_C_per_seed {r3['mean_edge']:+.3f} "
    f"[{r3['ci95'][0]:+.3f}, {r3['ci95'][1]:+.3f}], seeds positive {r3['n_seeds_edge_positive']}/"
    f"{r3['n_seeds']} -> {r3['verdict']}")

# =============================================================================== (b) reduced protocol
log("(b) designed reduced protocol")
KMAX = max(KS_PROT)


def greedy_choose(SYt, Tt, kmax):
    """Greedy forward selection on training catalysts only. Returns slots and the training trace."""
    chosen, cur, trace = [], np.full(len(Tt), np.nan), []
    for _ in range(kmax):
        best, bj = -np.inf, -1
        for j in range(NS):
            if j in chosen:
                continue
            new = np.fmax(cur, SYt[:, j])
            m = ~np.isnan(new)
            if m.sum() < 3:
                continue
            r = spearmanr(new[m], Tt[m])[0]
            if np.isfinite(r) and r > best:
                best, bj = r, j
        chosen.append(bj)
        cur = np.fmax(cur, SYt[:, bj])
        trace.append({'train_spearman': float(best), 'train_coverage': float(np.isfinite(cur).mean())})
    return chosen, trace


def most_often_max(SYt, kmax):
    am = np.nanargmax(SYt, 1)
    counts = np.bincount(am, minlength=NS)
    gm = np.nanmean(SYt, 0)
    order = np.lexsort((np.arange(NS), -gm, -counts))
    return order[:kmax].tolist(), counts


def train_objective(SYt, Tt, sl):
    """The greedy objective evaluated for any slot list, on training catalysts (diagnostic)."""
    v = np.fmax.reduce(SYt[:, sl], axis=1)
    m = np.isfinite(v)
    return {'train_spearman': float(spearmanr(v[m], Tt[m])[0]), 'train_coverage': float(m.mean())}


def nanmax_rows(A):
    return np.fmax.reduce(A, axis=-1)          # NaN only where every entry is NaN


CHOOSERS = ['greedy', 'most_often_max']
DSCORE = {ch: {K: np.full((NSEED, n_cat), np.nan) for K in KS_PROT} for ch in CHOOSERS}
RSCORE = {K: np.full((NSEED, N_RANDOM, n_cat), np.nan) for K in KS_PROT}
CPRED = {K: np.full((NSEED, n_cat), np.nan) for K in KS_PROT}
CPRED_ALLSLOT = np.full((NSEED, n_cat), np.nan)
BCHECK = {'B_median': np.full((NSEED, n_cat), np.nan), 'B_max': np.full((NSEED, n_cat), np.nan)}
CHOICES, LABEL_KEPT = [], {K: [] for K in KS_PROT}
for i, s in enumerate(SEEDS):
    fc = catalyst_folds(s, n_cat)
    for k in range(5):
        tr, va = np.where(fc != k)[0], np.where(fc == k)[0]
        assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == n_cat   # choice vs held-out
        SYtr, Ttr = SY[tr], TRUTH[tr]                                             # chooser sees only these
        g_sl, g_tr = greedy_choose(SYtr, Ttr, KMAX)
        m_sl, m_cnt = most_often_max(SYtr, KMAX)
        CHOICES.append({'seed': s, 'fold': k, 'n_train_catalysts': int(len(tr)),
                        'n_heldout_catalysts': int(len(va)), 'greedy_slots': g_sl, 'greedy_train_trace': g_tr,
                        'most_often_max_slots': m_sl,
                        'most_often_max_counts': [int(m_cnt[j]) for j in m_sl],
                        'most_often_max_train_objective_at_K': {
                            f'K{K}': train_objective(SYtr, Ttr, m_sl[:K]) for K in KS_PROT}})
        SYva = SY[va]
        for K in KS_PROT:
            DSCORE['greedy'][K][i, va] = nanmax_rows(SYva[:, g_sl[:K]])
            DSCORE['most_often_max'][K][i, va] = nanmax_rows(SYva[:, m_sl[:K]])
            rng = np.random.default_rng([BOOT_SEED, s, k, K])
            draws = np.stack([rng.choice(NS, K, replace=False) for _ in range(N_RANDOM)])
            RSCORE[K][i][:, va] = nanmax_rows(SYva[:, draws]).T
            lab = nanmax_rows(SYtr[:, g_sl[:K]])
            keep = np.isfinite(lab)
            LABEL_KEPT[K].append(float(keep.mean()))
            CPRED[K][i, va] = fit_predict(COMP[tr][keep], lab[keep], COMP[va], s)
        CPRED_ALLSLOT[i, va] = fit_predict(COMP[tr], TRUTH[tr], COMP[va], s)
        BCHECK['B_median'][i, va] = fit_predict(COMP[tr], Y_MEDIAN[tr], COMP[va], s)
        BCHECK['B_max'][i, va] = fit_predict(COMP[tr], TRUE_MAX[tr], COMP[va], s)
    log(f"  seed {s}: greedy K=3 picks by fold " +
        '; '.join(','.join(str(j) for j in c['greedy_slots'][:3]) for c in CHOICES if c['seed'] == s))
for ch in CHOOSERS:
    for K in KS_PROT:
        assert not np.isnan(DSCORE[ch][K]).all(1).any()


def prot_metrics(sc, m):
    """headline (missing): covered catalysts in m; zero_imputed_all: every catalyst in m, uncovered = 0."""
    cov = np.isfinite(sc) & m
    head = screen(TRUTH, sc, cov)
    zsame = screen(TRUTH, np.nan_to_num(sc, nan=0.0), cov)
    assert zsame == head, 'zero-imputation on covered catalysts must equal headline'
    zall = screen(TRUTH, np.nan_to_num(sc, nan=0.0), m)
    return {'coverage': float(cov.sum() / m.sum()), 'headline_missing': head, 'zero_imputed_all': zall}


PROT = {}
for ch in CHOOSERS:
    PROT[ch] = {}
    for K in KS_PROT:
        PROT[ch][f'K{K}'] = {}
        for pn, m in POPS.items():
            per = [prot_metrics(DSCORE[ch][K][i], m) for i in range(NSEED)]
            PROT[ch][f'K{K}'][pn] = {
                'per_seed': per,
                'seed_mean_of_pooled_oof': {
                    'coverage': float(np.mean([p['coverage'] for p in per])),
                    'headline_missing_spearman': float(np.mean([p['headline_missing']['spearman'] for p in per])),
                    'headline_missing_precision_top10pct': float(np.mean([p['headline_missing']['precision_top10pct'] for p in per])),
                    'zero_imputed_all_spearman': float(np.mean([p['zero_imputed_all']['spearman'] for p in per])),
                    'zero_imputed_all_precision_top10pct': float(np.mean([p['zero_imputed_all']['precision_top10pct'] for p in per]))}}

# fast precision must equal cat_metrics on the designed scores (checked, then used for random draws)
_sc = DSCORE['greedy'][3][0]; _cv = np.isfinite(_sc)
assert prec_top10(TRUTH[_cv], _sc[_cv]) == screen(TRUTH, _sc, _cv)['precision_top10pct']
_z = np.nan_to_num(_sc, nan=0.0)
assert prec_top10(TRUTH, _z) == screen(TRUTH, _z, np.ones(n_cat, bool))['precision_top10pct']

RAND = {}
for K in KS_PROT:
    RAND[f'K{K}'] = {}
    for pn, m in POPS.items():
        A = {q: np.empty((NSEED, N_RANDOM)) for q in ('cov', 'sp_h', 'pr_h', 'sp_z', 'pr_z')}
        for i in range(NSEED):
            for d in range(N_RANDOM):
                sc = RSCORE[K][i, d]
                cov = np.isfinite(sc) & m
                z = np.nan_to_num(sc, nan=0.0)
                A['cov'][i, d] = cov.sum() / m.sum()
                A['sp_h'][i, d] = spearmanr(sc[cov], TRUTH[cov])[0]
                A['pr_h'][i, d] = prec_top10(TRUTH[cov], sc[cov])
                A['sp_z'][i, d] = spearmanr(z[m], TRUTH[m])[0]
                A['pr_z'][i, d] = prec_top10(TRUTH[m], z[m])
        smean = {q: v.mean(0) for q, v in A.items()}        # seed mean per draw index: N_RANDOM values
        RAND[f'K{K}'][pn] = {
            'what': (f'{N_RANDOM} draws; draw d of every fold pooled per seed; seed_mean = mean over seeds '
                     'per draw index'),
            'coverage_mean': float(A['cov'].mean()),
            'headline_missing_spearman': {
                'seed_mean_draws_mean': float(smean['sp_h'].mean()),
                'seed_mean_draws_p2_5': float(np.percentile(smean['sp_h'], 2.5)),
                'seed_mean_draws_p97_5': float(np.percentile(smean['sp_h'], 97.5)),
                'per_seed_draws_p97_5_stricter_secondary': float(np.percentile(A['sp_h'], 97.5))},
            'headline_missing_precision_top10pct': {
                'seed_mean_draws_mean': float(smean['pr_h'].mean()),
                'seed_mean_draws_p97_5': float(np.percentile(smean['pr_h'], 97.5))},
            'zero_imputed_all_spearman': {
                'seed_mean_draws_mean': float(smean['sp_z'].mean()),
                'seed_mean_draws_p97_5': float(np.percentile(smean['sp_z'], 97.5))},
            'zero_imputed_all_precision_top10pct': {
                'seed_mean_draws_mean': float(smean['pr_z'].mean()),
                'seed_mean_draws_p97_5': float(np.percentile(smean['pr_z'], 97.5))}}
log("  random draws scored")

RULE_B = {'rule': ('designed protocol "better than random at K" if its seed-mean Spearman (headline missing '
                   'scoring, all_917) exceeds the 97.5th percentile of the random seed-mean Spearman values at '
                   'the same K'), 'results': {}}
for ch in CHOOSERS:
    for K in KS_PROT:
        dval = PROT[ch][f'K{K}']['all_917']['seed_mean_of_pooled_oof']['headline_missing_spearman']
        rr = RAND[f'K{K}']['all_917']['headline_missing_spearman']
        RULE_B['results'][f'{ch}_K{K}'] = {
            'designed_seed_mean_spearman': dval, 'random_seed_mean_p97_5': rr['seed_mean_draws_p97_5'],
            'random_seed_mean_mean': rr['seed_mean_draws_mean'],
            'verdict': ('better than random at K' if dval > rr['seed_mean_draws_p97_5']
                        else 'not better than random at K'),
            'secondary_vs_per_seed_p97_5': bool(dval > rr['per_seed_draws_p97_5_stricter_secondary'])}

# slot choice summary
slot_freq = {}
for K in (3, 5):
    for ch, key in (('greedy', 'greedy_slots'), ('most_often_max', 'most_often_max_slots')):
        cnt = pd.Series([j for c in CHOICES for j in c[key][:K]]).value_counts()
        slot_freq[f'{ch}_K{K}'] = [{'slot': int(j), 'label': slot_label(j), 'n_folds_chosen': int(v)}
                                   for j, v in cnt.items()]
n_folds_total = len(CHOICES)
TRAIN_OBJ = {'what': ('the greedy objective (Spearman of max over chosen measured slots vs truth, covered '
                      'training catalysts) for each chooser\'s own K slots; mean over seed-fold pairs; '
                      'training data only, diagnostic'),
             **{f'K{K}': {'greedy': float(np.mean([c['greedy_train_trace'][K - 1]['train_spearman'] for c in CHOICES])),
                          'most_often_max': float(np.mean([c['most_often_max_train_objective_at_K'][f'K{K}']['train_spearman']
                                                           for c in CHOICES])),
                          'greedy_coverage': float(np.mean([c['greedy_train_trace'][K - 1]['train_coverage'] for c in CHOICES])),
                          'most_often_max_coverage': float(np.mean([c['most_often_max_train_objective_at_K'][f'K{K}']['train_coverage']
                                                                    for c in CHOICES]))} for K in KS_PROT}}

# composition model on K-slot labels
B_NPZ = {'B_median': Z['cat_B_median'].astype(float), 'B_max': Z['cat_B_max'].astype(float)}
COMPM = {}
variants = {f'greedy_K{K}_label': CPRED[K] for K in KS_PROT}
variants['all_measured_slots_label_control'] = CPRED_ALLSLOT
for vn, P in variants.items():
    Pa = P.mean(0)
    blk = {}
    for pn, m in POPS.items():
        per = [screen(TRUE_MAX, P[i], m) for i in range(NSEED)]
        blk[pn] = {'per_seed': per,
                   'per_seed_mean': {q: float(np.mean([p[q] for p in per]))
                                     for q in ('spearman', 'precision_top10pct', 'enrichment_top10pct')},
                   'seed_averaged_predictions': screen(TRUE_MAX, Pa, m)}
        for bn in ('B_median', 'B_max'):
            mets = ['spearman', 'enrichment'] if pn == 'all_917' else ['spearman']
            for met in mets:
                bb = paired_catalyst_bootstrap(TRUE_MAX, Pa, B_NPZ[bn], mask=m, n_boot=N_BOOT,
                                               seed=BOOT_SEED, metric=met)
                bb['reading'] = reading(bb['ci95_low'], bb['ci95_high'], f'above {bn}', f'below {bn}')
                blk[pn][f'minus_{bn}_{met}_seed_averaged'] = bb
    COMPM[vn] = blk
COMPM['training_label_coverage'] = {
    f'greedy_K{K}': {'mean_share_training_catalysts_kept': float(np.mean(LABEL_KEPT[K])),
                     'min_share': float(np.min(LABEL_KEPT[K]))} for K in KS_PROT}
COMPM['B_baselines_from_npz_seed_averaged'] = {
    bn: {pn: screen(TRUE_MAX, B_NPZ[bn], m) for pn, m in POPS.items()} for bn in B_NPZ}
if not DEV:
    COMPM['pipeline_check_B_retrained_here_vs_npz'] = {
        bn: {'max_abs_diff_seed_averaged_predictions': float(np.abs(BCHECK[bn].mean(0) - B_NPZ[bn]).max()),
             'spearman_all_917_retrained': screen(TRUE_MAX, BCHECK[bn].mean(0), POPS['all_917'])['spearman']}
        for bn in BCHECK}
else:
    COMPM['pipeline_check_B_retrained_here_vs_npz'] = 'skipped in --dev (npz holds 5-seed averages)'

RES['b_reduced_protocol'] = {
    'caveat': ('Truth is a max over the same single runs the chosen slots are drawn from, so a chosen slot '
               'that holds a catalyst\'s max scores it exactly. These numbers are optimistic about '
               'prospective use.'),
    'truth': 'max over measured slot yields (composition models: observed row max)',
    'zero_imputation_note': ('zero_imputed_all is a SENSITIVITY CHECK: an uncovered catalyst scores 0. '
                             'phase12 exclusion_test.rule_R4: "not outcome-coupled within catalyst by this '
                             'rule", so unmeasured slots are treated as missing in the headline. '
                             'Zero-imputation on the covered catalysts equals the headline (asserted).'),
    'aggregation': 'seed_mean_of_pooled_oof',
    'designed': PROT, 'random': RAND, 'rule_b': RULE_B,
    'slot_choice_frequency': {'n_seed_fold_pairs': n_folds_total, **slot_freq},
    'training_objective_by_chooser': TRAIN_OBJ,
    'choices_per_seed_fold': CHOICES,
    'composition_model_on_K_slot_labels': COMPM}
log("(b) rule: " + '; '.join(f"{k} {v['designed_seed_mean_spearman']:.3f} vs p97.5 "
                             f"{v['random_seed_mean_p97_5']:.3f}" for k, v in RULE_B['results'].items()))

# =============================================================================== (c) shape
log("(c) composition x condition")
SY3, M3 = SY.reshape(n_cat, 5, 27), MEAS.reshape(n_cat, 5, 27)
NC = M3.sum(2)
ELIG, COMPLETE = NC >= MIN_SHAPE, NC == 27
CM = np.where(M3, SY3, 0.0).sum(2) / np.maximum(NC, 1)            # cell mean over measured slots


def shape_err(P3):
    """e_r for every measured slot of an eligible cell; NaN elsewhere. P3: (n_cat, 5, 27)."""
    yv = np.where(M3, SY3, 0.0); pv = np.where(M3, P3, 0.0)
    nm1 = np.maximum(NC - 1, 1)[..., None]
    e = (yv - (yv.sum(2, keepdims=True) - yv) / nm1) - (pv - (pv.sum(2, keepdims=True) - pv) / nm1)
    return np.where(M3 & ELIG[..., None], e, np.nan)


def fold_effect(trm):
    valid = M3 & (ELIG & trm[:, None])[..., None]
    D = np.where(valid, SY3 - CM[..., None], 0.0)
    n = valid.sum(0)
    return np.where(n > 0, D.sum(0) / np.maximum(n, 1), 0.0), int((n == 0).sum())


def amplitudes(eff):
    E = np.broadcast_to(eff[None], SY3.shape)
    em = np.where(M3, E, 0.0).sum(2) / np.maximum(NC, 1)
    CE = np.where(M3, E - em[..., None], 0.0)
    DY = np.where(M3, SY3 - CM[..., None], 0.0)
    den = (CE ** 2).sum(2)
    return np.where(ELIG & (den > 0), (DY * CE).sum(2) / np.where(den > 0, den, 1.0), np.nan)


def collapse(rowpred):
    s_ = np.bincount(KEY, rowpred, minlength=n_cat * NS)
    return np.where(CNT > 0, s_ / np.maximum(CNT, 1), np.nan).reshape(n_cat, 5, 27)


P_SH = np.zeros((NSEED, n_cat, 5, 27)); P_AMP = np.zeros((NSEED, n_cat, 5, 27))
n_eff_empty, amp_train_cells, amp_r = 0, [], []
for i, s in enumerate(SEEDS):
    fc = catalyst_folds(s, n_cat)
    for k in range(5):
        tr, va = fc != k, fc == k
        eff, ne = fold_effect(tr)
        n_eff_empty += ne
        P_SH[i, va] = eff
        A = amplitudes(eff)
        ct = np.argwhere(ELIG & tr[:, None] & np.isfinite(A))
        cv = np.argwhere(np.broadcast_to(va[:, None], (n_cat, 5)))
        Xt = np.hstack([COMP[ct[:, 0]], np.array(TEMPS, float)[ct[:, 1]][:, None]])
        Xv = np.hstack([COMP[cv[:, 0]], np.array(TEMPS, float)[cv[:, 1]][:, None]])
        ah = fit_predict(Xt, A[ct[:, 0], ct[:, 1]], Xv, s)
        AH = np.zeros((n_cat, 5)); AH[cv[:, 0], cv[:, 1]] = ah
        P_AMP[i, va] = AH[va][..., None] * eff[None]
        amp_train_cells.append(len(ct))
        hv = ELIG & va[:, None] & np.isfinite(A)
        amp_r.append(float(np.corrcoef(AH[hv], A[hv])[0, 1]))
PRED_C = {'shared_effect': P_SH.mean(0), 'lgbm_C': collapse(Z['rows_C'].astype(float)),
          'lgbm_Cp': collapse(Z['rows_Cp'].astype(float)), 'amplitude': P_AMP.mean(0),
          'flat_p0': np.zeros((n_cat, 5, 27))}
E_C = {m: shape_err(P) for m, P in PRED_C.items()}

# (5) per-cell linear model, leave one slot out
XC = np.column_stack([np.ones(27), SL_TOT[:27], SL_AR[:27], SL_R[:27]])
E_LOO = np.full((n_cat, 5, 27), np.nan)
n_rank_def, n_loo = 0, 0
for c, ti in np.argwhere(ELIG):
    idx = np.where(M3[c, ti])[0]
    yv, X = SY3[c, ti, idx], XC[idx]
    for j in range(len(idx)):
        o = np.arange(len(idx)) != j
        beta, _, rank, _ = np.linalg.lstsq(X[o], yv[o], rcond=None)
        n_rank_def += rank < 4; n_loo += 1
        E_LOO[c, ti, idx[j]] = yv[j] - X[j] @ beta
E_C['per_cell_linear_LOO_reference_bound'] = E_LOO
ref_mask = np.isfinite(E_C['lgbm_C'])
for m, E in E_C.items():
    assert (np.isfinite(E) == ref_mask).all(), m
MODELS_C = list(E_C)

SHAPE_POPS = {'all_917': (POPS['all_917'], ELIG), 'EE_771': (POPS['EE_771'], ELIG),
              'ND_889': (POPS['ND_889'], ELIG), 'complete_27_slot_cells_all_917': (POPS['all_917'], COMPLETE)}


def sse_cnt(E, cm, cellm):
    sel = np.isfinite(E) & (cm[:, None] & cellm)[..., None]
    return np.where(sel, E ** 2, 0.0).sum((1, 2)), sel.sum((1, 2))


def boot_rmse_diff(sa, sb, cnt):
    idx = np.where(cnt > 0)[0]
    I = idx[resample_matrix(len(idx))]
    f = lambda a, b, n: np.sqrt(a / n) - np.sqrt(b / n)
    pt = f(sa[idx].sum(), sb[idx].sum(), cnt[idx].sum())
    bd = f(sa[I].sum(1), sb[I].sum(1), cnt[I].sum(1))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    dm = (sa - sb)[I].mean(1)
    dlo, dhi = np.percentile(dm, [2.5, 97.5])
    return {'n_catalysts': int(len(idx)), 'rmse_diff_point': float(pt), 'rmse_diff_boot_mean': float(bd.mean()),
            'ci95_low': float(lo), 'ci95_high': float(hi), 'clears_zero': bool(lo > 0 or hi < 0),
            'reading': reading(lo, hi, 'model worse than lgbm_C', 'model better than lgbm_C'),
            'mean_per_catalyst_sse_diff': float((sa - sb)[idx].mean()),
            'mean_per_catalyst_sse_diff_ci95': [float(dlo), float(dhi)], 'n_boot': N_BOOT}


SHAPE = {}
for pn, (cm, cellm) in SHAPE_POPS.items():
    S = {m: sse_cnt(E, cm, cellm) for m, E in E_C.items()}
    n_slots = int(S['lgbm_C'][1].sum())
    blk = {'n_catalysts': int((S['lgbm_C'][1] > 0).sum()), 'n_cells': int((cm[:, None] & cellm).sum()),
           'n_slots': n_slots,
           'within_shape_rmse_seed_averaged': {m: float(np.sqrt(v[0].sum() / n_slots)) for m, v in S.items()}}
    null = S['flat_p0'][0].sum()
    blk['share_of_flat_sse_explained'] = {m: float(1 - v[0].sum() / null) for m, v in S.items()}
    blk['vs_lgbm_C_catalyst_bootstrap'] = {f'{m}_minus_lgbm_C': boot_rmse_diff(S[m][0], S['lgbm_C'][0], S['lgbm_C'][1])
                                           for m in MODELS_C if m != 'lgbm_C'}
    r_ = blk['within_shape_rmse_seed_averaged']
    gap = r_['shared_effect'] - r_['per_cell_linear_LOO_reference_bound']
    blk['lgbm_C_share_of_shared_to_reference_gap_closed'] = (
        float((r_['shared_effect'] - r_['lgbm_C']) / gap) if gap != 0 else None)
    SHAPE[pn] = blk

RULE_C = {'rule': ('pursue only if a model among shared_effect, lgbm_Cp, amplitude has, on all_917: RMSE(lgbm_C) '
                   '- RMSE(model) >= 0.05 AND the 95% CI of RMSE(model) - RMSE(lgbm_C) entirely below zero'),
          'candidates': {}}
for m in ['shared_effect', 'lgbm_Cp', 'amplitude']:
    b = SHAPE['all_917']['vs_lgbm_C_catalyst_bootstrap'][f'{m}_minus_lgbm_C']
    imp = -b['rmse_diff_point']
    RULE_C['candidates'][m] = {'improvement_over_lgbm_C': float(imp), 'ci95_of_model_minus_lgbm_C':
                               [b['ci95_low'], b['ci95_high']],
                               'meets_rule': bool(imp >= 0.05 and b['ci95_high'] < 0)}
_rs = SHAPE['all_917']['within_shape_rmse_seed_averaged']
if any(v['meets_rule'] for v in RULE_C['candidates'].values()):
    RULE_C['verdict'] = 'pursue: ' + ', '.join(m for m, v in RULE_C['candidates'].items() if v['meets_rule'])
else:
    RULE_C['verdict'] = 'stop: headroom too small'
RULE_C['numbers'] = (f"all_917 within-shape RMSE (seed-averaged): lgbm_C {_rs['lgbm_C']:.3f}, lgbm_Cp "
                     f"{_rs['lgbm_Cp']:.3f}, amplitude {_rs['amplitude']:.3f}, shared_effect "
                     f"{_rs['shared_effect']:.3f}; reference bound (per-cell LOO, uses the catalyst's own data) "
                     f"{_rs['per_cell_linear_LOO_reference_bound']:.3f}; flat {_rs['flat_p0']:.3f}")

# in-sample row-level framing, for comparison with the audit's "4.6 of 19.9 points"
TI_ROW = ROW_SLOT // 27
CELL_ROW = groups * 5 + TI_ROW
cmr = (np.bincount(CELL_ROW, y) / np.maximum(np.bincount(CELL_ROW), 1))[CELL_ROW]
dev_r = y - cmr
eff_r = (np.bincount(ROW_SLOT, dev_r, minlength=NS) / np.bincount(ROW_SLOT, minlength=NS))[ROW_SLOT]
fit_r = eff_r - (np.bincount(CELL_ROW, eff_r) / np.maximum(np.bincount(CELL_ROW), 1))[CELL_ROW]
TSS, WSS, RSS = ((y - y.mean()) ** 2).sum(), (dev_r ** 2).sum(), ((dev_r - fit_r) ** 2).sum()
INSAMPLE = {'what': ('IN-SAMPLE, descriptive: all rows of all_917, cell = (catalyst, temperature); shared effect '
                     'per (temperature, condition) = mean row deviation from its cell mean over all cells; '
                     'centred within cell. Not out-of-fold, not the shape metric.'),
            'within_cell_pct_of_total_variance': float(100 * WSS / TSS),
            'pct_points_explained_by_shared_effect': float(100 * (WSS - RSS) / TSS)}

# audit-convention check (rows, not slot means): explains the gap to the audit's numbers; not the metric
NROW_CELL = np.bincount(CELL_ROW, minlength=n_cat * 5)


def row_shape_err(p):
    sy_, sp_ = np.bincount(CELL_ROW, y, minlength=n_cat * 5), np.bincount(CELL_ROW, p, minlength=n_cat * 5)
    nm1 = np.maximum(NROW_CELL[CELL_ROW] - 1, 1).astype(float)
    return (y - (sy_[CELL_ROW] - y) / nm1) - (p - (sp_[CELL_ROW] - p) / nm1)


def row_loo(Xr, cells):
    out = np.full(N, np.nan)
    o_ = np.argsort(CELL_ROW, kind='stable')
    bnd = np.r_[0, np.cumsum(np.bincount(CELL_ROW[o_], minlength=n_cat * 5))]
    for ci in cells:
        idx = o_[bnd[ci]:bnd[ci + 1]]
        X, yy = Xr[idx], y[idx]
        for j in range(len(idx)):
            o = np.arange(len(idx)) != j
            out[idx[j]] = yy[j] - X[j] @ np.linalg.lstsq(X[o], yy[o], rcond=None)[0]
    return out


ROWP = {m: PRED_C[m].reshape(n_cat, NS)[groups, ROW_SLOT]
        for m in ['shared_effect', 'lgbm_C', 'lgbm_Cp', 'amplitude']}
E_ROW = {m: row_shape_err(p) for m, p in ROWP.items()}
E_ROW['per_cell_linear_LOO_design_axes'] = row_loo(
    np.column_stack([np.ones(N), SL_TOT[ROW_SLOT], SL_AR[ROW_SLOT], SL_R[ROW_SLOT]]), np.where(NROW_CELL >= 8)[0])
E_ROW['per_cell_linear_LOO_raw_flows'] = row_loo(
    np.column_stack([np.ones(N)] + [df[c].values.astype(float) for c in COND_COLS]), np.where(NROW_CELL == 27)[0])
AUDIT_CONV = {'what': ('ADDED CHECK, audit convention, not the metric: rows instead of slot means (the 1,906 '
                       'duplicated slots enter twice and "other rows" include the other run), cells by row count. '
                       'per_cell_linear_LOO_raw_flows = per-cell OLS on intercept + raw Ar, CH4, O2 flows, leave '
                       'one row out; complete 27-row cells only (ill-conditioned in small cells).'),
              'seed_averaging': 'same seed-averaged predictions as the metric'}
for lab, rm in (('rows_cells_ge_8_rows_all_917', NROW_CELL[CELL_ROW] >= 8),
                ('rows_cells_with_27_rows_all_917', NROW_CELL[CELL_ROW] == 27)):
    AUDIT_CONV[lab] = {'n_rows': int(rm.sum()),
                       **{m: float(np.sqrt(np.mean(E[rm] ** 2))) for m, E in E_ROW.items()
                          if np.isfinite(E[rm]).all()}}

RES['c_composition_x_condition'] = {
    'metric': ('e_r = (y_r - mean y over the cell\'s other measured slots) - (p_r - mean p over the same slots); '
               'slot yields; cells with >= 8 measured slots; within-shape RMSE = sqrt(mean e_r^2); '
               'seed-averaged predictions'),
    'n_eligible_cells_all_917': int(ELIG.sum()), 'n_complete_27_slot_cells_all_917': int(COMPLETE.sum()),
    'shared_effect_slots_without_training_data': int(n_eff_empty),
    'amplitude_model': {'training_cells_per_fold_mean': float(np.mean(amp_train_cells)),
                        'per_fold_pearson_a_hat_vs_a_heldout_eligible_cells': amp_r,
                        'per_fold_pearson_mean': float(np.mean(amp_r))},
    'per_cell_linear_LOO': {'n_fits': int(n_loo), 'n_rank_deficient_fits': int(n_rank_def),
                            'note': 'REFERENCE BOUND: uses the held-out catalyst\'s own cell data'},
    'results': SHAPE, 'rule_c': RULE_C, 'in_sample_row_level_framing': INSAMPLE,
    'audit_convention_check_rows': AUDIT_CONV}
log(f"(c) rule: {RULE_C['verdict']}; {RULE_C['numbers']}")

# =============================================================================== (d) argmax table
log("(d) argmax table")
R = pd.DataFrame({'c': groups, 'y': y, 'Temperature_C': df.Temperature_C.values.astype(float),
                  'ch4_o2': ax.ch4_o2.values, 'total_flow': ax.total_flow.values,
                  'ar_fraction': ax.ar_fraction.values})
LEVELS = {'ch4_o2': [2.0, 4.0, 6.0], 'total_flow': [20.0, 15.0, 10.0], 'ar_fraction': [0.70, 0.40, 0.15],
          'Temperature_C': [float(t) for t in TEMPS]}
imax = R.groupby('c').y.idxmax().reindex(range(n_cat)).values
assert (R.loc[imax, 'y'].values == TRUE_MAX).all()
n_tied_max_rows = int((R.y.values == TRUE_MAX[groups]).sum() - n_cat)
BEST_T = R.loc[imax, 'Temperature_C'].values
cellmax = R.groupby(['c', 'Temperature_C']).y.max().unstack().reindex(index=range(n_cat), columns=LEVELS['Temperature_C'])
assert ((cellmax.values == cellmax.max(1).values[:, None]).sum(1) == 1).all()      # unique best cell
assert (np.array(LEVELS['Temperature_C'])[np.nanargmax(cellmax.values, 1)] == BEST_T).all()


def pref_weights(V):
    """V: (n_cat, n_levels) statistic per level (NaN = level absent). Equal split among tied maxima."""
    mx = np.nanmax(V, 1, keepdims=True)
    t = (V == mx)
    return t / t.sum(1, keepdims=True), int((t.sum(1) > 1).sum())


W, NTIE = {}, {}
for a, lv in LEVELS.items():
    W[('S1', a)] = (R.loc[imax, a].values[:, None] == np.array(lv)[None]).astype(float)
    NTIE[('S1', a)] = 0
    top3 = (R.sort_values('y', ascending=False, kind='stable').groupby(['c', a]).head(3)
            .groupby(['c', a]).y.mean().unstack(a).reindex(index=range(n_cat), columns=lv).values)
    W[('S2', a)], NTIE[('S2', a)] = pref_weights(top3)
    if a != 'Temperature_C':
        inb = R[R.Temperature_C.values == BEST_T[R.c.values]]
        s3 = inb.groupby(['c', a]).y.mean().unstack(a).reindex(index=range(n_cat), columns=lv).values
        W[('S3', a)], NTIE[('S3', a)] = pref_weights(s3)
for w in W.values():
    assert np.allclose(w.sum(1), 1)

STATS = {'S1': 'single highest row', 'S2': 'mean of the 3 highest rows per level (fewer if the level has < 3)',
         'S3': 'per-level mean within the best-temperature cell (cell with the highest max)'}
TOPK = int(round(0.10 * n_cat))
TOP = np.argsort(-TRUE_MAX, kind='stable')[:TOPK]
ARG = {'statistics': STATS, 'ties': 'split equally among tied levels; n_tied = catalysts with a tie',
       'n_tied_max_rows_all_917': n_tied_max_rows, 'shares': {}, 'ba_ar_0_70': {}, 'top_decile_ratio_2': {}}
for (st, a), w in W.items():
    lv = LEVELS[a]
    ent = {'n_tied': NTIE[(st, a)],
           'overall_all_917': {f'{v:g}': float(w[:, j].mean()) for j, v in enumerate(lv)},
           'overall_counts_all_917': {f'{v:g}': float(w[:, j].sum()) for j, v in enumerate(lv)},
           'by_support_family': {f: {'n': int((SUP == f).sum()),
                                     **{f'{v:g}': float(w[SUP == f, j].mean()) for j, v in enumerate(lv)}}
                                 for f in FAMS}}
    ARG['shares'].setdefault(a, {})[st] = ent
for st in ('S1', 'S2', 'S3'):
    w = W[(st, 'ar_fraction')][:, 0]                      # preference for Ar fraction 0.70
    blk = {}
    for dn, B in (('Ba_any_token_promoter_or_BaO_support', BA_ANY), ('Ba_promoter_only', BA_PROMOTER)):
        e = {'overall': {'Ba': float(w[B].mean()), 'non_Ba': float(w[~B].mean()),
                         'n_Ba': int(B.sum()), 'n_non_Ba': int((~B).sum())}}
        for t in (800.0, 850.0):
            sm = BEST_T == t
            e[f'best_T_{int(t)}'] = {'Ba': float(w[sm & B].mean()), 'non_Ba': float(w[sm & ~B].mean()),
                                     'n_Ba': int((sm & B).sum()), 'n_non_Ba': int((sm & ~B).sum())}
        blk[dn] = e
    ARG['ba_ar_0_70'][st] = blk
    ARG['top_decile_ratio_2'][st] = {'K': TOPK, 'count_at_ratio_2': float(W[(st, 'ch4_o2')][TOP, 0].sum())}
ARG['best_T_definition'] = 'temperature of the S1 row = the cell with the highest max (unique for every catalyst)'


def cell_best_table(a):
    t = (R.groupby(['c', 'Temperature_C', a]).y.max().groupby(['Temperature_C', a]).agg(['mean', 'size']))
    return [{'Temperature_C': int(T_), a: float(v), 'mean_cell_best_yield': float(r['mean']),
             'n_catalysts': int(r['size'])} for (T_, v), r in t.iterrows()]


ARG['mean_cell_best_yield_by_T_x_ratio'] = {
    'what': 'per (catalyst, temperature, CH4:O2): max row yield; mean over catalysts that have it',
    'rows': cell_best_table('ch4_o2')}
ARG['mean_cell_best_yield_by_T_x_ar_fraction'] = {
    'what': 'per (catalyst, temperature, Ar fraction): max row yield; mean over catalysts that have it',
    'rows': cell_best_table('ar_fraction')}
RES['d_argmax_table'] = ARG
_s1 = ARG['shares']
log("(d) S1 shares: ratio " + '/'.join(f"{v:.3f}" for v in _s1['ch4_o2']['S1']['overall_all_917'].values())
    + "; total flow 20/15/10 " + '/'.join(f"{v:.3f}" for v in _s1['total_flow']['S1']['overall_all_917'].values()))

# =============================================================================== reference check
A_ = RES['a_condition_recommendation']
ALLP = lambda sc, k, st: A_['regret'][sc][f'k{k}']['all_917'][st]
EDGE = lambda sc, k: A_['comparisons_catalyst_bootstrap'][sc][f'k{k}']['all_917']['family_rule_minus_model_C_per_seed']
SH = RES['c_composition_x_condition']['results']
PR = RES['b_reduced_protocol']['designed']['greedy']
RD = RES['b_reduced_protocol']['random']
CM_ = RES['b_reduced_protocol']['composition_model_on_K_slot_labels']
REFN = [
    ('a optimistic k1 model C mean regret', 2.318, ALLP('optimistic', 1, 'model_C_per_seed')['mean_regret_seed_averaged'],
     'audit mostly 1-2 seeds; here 5-seed average, all_917'),
    ('a optimistic k1 global rule mean regret', 3.012, ALLP('optimistic', 1, 'global_rule')['mean_regret_seed_averaged'], ''),
    ('a optimistic k3 model C mean regret', 1.423, ALLP('optimistic', 3, 'model_C_per_seed')['mean_regret_seed_averaged'], ''),
    ('a optimistic k3 global rule mean regret', 1.743, ALLP('optimistic', 3, 'global_rule')['mean_regret_seed_averaged'], ''),
    ('a optimistic k1 model edge over family (audit 0.09-0.18)', 0.135, EDGE('optimistic', 1)['mean_diff'],
     'audit gave a range over runs; midpoint shown'),
    ('a optimistic k3 model edge over family (audit 0.17-0.18)', 0.175, EDGE('optimistic', 3)['mean_diff'], 'midpoint'),
    ('a family share of model gain over global, optimistic k1', 0.85,
     A_['family_rule_share_of_model_C_gain_over_global']['optimistic_k1'], 'audit "~85%"; k not stated'),
    ('a family share of model gain over global, optimistic k3', 0.85,
     A_['family_rule_share_of_model_C_gain_over_global']['optimistic_k3'], ''),
    ('b greedy K3 zero-imputed all Spearman', 0.921, PR['K3']['all_917']['seed_mean_of_pooled_oof']['zero_imputed_all_spearman'],
     'audit: split-half, failures as 0; here 5-fold OOF, sensitivity scoring; this greedy drops uncovered '
     'catalysts from its objective (spec), so its sets cover fewer catalysts and lose more under zero-imputation'),
    ('b random K3 zero-imputed all Spearman (mean)', 0.818, RD['K3']['all_917']['zero_imputed_all_spearman']['seed_mean_draws_mean'], ''),
    ('b greedy K10 zero-imputed all Spearman', 0.969, PR['K10']['all_917']['seed_mean_of_pooled_oof']['zero_imputed_all_spearman'], ''),
    ('b random K10 zero-imputed all Spearman (mean)', 0.922, RD['K10']['all_917']['zero_imputed_all_spearman']['seed_mean_draws_mean'], ''),
    ('b greedy K20 zero-imputed all Spearman', 0.986, PR['K20']['all_917']['seed_mean_of_pooled_oof']['zero_imputed_all_spearman'], ''),
    ('b random K20 zero-imputed all Spearman (mean)', 0.949, RD['K20']['all_917']['zero_imputed_all_spearman']['seed_mean_draws_mean'], ''),
    ('b greedy K10 headline (missing) Spearman (audit 0.91-0.94)', 0.925,
     PR['K10']['all_917']['seed_mean_of_pooled_oof']['headline_missing_spearman'], 'midpoint'),
    ('b comp model greedy K10 label minus B_median, Spearman (audit -0.003 to -0.005)', -0.004,
     CM_['greedy_K10_label']['all_917']['minus_B_median_spearman_seed_averaged']['delta_point'],
     'audit label/baseline not stated; vs B_max also stored'),
    ('c complete cells shared effect [metric: slot means, 27-slot cells]', 1.481,
     SH['complete_27_slot_cells_all_917']['within_shape_rmse_seed_averaged']['shared_effect'], 'metric convention'),
    ('c complete cells per-cell slopes [metric: design axes]', 1.155,
     SH['complete_27_slot_cells_all_917']['within_shape_rmse_seed_averaged']['per_cell_linear_LOO_reference_bound'], 'metric convention'),
    ('c complete cells LGBM C [metric]', 1.249, SH['complete_27_slot_cells_all_917']['within_shape_rmse_seed_averaged']['lgbm_C'], 'metric convention'),
    ('c cells >= 8 shared effect [metric: slot means, >= 8 slots]', 1.620, SH['all_917']['within_shape_rmse_seed_averaged']['shared_effect'], 'metric convention'),
    ('c cells >= 8 per-cell slopes [metric: design axes]', 1.426, SH['all_917']['within_shape_rmse_seed_averaged']['per_cell_linear_LOO_reference_bound'], 'metric convention'),
    ('c cells >= 8 LGBM C [metric]', 1.481, SH['all_917']['within_shape_rmse_seed_averaged']['lgbm_C'], 'metric convention'),
    ('c LGBM share of shared-to-slopes gap closed [metric, cells >= 8]', 0.70, SH['all_917']['lgbm_C_share_of_shared_to_reference_gap_closed'], 'metric convention'),
    ('c 27-row cells shared effect [audit convention: rows]', 1.481, AUDIT_CONV['rows_cells_with_27_rows_all_917']['shared_effect'], ''),
    ('c 27-row cells per-cell slopes, raw flows [audit convention]', 1.155,
     AUDIT_CONV['rows_cells_with_27_rows_all_917']['per_cell_linear_LOO_raw_flows'], 'audit slopes were on raw flows'),
    ('c 27-row cells LGBM C [audit convention]', 1.249, AUDIT_CONV['rows_cells_with_27_rows_all_917']['lgbm_C'],
     'here 5-seed-averaged predictions; audit mostly 1-2 seeds'),
    ('c >= 8-row cells shared effect [audit convention]', 1.620, AUDIT_CONV['rows_cells_ge_8_rows_all_917']['shared_effect'], ''),
    ('c >= 8-row cells per-cell slopes, design axes [audit convention]', 1.426,
     AUDIT_CONV['rows_cells_ge_8_rows_all_917']['per_cell_linear_LOO_design_axes'],
     'audit basis for small cells unknown; raw flows are ill-conditioned there'),
    ('c >= 8-row cells LGBM C [audit convention]', 1.481, AUDIT_CONV['rows_cells_ge_8_rows_all_917']['lgbm_C'], ''),
    ('c in-sample within-cell pct of variance', 19.9, INSAMPLE['within_cell_pct_of_total_variance'], ''),
    ('c in-sample pct points explained by shared effect', 4.6, INSAMPLE['pct_points_explained_by_shared_effect'], ''),
    ('d S1 ratio 2 share', 0.845, _s1['ch4_o2']['S1']['overall_all_917']['2'], ''),
    ('d S1 ratio 4 share', 0.083, _s1['ch4_o2']['S1']['overall_all_917']['4'], ''),
    ('d S1 ratio 6 share', 0.072, _s1['ch4_o2']['S1']['overall_all_917']['6'], ''),
    ('d S1 total flow 20 share', 0.636, _s1['total_flow']['S1']['overall_all_917']['20'], ''),
    ('d S1 total flow 15 share', 0.226, _s1['total_flow']['S1']['overall_all_917']['15'], ''),
    ('d S1 total flow 10 share', 0.139, _s1['total_flow']['S1']['overall_all_917']['10'], ''),
    ('d S1 BaO support Ar 0.70 share', 0.81, _s1['ar_fraction']['S1']['by_support_family']['BaO']['0.7'], ''),
    ('d S1 La2O3 support Ar 0.70 share', 0.22, _s1['ar_fraction']['S1']['by_support_family']['La2O3']['0.7'], ''),
    ('d S1 CeO2 support Ar 0.70 share', 0.26, _s1['ar_fraction']['S1']['by_support_family']['CeO2']['0.7'], ''),
    ('d S1 CaO support Ar 0.70 share', 0.28, _s1['ar_fraction']['S1']['by_support_family']['CaO']['0.7'], ''),
    ('d S1 best-T 850 Ba Ar 0.70 share', 0.852, ARG['ba_ar_0_70']['S1']['Ba_any_token_promoter_or_BaO_support']['best_T_850']['Ba'], ''),
    ('d S1 best-T 850 non-Ba Ar 0.70 share', 0.503, ARG['ba_ar_0_70']['S1']['Ba_any_token_promoter_or_BaO_support']['best_T_850']['non_Ba'], ''),
    ('d S1 top-decile at ratio 2 (of 92)', 88, ARG['top_decile_ratio_2']['S1']['count_at_ratio_2'], ''),
] + [(f'd S1 temperature argmax count {t}', v, _s1['Temperature_C']['S1']['overall_counts_all_917'][str(t)], '')
     for t, v in zip(TEMPS, [102, 170, 238, 311, 96])]
RES['reference_check'] = {
    'what': ('this script vs approximate audit values (uncommitted, mostly 1-2 seeds, conventions may differ); '
             'for checking only'),
    'rows': [{'quantity': q, 'audit': a, 'this_script': (None if v is None else float(v)), **({'note': n} if n else {})}
             for q, a, v, n in REFN]}

RES['about']['runtime_minutes'] = (time.time() - t0) / 60
if not DEV:
    json.dump(RES, open(OUT_JSON, 'w'), indent=1, default=_json_default)
    log(f"wrote {OUT_JSON}")
else:
    log("dev run: nothing written")

# =============================================================================== printed summary
L = 'seed-averaged per-catalyst regret' if NSEED > 1 else 'regret (1 seed)'
print("\n=========== phase13: where the gas-flow conditions add deployable value ===========")
print(f"Seeds {SEEDS}; bootstrap {N_BOOT} catalyst resamples, seed {BOOT_SEED}. Truth (a, b) = max over measured "
      f"slot yields; differs from the observed row max for {TRUTH_DIFF.sum()} doubled catalysts.")
print(f"\n(a) Recommendation regret, all_917 (mean of {L}; mean truth {MEAN_TRUTH['all_917']:.3f}).")
print(f"    {'strategy':24s} " + ' '.join(f"{'head k' + str(k):>9s}" for k in KS_REC) + ' | '
      + ' '.join(f"{'opt k' + str(k):>8s}" for k in KS_REC) + ' | ' + ' '.join(f"{'unm k' + str(k):>7s}" for k in KS_REC))
for st in STRATS:
    print(f"    {st:24s} " + ' '.join(f"{ALLP('headline', k, st)['mean_regret_seed_averaged']:9.3f}" for k in KS_REC)
          + ' | ' + ' '.join(f"{ALLP('optimistic', k, st)['mean_regret_seed_averaged']:8.3f}" for k in KS_REC)
          + ' | ' + ' '.join(f"{UNM_A[f'k{k}']['all_917'][st]:7.3f}" for k in KS_REC))
print("    head = headline (unmeasured pick = failed run, 0); opt = optimistic (skip unmeasured); "
      "unm = share of top-k picks unmeasured. model_C_seed_averaged is an added check.")
print("    Regret differences, all_917, mean of seed-averaged per-catalyst differences [95% catalyst-bootstrap CI]; > 0 = second strategy better:")
for sc in SCORINGS:
    for k in KS_REC:
        parts = []
        for a, b in PAIRS_A:
            r = A_['comparisons_catalyst_bootstrap'][sc][f'k{k}']['all_917'][f'{a}_minus_{b}']
            parts.append(f"{a.split('_')[0]}-{'C' if b == 'model_C_per_seed' else 'Cp'} {r['mean_diff']:+.3f} "
                         f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]")
        print(f"    {sc:10s} k={k}: " + '; '.join(parts))
for r_ in [RULE_A['primary_k3_headline_all_917']] + RULE_A['secondary_same_evaluation']:
    tag = 'RULE (a)' if r_ is RULE_A['primary_k3_headline_all_917'] else 'secondary'
    print(f"    {tag:9s} {r_['scoring']:10s} k={r_['k']}: family - model_C_per_seed {r_['mean_edge']:+.3f} "
          f"[{r_['ci95'][0]:+.3f}, {r_['ci95'][1]:+.3f}], per-seed edge > 0 in {r_['n_seeds_edge_positive']}/"
          f"{r_['n_seeds']} seeds -> {r_['verdict']}")

print("\n(b) Designed reduced protocol, all_917, seed mean of pooled OOF. CAVEAT: optimistic about prospective use.")
print(f"    {'chooser':16s} {'K':>3s} {'cover':>6s} {'rho miss':>9s} {'prec miss':>9s} {'rho zero-all':>12s} | "
      f"{'rand rho mean':>13s} {'rand p97.5':>10s}  verdict")
for ch in CHOOSERS:
    for K in KS_PROT:
        p = PROT[ch][f'K{K}']['all_917']['seed_mean_of_pooled_oof']
        rr = RAND[f'K{K}']['all_917']['headline_missing_spearman']
        print(f"    {ch:16s} {K:3d} {p['coverage']:6.3f} {p['headline_missing_spearman']:9.4f} "
              f"{p['headline_missing_precision_top10pct']:9.3f} {p['zero_imputed_all_spearman']:12.4f} | "
              f"{rr['seed_mean_draws_mean']:13.4f} {rr['seed_mean_draws_p97_5']:10.4f}  "
              f"{RULE_B['results'][f'{ch}_K{K}']['verdict']}")
for K in KS_PROT:
    rr = RAND[f'K{K}']['all_917']
    print(f"    random K={K:2d}: coverage {rr['coverage_mean']:.3f}; zero-imputed-all rho mean "
          f"{rr['zero_imputed_all_spearman']['seed_mean_draws_mean']:.4f} (sensitivity, cites phase12 R4)")
print("    rho miss = headline (missing), covered catalysts only; rho zero-all = sensitivity check, uncovered = 0.")
print("    Training objective (greedy's own objective, training catalysts, mean over seed-fold pairs) greedy vs "
      "most_often_max: " + '; '.join(f"K={K} {TRAIN_OBJ[f'K{K}']['greedy']:.4f} vs {TRAIN_OBJ[f'K{K}']['most_often_max']:.4f}"
                                     for K in KS_PROT))
print("    Composition model (65 cols) on greedy K-slot labels vs observed row max; seed-averaged predictions; "
      "delta vs B_median [CI]:")
for vn in list(variants):
    for pn in POPS:
        e = CM_[vn][pn]
        b = e['minus_B_median_spearman_seed_averaged']; bm = e['minus_B_max_spearman_seed_averaged']
        print(f"      {vn:34s} {pn:8s} rho {e['seed_averaged_predictions']['spearman']:.4f} | vs B_median "
              f"{b['delta_point']:+.4f} [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] | vs B_max "
              f"{bm['delta_point']:+.4f} [{bm['ci95_low']:+.4f}, {bm['ci95_high']:+.4f}]")
if not DEV:
    pc = CM_['pipeline_check_B_retrained_here_vs_npz']
    print("    Pipeline check, B retrained here vs npz (seed-averaged): " + ', '.join(
        f"{k} max |diff| {v['max_abs_diff_seed_averaged_predictions']:.2e}" for k, v in pc.items()))

print("\n(c) Within-shape RMSE, seed-averaged predictions (cells >= 8 measured slots; complete = 27 slots):")
print(f"    {'model':38s} " + ' '.join(f"{pn[:14]:>14s}" for pn in SHAPE_POPS))
for m in MODELS_C:
    print(f"    {m:38s} " + ' '.join(f"{SHAPE[pn]['within_shape_rmse_seed_averaged'][m]:14.4f}" for pn in SHAPE_POPS))
print("    vs lgbm_C, all_917, RMSE(model) - RMSE(lgbm_C) [95% catalyst-bootstrap CI]:")
for m in MODELS_C:
    if m == 'lgbm_C':
        continue
    b = SHAPE['all_917']['vs_lgbm_C_catalyst_bootstrap'][f'{m}_minus_lgbm_C']
    print(f"      {m:38s} {b['rmse_diff_point']:+.4f} [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] {b['reading']}")
print(f"    RULE (c): {RULE_C['verdict']}")
print(f"    In-sample row-level framing (descriptive): within-cell {INSAMPLE['within_cell_pct_of_total_variance']:.1f}% "
      f"of variance; shared effect explains {INSAMPLE['pct_points_explained_by_shared_effect']:.1f} points.")
for lab in ('rows_cells_ge_8_rows_all_917', 'rows_cells_with_27_rows_all_917'):
    print(f"    Audit-convention check ({lab}, rows not slot means; not the metric): " + ', '.join(
        f"{k} {v:.3f}" for k, v in AUDIT_CONV[lab].items() if k != 'n_rows'))

print("\n(d) Argmax shares, all_917 (S1 single highest row / S2 top-3 mean per level / S3 per-level mean at best T):")
for a in ['ch4_o2', 'total_flow', 'ar_fraction', 'Temperature_C']:
    for st in ('S1', 'S2', 'S3'):
        if st in ARG['shares'][a]:
            e = ARG['shares'][a][st]
            print(f"    {a:13s} {st}: " + ', '.join(f"{k} {v:.3f}" for k, v in e['overall_all_917'].items())
                  + f" (tied {e['n_tied']})")
for st in ('S1', 'S2', 'S3'):
    e = ARG['ba_ar_0_70'][st]['Ba_any_token_promoter_or_BaO_support']
    print(f"    {st} share preferring Ar 0.70, Ba (promoter or BaO) vs non-Ba: overall {e['overall']['Ba']:.3f} vs "
          f"{e['overall']['non_Ba']:.3f}; best T 800 {e['best_T_800']['Ba']:.3f} vs {e['best_T_800']['non_Ba']:.3f}; "
          f"best T 850 {e['best_T_850']['Ba']:.3f} vs {e['best_T_850']['non_Ba']:.3f}; top decile at ratio 2 "
          f"{ARG['top_decile_ratio_2'][st]['count_at_ratio_2']:.1f}/{TOPK}")

print("\nReference check (audit -> this script; audit values approximate, conventions may differ):")
for q, a, v, n in REFN:
    print(f"    {q:72s} {a:>8} -> {('-' if v is None else f'{v:.4g}'):>8s} {n}")
log("done")
