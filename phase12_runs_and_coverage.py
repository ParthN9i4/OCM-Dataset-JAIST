"""
phase12_runs_and_coverage.py -- the data-structure facts the documents rely on. No models are trained.

Each fact below is computed from OCM_data_with_conditions.csv via load_with_conditions() and stored in
phase12_runs_and_coverage.json, so no document has to carry a hand-copied number.

No number here comes from a model, a seed, or a prediction. "Per-seed mean" and "seed-averaged" do not
apply. Each stored number names its population: all_917 (all catalysts), ND_889 (without the 28
doubled catalysts), doubled_28, or a named set of cells or slots.

TERMS. A cell is a (catalyst, temperature) pair: 917 x 5 = 4,585 possible, 4,399 observed. A slot is a
(catalyst, temperature, condition) triple, with 27 gas-flow conditions. A duplicated slot is a slot
measured twice. For a duplicated slot, hi and lo are the larger and smaller of its two yields. They are
per-pair order statistics. The file cannot say which run a row came from: it lists each catalyst's rows
sorted by yield, so row order carries no run identity (checked below).

SECTIONS.
  1. Design. The 27 conditions form an exact 3x3x3 grid; 5 temperatures; 135 slots.
  2. Two-run structure of the 28 doubled catalysts: counts, hi/lo ratios, a smoothness test and a
     shared-pattern test.
  3. Top-decile inflation: are the 28 over-represented among the top 10% by observed max?
  4. Mass-balance exclusion test: are excluded slots ones that would have scored low?
  5. Precision: within-cell scatter in complete cells.
  6. Floors for grouping by (catalyst, temperature).
  7. Two numbers quoted in the sent work note that no other JSON stores.

RULES (stated here, before the results).

  R2 (section 2, shared condition pattern). Cells with >= 10 duplicated slots. Per cell, rho =
  Spearman(hi, lo) across its duplicated conditions. Null A permutes lo across conditions within the
  cell, hi fixed, 200 permutations. The two runs "share the condition pattern" only if BOTH hold:
    (a) the observed across-cell median rho exceeds the 95th percentile of the across-cell median
        under null A (each null draw permutes every cell once), and
    (b) the share of cells whose rho exceeds their own null-A 95th percentile is above the 5% expected
        by chance, one-sided binomial p < 0.05.
  Otherwise: "no detectable shared condition pattern". The two-part form is this script's reading of
  the spec's "observed median clearly above null A".
  Null B pools a cell's values, pairs them at random and splits each pair into max and min. It is
  reported but does not enter R2: it tests a different hypothesis (see null_B_note in the JSON).
  Null C (added check, not part of R2) is null A with each permuted pair re-sorted into max and min.

  R4 (section 4, exclusion). Exclusion is "outcome-coupled within catalyst" only if, in at least 2 of
  the 3 CH4:O2 ratios, the excluded-slot mean percentile is lower than the present-slot mean percentile
  by >= 0.05 AND a catalyst-level bootstrap 95% CI of that difference excludes zero (3000 resamples of
  the 917 catalysts, seed 7). Otherwise: "not outcome-coupled within catalyst by this rule".
  Scope: R4 covers only slots with a measured T-50 or T+50 neighbour. Slots without one are counted and
  described (rule_R4.scope), not scored.

METHOD NOTES.
  Section 2 smoothness. Cells with >= 15 duplicated slots. Within each cell, a 7-parameter categorical
  main-effects model (intercept + 2 total-flow + 2 Ar-fraction + 2 CH4:O2 dummies) is fitted
  separately to the hi members, the lo members, the pair means, and one random member per pair (200
  draws, seed 7). Residual SD = sqrt(RSS / (n - rank)). We report the median over cells, and the same
  for ordinary complete cells (27 distinct conditions, ND_889 catalysts only). For the random member,
  the primary number averages the SD over draws within a cell; the per-cell median over draws is also
  stored.
  Section 4 percentile. Duplicated slots are averaged first. Within-cell percentile rank =
  (r - 1) / (n - 1), r the average rank among the cell's n measured slots (0 = cell worst, 1 = cell
  best); undefined for n = 1. Top decile = percentile >= 0.9. Cell best = equal to the cell max. Each
  (catalyst, temperature, condition) slot in an observed cell is "present" if measured, else
  "excluded". Its score is the percentile of the same catalyst and condition at T-50 or T+50, where
  measured; the mean of the two if both exist. Slots with no such neighbour are dropped and counted.
  Absent cells (no row at that temperature) are not slots of this test; they are counted separately.
  Section 4 added checks (not part of R4): (i) percentile = r / n; (ii) the audit's variant, r / n with
  each (slot, measured neighbour) pair counted separately instead of averaging T-50 and T+50; (iii)
  neighbours restricted to complete 27-slot cells, because one slot's percentile depends on which other
  slots its cell holds; (iv) the nearest measured temperature at any distance, to score more slots.
  Section 5. "Within-session condition-to-condition scatter on complete cells; an upper bound on
  independent per-condition noise; NOT run-to-run repeatability."

Output:  phase12_runs_and_coverage.json
Usage:   python3 phase12_runs_and_coverage.py [--dev]
         --dev = 50 random draws, 300 bootstrap resamples, no file written.
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, hypergeom, binomtest
from ocm_eval import (load_with_conditions, TARGET, COND_COLS, design_axes, slot_keys,
                      condition_grid, doubled_catalyst_mask, equal_effort_mask, MIN_CELL)

DEV = '--dev' in sys.argv
N_DRAW = 50 if DEV else 200          # random-member draws and null A / B / C draws (section 2)
N_BOOT = 300 if DEV else 3000        # catalyst bootstrap resamples (section 4)
RNG_SEED = 7
OUT_JSON = 'phase12_runs_and_coverage.json'
LEVELS = {'total_flow': [10.0, 15.0, 20.0], 'ar_fraction': [0.70, 0.40, 0.15], 'ch4_o2': [2.0, 4.0, 6.0]}
TEMPS = [700, 750, 800, 850, 900]
AX = ['total_flow', 'ar_fraction', 'ch4_o2']

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ------------------------------------------------------------------------------- data
df, feat_all, groups, n_cat = load_with_conditions()
y = df[TARGET].values.astype(float)
N = len(y)
T = df.Temperature_C.values.astype(int)
ax = design_axes(df)
GRID = condition_grid(df)
key2slot = {k: int(s) for k, s in zip(zip(GRID.Temperature_C.astype(float), GRID.total_flow,
                                          GRID.ar_fraction, GRID.ch4_o2), GRID.slot)}
ROW_SLOT = np.array([key2slot[k] for k in slot_keys(df)])
TI, COND = ROW_SLOT // 27, ROW_SLOT % 27                # temperature index 0..4, condition 0..26
DBL = doubled_catalyst_mask(df, groups, n_cat)
NAMES = df.groupby(groups).Name.first().reindex(range(n_cat)).values
TRUE_MAX = pd.Series(y).groupby(groups).max().reindex(range(n_cat)).values
assert N == 89074 and n_cat == 917 and DBL.sum() == 28
log(f"{N:,} rows, {n_cat} catalysts, {DBL.sum()} doubled")

RES = {'about': {
    'script': 'phase12_runs_and_coverage.py',
    'what': 'Data-structure facts the documents rely on. No models are trained.',
    'seed_convention': 'Not applicable: no model, seed or prediction enters any number here.',
    'populations': {'all_917': 'all catalysts', 'ND_889': 'without the 28 doubled catalysts',
                    'doubled_28': 'catalysts with at least one slot measured twice',
                    'EE_771': f'equal effort: >= {MIN_CELL} rows in some (catalyst, temperature) cell'},
    'rows_all_917': N, 'n_catalysts_all_917': n_cat,
    'n_catalysts_EE_771': int(equal_effort_mask(groups, T, n_cat).sum()),
    'n_catalysts_ND_889': int((~DBL).sum()),
    'dev_mode': DEV, 'n_random_draws_section2': N_DRAW, 'n_boot_section4': N_BOOT,
    'rng_seed': RNG_SEED}}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ===================================================================== 1. design
g27 = GRID[GRID.Temperature_C == 700][AX].reset_index(drop=True)
combos = set(map(tuple, g27.values))
want = {(a, b, c) for a in LEVELS['total_flow'] for b in LEVELS['ar_fraction'] for c in LEVELS['ch4_o2']}
assert len(GRID) == 135 and sorted(GRID.Temperature_C.unique()) == TEMPS
assert len(g27) == 27 and combos == want                      # each combination exactly once
for t in TEMPS:                                               # same 27, same order, at every T
    assert (GRID[GRID.Temperature_C == t][AX].values == g27.values).all()
assert (GRID.slot.values // 27 == np.repeat(np.arange(5), 27)).all()
raw = df[COND_COLS].values.astype(float); tot = raw.sum(1)
dev_axes = {'total_flow': float(np.abs(tot - ax.total_flow.values).max()),
            'ar_fraction': float(np.abs(raw[:, 0] / tot - ax.ar_fraction.values).max()),
            'ch4_o2': float(np.abs(raw[:, 1] / raw[:, 2] - ax.ch4_o2.values).max())}
assert max(dev_axes.values()) < 1e-6
RES['design'] = {
    'grid': '3x3x3: total flow 10/15/20 (file units) x Ar fraction 0.70/0.40/0.15 x CH4:O2 2/4/6',
    'n_conditions': 27, 'each_combination_exactly_once': True, 'temperatures_C': TEMPS,
    'n_slots_per_catalyst': 135,
    'max_abs_deviation_raw_flows_vs_design_axes_all_rows': dev_axes}
log("1 design: exact 3x3x3 x 5 temperatures = 135 slots (asserted)")

# ===================================================================== 2. two-run structure
key = groups * 135 + ROW_SLOT
cnt = pd.Series(key).value_counts()
assert cnt.max() == 2
dkeys = np.sort(cnt[cnt == 2].index.values)
assert len(dkeys) == 1906 and DBL[np.unique(dkeys // 135)].all()
sl = pd.DataFrame({'key': key, 'y': y}).groupby('key').y.agg(['min', 'max', 'mean', 'size'])
P = sl.loc[dkeys]
P = P.assign(c=dkeys // 135, ti=(dkeys % 135) // 27, cond=dkeys % 27, hi=P['max'], lo=P['min'])
P['T'] = np.array(TEMPS)[P.ti.values]
P['ratio'] = np.where(P.lo > 0, P.hi / P.lo.where(P.lo > 0), np.nan)
n_lo0 = int((P.lo == 0).sum())
cells_dup = P.groupby(['c', 'T']).size()
temps_per_cat = P.groupby('c').T.nunique()

# file order: are each catalyst's rows listed in non-increasing yield order?
file_sorted = pd.Series(y).groupby(groups).apply(lambda s: bool((np.diff(s.values) <= 0).all()))

cat_rows = []
for c in np.where(DBL)[0]:
    pc = P[P.c == c]
    cat_rows.append({'name': NAMES[c], 'catalyst_code': int(c), 'n_rows': int((groups == c).sum()),
                     'n_duplicated_slots': int(len(pc)), 'temperatures_with_duplicated_slots':
                     sorted(int(t) for t in pc['T'].unique()),
                     'observed_max': float(TRUE_MAX[c]),
                     'median_hi_over_lo_pairs_lo_gt_0': float(pc.ratio.median())})
cat_rows.sort(key=lambda r: r['name'])

Pv = P[P.lo > 0]
ratio_pooled = {'overall': float(Pv.ratio.median()),
                **{str(t): float(Pv[Pv['T'] == t].ratio.median()) for t in TEMPS}}
cell_med = Pv.groupby(['c', 'T']).ratio.median()
ratio_cellmed = {'overall': float(cell_med.median()),
                 **{str(t): float(cell_med.xs(t, level='T').median()) for t in TEMPS}}
ratio_q = Pv.ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

# ---- design matrices for categorical models on the 27 conditions
LV = {a: np.array([LEVELS[a].index(v) for v in g27[a].values]) for a in AX}   # level index 0..2
D_main = [np.ones(27)] + [(LV[a] == k).astype(float) for a in AX for k in (1, 2)]
D7 = np.column_stack(D_main)                                                   # 7 columns
D_int = [(LV[a] == i).astype(float) * (LV[b] == j)
         for ai, a in enumerate(AX) for b in AX[ai + 1:] for i in (1, 2) for j in (1, 2)]
D19 = np.column_stack(D_main + D_int)                                          # 19 columns
assert np.linalg.matrix_rank(D7) == 7 and np.linalg.matrix_rank(D19) == 19


def resid_sd(X, Y):
    """df-corrected residual SD of OLS fits of each column of Y on X."""
    beta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    rss = ((Y - X @ beta) ** 2).sum(0)
    return np.sqrt(rss / (X.shape[0] - rank)), X.shape[0] - rank


# ---- ordinary complete cells: ND_889 catalysts, 27 distinct conditions (27 rows)
cell_id = groups * 5 + TI
cd = pd.DataFrame({'cell': cell_id, 'cond': COND, 'y': y, 'dbl': DBL[groups]})
cell_n = cd.groupby('cell').cond.nunique()
comp_cells = np.sort(cell_n[(cell_n == 27)].index.values)
comp_cells = comp_cells[~DBL[comp_cells // 5]]
cc = cd[cd.cell.isin(comp_cells)].sort_values(['cell', 'cond'])
assert (cc.groupby('cell').size() == 27).all()
Ycc = cc.y.values.reshape(len(comp_cells), 27).T                              # 27 x n_cells
assert (cc.cond.values.reshape(len(comp_cells), 27) == np.arange(27)).all()
sd7_cc, df7_cc = resid_sd(D7, Ycc)
cc_mean = Ycc.mean(0)

# ---- smoothness test, cells with >= 15 duplicated slots
rng = np.random.default_rng(RNG_SEED)
smooth_cells = cells_dup[cells_dup >= 15].index
sm = {'hi': [], 'lo': [], 'pair_mean': [], 'random_member': []}
sm_rel = {k: [] for k in sm}
rnd_med = []                                             # per-cell median over draws (alternative)
for c, t in smooth_cells:
    pc = P[(P.c == c) & (P['T'] == t)].sort_values('cond')
    X = D7[pc.cond.values]
    hi, lo = pc.hi.values, pc.lo.values
    pick = rng.random((N_DRAW, len(pc))) < 0.5
    rnd = np.where(pick, hi[None], lo[None]).T                                   # n x N_DRAW
    for k, v in (('hi', hi), ('lo', lo), ('pair_mean', (hi + lo) / 2)):
        s = resid_sd(X, v[:, None])[0][0]
        sm[k].append(s); sm_rel[k].append(s / v.mean())
    s_r = resid_sd(X, rnd)[0]
    sm['random_member'].append(s_r.mean()); sm_rel['random_member'].append((s_r / rnd.mean(0)).mean())
    rnd_med.append(np.median(s_r))


def random_member_sd(seed):
    """Median over cells of the random-member residual SD, both aggregators, for one RNG seed."""
    r_ = np.random.default_rng(seed); a_, b_ = [], []
    for c, t in smooth_cells:
        pc = P[(P.c == c) & (P['T'] == t)].sort_values('cond')
        hi, lo = pc.hi.values, pc.lo.values
        s_r = resid_sd(D7[pc.cond.values], np.where(r_.random((N_DRAW, len(pc))) < 0.5, hi[None], lo[None]).T)[0]
        a_.append(s_r.mean()); b_.append(np.median(s_r))
    return float(np.median(a_)), float(np.median(b_))


assert np.isclose(random_member_sd(RNG_SEED)[0], np.median(sm['random_member']))
rnd_seeds = {s: random_member_sd(s) for s in range(6)}
smooth = {
    'population': 'cells of the doubled_28 with >= 15 duplicated slots',
    'n_cells': int(len(smooth_cells)),
    'model': '7-parameter categorical main effects (total flow, Ar fraction, CH4:O2), fitted per cell',
    'estimator': 'df-corrected residual SD per cell, median over cells; random_member = per-cell mean '
                 f'over {N_DRAW} draws (seed {RNG_SEED}), then median over cells',
    'median_resid_sd': {k: float(np.median(v)) for k, v in sm.items()},
    'random_member_alternative_aggregator': {
        'what': f'per-cell median over {N_DRAW} draws (seed {RNG_SEED}), then median over cells',
        'median_resid_sd': float(np.median(rnd_med)),
        'rng_seed_spread_seeds_0_to_5': {
            'per_cell_mean_aggregator_min_max': [min(v[0] for v in rnd_seeds.values()),
                                                 max(v[0] for v in rnd_seeds.values())],
            'per_cell_median_aggregator_min_max': [min(v[1] for v in rnd_seeds.values()),
                                                   max(v[1] for v in rnd_seeds.values())]},
        'note': ('Monte Carlo: the random-member value moves with the RNG seed and with the aggregator. '
                 'The audit value (~2.50) ' +
                 ('lies inside' if min(v[1] for v in rnd_seeds.values()) <= 2.50 <= max(v[1] for v in rnd_seeds.values())
                  else 'lies outside') + ' the seed range of the per-cell-median aggregator. The two '
                 f'aggregators give {np.median(sm["random_member"]) / np.median(sm["hi"]):.2f} and '
                 f'{np.median(rnd_med) / np.median(sm["hi"]):.2f} times the hi value.')},
    'median_relative_resid_sd_sd_over_series_mean': {k: float(np.median(v)) for k, v in sm_rel.items()},
    'ordinary_complete_cells': {
        'population': 'cells with 27 distinct conditions (27 rows), ND_889 catalysts only',
        'n_cells': int(len(comp_cells)), 'residual_df': int(df7_cc),
        'median_resid_sd': float(np.median(sd7_cc)),
        'median_relative_resid_sd': float(np.median(sd7_cc / cc_mean))},
}


# ---- shared-pattern test, cells with >= 10 duplicated slots
def rank_rows(a):
    return rankdata(a, axis=-1)


def corr_rows(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


rng = np.random.default_rng(RNG_SEED)
sp_cells = cells_dup[cells_dup >= 10].index
obs, nA, nB, nC, n_pairs = [], [], [], [], []
for c, t in sp_cells:
    pc = P[(P.c == c) & (P['T'] == t)]
    hi, lo = pc.hi.values, pc.lo.values; n = len(hi); n_pairs.append(n)
    obs.append(spearmanr(hi, lo)[0])
    perm = np.argsort(rng.random((N_DRAW, n)), axis=1)
    lo_p = lo[perm]
    nA.append(corr_rows(rank_rows(hi)[None], rank_rows(lo_p)))
    nC.append(corr_rows(rank_rows(np.maximum(hi[None], lo_p)), rank_rows(np.minimum(hi[None], lo_p))))
    pool = np.r_[hi, lo]
    q = pool[np.argsort(rng.random((N_DRAW, 2 * n)), axis=1)]
    a, b = q[:, :n], q[:, n:]
    nB.append(corr_rows(rank_rows(np.maximum(a, b)), rank_rows(np.minimum(a, b))))
obs = np.array(obs); nA = np.array(nA); nB = np.array(nB); nC = np.array(nC)    # cells x draws
assert np.isfinite(obs).all()
p95_cell = np.nanpercentile(nA, 95, axis=1)
above = int((obs > p95_cell).sum())
medA_draws = np.nanmedian(nA, axis=0)                     # across-cell median, per null-A draw
binom_p = float(binomtest(above, len(obs), 0.05, alternative='greater').pvalue)
r2_a = bool(np.median(obs) > np.percentile(medA_draws, 95))
r2_b = bool(above / len(obs) > 0.05 and binom_p < 0.05)
shares_pattern = r2_a and r2_b
shared = {
    'population': 'cells of the doubled_28 with >= 10 duplicated slots',
    'n_cells': int(len(sp_cells)),
    'statistic': 'Spearman(hi, lo) across the duplicated conditions of one cell',
    'observed_median_over_cells': float(np.median(obs)),
    'observed_quartiles_over_cells': [float(v) for v in np.percentile(obs, [25, 75])],
    'null_A': {'what': f'permute lo across conditions within the cell, hi fixed; {N_DRAW} draws',
               'median_over_cells_of_per_cell_null_mean': float(np.median(np.nanmean(nA, 1))),
               'across_cell_median_per_draw_p95': float(np.percentile(medA_draws, 95)),
               'across_cell_median_per_draw_max': float(medA_draws.max()),
               'n_cells_above_own_p95': above,
               'share_cells_above_own_p95': above / len(obs),
               'share_expected_by_chance': 0.05,
               'binomial_p_one_sided': binom_p},
    'null_B': {'what': f'pool the cell values, pair at random, split into max/min; {N_DRAW} draws',
               'median_over_cells_of_per_cell_null_mean': float(np.median(np.nanmean(nB, 1))),
               'population_value_continuous_iid': 7 / 15,
               'finite_sample_expectation_median_over_cells':
                   float(np.median([((n - 2) * 7 / 15 + 1) / (n + 1) for n in n_pairs])),
               'finite_sample_note': ('E[sample Spearman] = ((n-2) rho_S + 3 tau) / (n+1) with rho_S = '
                                      '7/15 and Kendall tau = 1/3 for (min, max) of iid pairs; n = pairs '
                                      'in the cell. This is why the per-cell value sits below 7/15.')},
    'null_B_note': ('Null B is the order-statistic value of Spearman(max, min) for random pairs, 7/15 '
                    'for any continuous data, so it asks whether pairing by condition beats random '
                    'pairing of exchangeable values, not whether two offset runs share a condition '
                    'pattern.'),
    'null_C_added_check': {'what': 'null A with each permuted pair re-sorted into max/min '
                                   '(keeps the order-statistic effect); not part of rule R2',
                           'median_over_cells_of_per_cell_null_mean': float(np.median(np.nanmean(nC, 1))),
                           'n_cells_above_own_p95': int((obs > np.nanpercentile(nC, 95, axis=1)).sum())},
    'rule_R2': {'a_median_above_null_A_p95_of_median': r2_a,
                'b_share_above_own_p95_exceeds_chance': r2_b, 'passes': shares_pattern},
}
msd = smooth['median_resid_sd']
offset_txt = (f"Offset evidence: a series built from a random member of each pair is much rougher "
              f"(median residual SD {msd['random_member']:.2f} over {len(smooth_cells)} cells) than the hi "
              f"series ({msd['hi']:.2f}), the lo series ({msd['lo']:.2f}) or the pair means "
              f"({msd['pair_mean']:.2f}), as expected if each cell holds two smooth runs at different "
              f"levels. The per-pair median hi/lo ({ratio_pooled['overall']:.2f}) is >= 1 by construction, "
              f"so it is weak evidence on its own.")
conclusion = (('Two runs with a run-level offset that share the condition pattern (rule R2 passes). '
               if shares_pattern else
               'Run-level offset, but no detectable shared condition pattern (rule R2 fails). ')
              + offset_txt + f' Shared pattern: median Spearman(hi, lo) {np.median(obs):.3f} vs null-A '
                f'95th percentile of the median {np.percentile(medA_draws, 95):.3f}; {above} of {len(obs)} '
                'cells individually exceed their own null-A 95th percentile, the rest are not individually '
                'distinguishable from null A.')

RES['two_run_structure'] = {
    'population': 'doubled_28',
    'names': sorted(NAMES[DBL].tolist()),
    'catalysts': cat_rows,
    'n_duplicated_slots': int(len(dkeys)),
    'max_measurements_per_slot': int(cnt.max()),
    'duplicated_slots_outside_doubled_28': 0,
    'n_cells_with_duplicated_slots': int(len(cells_dup)),
    'n_temperatures_with_duplicated_slots_per_catalyst': {
        str(k): int(v) for k, v in temps_per_cat.value_counts().sort_index().items()},
    'n_catalysts_duplicated_at_all_5_temperatures': int((temps_per_cat == 5).sum()),
    'duplicated_slots_per_cell_quartiles': [float(v) for v in cells_dup.quantile([0.25, 0.5, 0.75])],
    'file_order_check': {
        'share_of_917_catalysts_listed_in_non_increasing_yield_order': float(file_sorted.mean()),
        'note': 'The conditions file lists each catalyst by descending yield, so row order cannot '
                'identify the run. hi and lo are per-pair order statistics.'},
    'hi_lo_ratio': {
        'n_pairs_lo_eq_0_excluded': n_lo0,
        'n_pairs_used': int(len(Pv)),
        'pooled_median_over_pairs': ratio_pooled,
        'median_of_per_cell_medians': ratio_cellmed,
        'pooled_quantiles_over_pairs': {f'q{int(q * 100)}': float(v) for q, v in ratio_q.items()},
        'note': 'pooled = median over all pairs; per-cell = median within each cell, then median over '
                'cells (each cell counts once). Both exclude pairs with lo = 0.'},
    'smoothness_test': smooth,
    'shared_pattern_test': shared,
    'conclusion': conclusion,
}
log(f"2 two-run: {len(dkeys)} duplicated slots, {len(cells_dup)} cells, "
    f"{(temps_per_cat == 5).sum()}/28 doubled catalysts duplicated at all 5 T; lo=0 pairs {n_lo0}")
log(f"  hi/lo (pairs lo>0) pooled median over pairs {ratio_pooled['overall']:.3f}, by T "
    f"{[round(ratio_pooled[str(t)], 2) for t in TEMPS]}; median of per-cell medians "
    f"{ratio_cellmed['overall']:.3f}")
log(f"  smoothness ({len(smooth_cells)} cells >=15 pairs; median df-corrected resid SD): "
    + ', '.join(f"{k} {v:.3f}" for k, v in smooth['median_resid_sd'].items())
    + f"; ordinary complete ND cells ({len(comp_cells)}) {smooth['ordinary_complete_cells']['median_resid_sd']:.3f}")
log(f"  shared pattern ({len(sp_cells)} cells >=10 pairs): median Spearman(hi,lo) {np.median(obs):.3f}; "
    f"null A {shared['null_A']['median_over_cells_of_per_cell_null_mean']:.3f} "
    f"(p95 of across-cell median {shared['null_A']['across_cell_median_per_draw_p95']:.3f}); "
    f"cells above own null-A p95 {above}/{len(obs)} (binomial p {binom_p:.2g}); "
    f"null B {shared['null_B']['median_over_cells_of_per_cell_null_mean']:.3f}; "
    f"null C {shared['null_C_added_check']['median_over_cells_of_per_cell_null_mean']:.3f}")
log("  conclusion:", conclusion)

# ===================================================================== 3. top-decile inflation
K = int(round(0.10 * n_cat))
max_mean = sl['mean'].groupby(sl.index.values // 135).max().reindex(range(n_cat)).values
max_min = sl['min'].groupby(sl.index.values // 135).max().reindex(range(n_cat)).values
assert np.allclose(max_mean[~DBL], TRUE_MAX[~DBL]) and np.allclose(max_min[~DBL], TRUE_MAX[~DBL])
exp_k = DBL.sum() * K / n_cat
topdec = {'population': 'all_917; truth = observed max per catalyst; top decile K = 92',
          'K': K, 'n_doubled': int(DBL.sum()), 'expected_count_if_random': float(exp_k)}
for lab, tm in (('raw_max', TRUE_MAX), ('duplicated_slots_as_pair_mean', max_mean),
                ('duplicated_slots_as_pair_min', max_min)):
    order = np.argsort(-tm, kind='stable')
    top = order[:K]
    k_obs = int(DBL[top].sum())
    pct = rankdata(tm, method='average') / n_cat
    topdec[lab] = {'count_doubled_in_top_K': k_obs,
                   'hypergeom_p_X_ge_observed': float(hypergeom.sf(k_obs - 1, n_cat, DBL.sum(), K)),
                   'kth_value': float(tm[order[K - 1]]), 'k_plus_1th_value': float(tm[order[K]]),
                   'median_percentile_rank_of_doubled_28': float(np.median(pct[DBL])),
                   'percentile_rank_definition': 'average rank / 917; 1 = best'}
RES['top_decile_inflation'] = topdec
log(f"3 top decile (all_917, K={K}, expected {exp_k:.2f}): "
    + '; '.join(f"{k} {topdec[k]['count_doubled_in_top_K']} (P {topdec[k]['hypergeom_p_X_ge_observed']:.3g}, "
                f"median pct rank {topdec[k]['median_percentile_rank_of_doubled_28']:.3f})"
                for k in ('raw_max', 'duplicated_slots_as_pair_mean', 'duplicated_slots_as_pair_min')))

# ===================================================================== 4. exclusion test
V = np.full((n_cat, 5, 27), np.nan)                       # duplicated slots averaged
sk = sl.index.values
V[sk // 135, (sk % 135) // 27, sk % 27] = sl['mean'].values
MEASV = ~np.isnan(V)
CE = MEASV.any(2)                                         # observed cells
RATIO_OF_COND = g27.ch4_o2.values
in_cell = np.broadcast_to(CE[:, :, None], V.shape)
status = np.where(MEASV, 0, 1)                            # 0 present, 1 excluded
NCELL = np.where(MEASV, MEASV.sum(2, keepdims=True), np.nan)   # measured slots in the slot's cell
rng = np.random.default_rng(RNG_SEED)
W = np.stack([np.bincount(rng.integers(0, n_cat, n_cat), minlength=n_cat) for _ in range(N_BOOT)])


def shift(A, d):
    """A at temperature index ti + d (same catalyst, same condition); NaN off the grid."""
    out = np.full_like(A, np.nan)
    if d > 0:
        out[:, :-d] = A[:, d:]
    else:
        out[:, -d:] = A[:, :d]
    return out


def nanmean2(a, b):
    with np.errstate(invalid='ignore'):
        return np.nanmean(np.stack([a, b]), 0)


def neighbour_sets(M, nb_mode):
    """Neighbour scores for every slot. M: dict of metric arrays (NaN where undefined). Returns a list of
    dicts; each dict is one set of records (a slot counts once per set in which it has a score).
      avg_pm50    : mean of T-50 and T+50 where measured (primary).
      pairs_pm50  : T-50 and T+50 as two separate records (the audit's variant).
      avg_nearest : the nearest measured temperature at any distance; mean of both sides at that distance."""
    if nb_mode == 'avg_pm50':
        return [{k: nanmean2(shift(A, -1), shift(A, 1)) for k, A in M.items()}]
    if nb_mode == 'pairs_pm50':
        return [{k: shift(A, -1) for k, A in M.items()}, {k: shift(A, 1) for k, A in M.items()}]
    out = {k: np.full_like(A, np.nan) for k, A in M.items()}
    for d in range(1, 5):
        cand = {k: nanmean2(shift(A, -d), shift(A, d)) for k, A in M.items()}
        fill = np.isnan(out['mean_percentile']) & ~np.isnan(cand['mean_percentile'])
        for k in out:
            out[k][fill] = cand[k][fill]
    return [out]


def exclusion_analysis(pct_def, nb_mode='avg_pm50', complete_nb_only=False):
    """Present vs excluded slots scored by a measured neighbour of the same catalyst and condition.
    pct_def: 'centred' = (r-1)/(n-1), undefined for n = 1 (primary); 'r_over_n' = r/n.
    nb_mode: see neighbour_sets. complete_nb_only: only neighbours in cells with all 27 slots measured."""
    PCT = np.full_like(V, np.nan); TOP = np.full_like(V, np.nan); BEST = np.full_like(V, np.nan)
    n1 = 0
    for c, ti in zip(*np.where(CE)):
        m = MEASV[c, ti]; v = V[c, ti, m]; n = m.sum()
        if n < 2:
            n1 += 1
            if pct_def == 'centred':
                continue
        p = (rankdata(v) - 1) / (n - 1) if pct_def == 'centred' else rankdata(v) / n
        PCT[c, ti, m] = p; TOP[c, ti, m] = p >= 0.9 - 1e-12; BEST[c, ti, m] = v == v.max()
    M = {'mean_percentile': PCT, 'share_top_decile': TOP, 'share_cell_best': BEST,
         'mean_neighbour_cell_size': np.where(np.isnan(PCT), np.nan, NCELL)}
    if complete_nb_only:
        M = {k: np.where(NCELL == 27, A, np.nan) for k, A in M.items()}
    sets = neighbour_sets(M, nb_mode)
    has_nb = np.zeros(V.shape, bool)
    metrics = list(M)
    S_sum = np.zeros((n_cat, 2, 27, len(metrics))); S_cnt = np.zeros((n_cat, 2, 27))
    for NB in sets:
        h = ~np.isnan(NB['mean_percentile']); has_nb |= h
        for st in (0, 1):
            m = in_cell & h & (status == st)
            S_cnt[:, st] += m.sum(1)
            for j, k in enumerate(metrics):
                S_sum[:, st, :, j] += np.where(m, NB[k], 0).sum(1)
    WS = np.einsum('bc,cskm->bskm', W, S_sum); WC = np.einsum('bc,csk->bsk', W, S_cnt)
    PS, PC = S_sum.sum(0), S_cnt.sum(0)                   # point: (2, 27, m), (2, 27)

    def summarise(cmask):
        out = {'n_present': int(PC[0, cmask].sum()), 'n_excluded': int(PC[1, cmask].sum())}
        for j, k in enumerate(metrics):
            pp = PS[0, cmask, j].sum() / PC[0, cmask].sum(); pe = PS[1, cmask, j].sum() / PC[1, cmask].sum()
            bp = WS[:, 0, cmask, j].sum(1) / WC[:, 0, cmask].sum(1)
            be = WS[:, 1, cmask, j].sum(1) / WC[:, 1, cmask].sum(1)
            lo_, hi_ = np.percentile(bp - be, [2.5, 97.5])
            out[k] = {'present': float(pp), 'excluded': float(pe), 'present_minus_excluded': float(pp - pe),
                      'ci95_low': float(lo_), 'ci95_high': float(hi_),
                      'ci_excludes_zero': bool(lo_ > 0 or hi_ < 0)}
        return out

    ex = {'overall_pooled_over_ratios': summarise(np.ones(27, bool))}
    ex['overall_pooled_over_ratios']['note'] = ('Confounded by CH4:O2: excluded slots concentrate at '
                                                'ratio 6, whose within-cell percentiles are low. R4 uses '
                                                'the per-ratio rows.')
    hits = 0
    for r in (2.0, 4.0, 6.0):
        s_ = summarise(RATIO_OF_COND == r)
        d_ = s_['mean_percentile']
        s_['R4_criterion_met'] = bool(d_['present_minus_excluded'] >= 0.05 and d_['ci95_low'] > 0)
        hits += s_['R4_criterion_met']
        ex[f'ch4_o2_{int(r)}'] = s_
    # added check: condition-stratified difference (27 conditions, weighted by n excluded)
    w_ex = PC[1]
    with np.errstate(invalid='ignore', divide='ignore'):
        dpt = PS[0, :, 0] / PC[0] - PS[1, :, 0] / PC[1]
        db = WS[:, 0, :, 0] / WC[:, 0] - WS[:, 1, :, 0] / WC[:, 1]
    okc = (PC[0] > 0) & (PC[1] > 0)
    db = np.where(np.isfinite(db) & okc[None], db, np.nan)
    wb = np.where(np.isnan(db), 0, w_ex[None])
    strat_b = np.nansum(np.nan_to_num(db) * wb, 1) / wb.sum(1)
    ex['added_check_condition_stratified_mean_percentile'] = {
        'what': 'present minus excluded mean percentile within each of the 27 conditions, averaged with '
                'weights = n excluded; not part of R4',
        'present_minus_excluded': float((dpt[okc] * w_ex[okc]).sum() / w_ex[okc].sum()),
        'ci95_low': float(np.percentile(strat_b, 2.5)), 'ci95_high': float(np.percentile(strat_b, 97.5))}
    info = {'n_cells_with_one_measured_slot': n1,
            'n_slots_dropped_no_measured_neighbour': {
                'present': int((in_cell & MEASV & ~has_nb).sum()),
                'excluded': int((in_cell & ~MEASV & ~has_nb).sum())},
            'has_nb': has_nb}
    return ex, hits, PC, info


def per_ratio(ex, key='mean_percentile'):
    return {f'ch4_o2_{r}': {**{k: ex[f'ch4_o2_{r}'][key][k] for k in
                               ('present', 'excluded', 'present_minus_excluded', 'ci95_low', 'ci95_high',
                                'ci_excludes_zero')},
                            'n_present': ex[f'ch4_o2_{r}']['n_present'],
                            'n_excluded': ex[f'ch4_o2_{r}']['n_excluded']} for r in (2, 4, 6)}


excl, r4_hits, PC, ex_info = exclusion_analysis('centred')
excl_rn, r4_hits_rn, _, ex_info_rn = exclusion_analysis('r_over_n')
excl_aud, r4_hits_aud, _, ex_info_aud = exclusion_analysis('r_over_n', 'pairs_pm50')
excl_c27, r4_hits_c27, _, _ = exclusion_analysis('centred', complete_nb_only=True)
excl_nr, r4_hits_nr, _, ex_info_nr = exclusion_analysis('centred', 'avg_nearest')
coupled = r4_hits >= 2
n_dropped = ex_info['n_slots_dropped_no_measured_neighbour']
n1_cells = ex_info['n_cells_with_one_measured_slot']
n_slots_in_cells = {'present': int((in_cell & MEASV).sum()), 'excluded': int((in_cell & ~MEASV).sum())}
strat = excl['added_check_condition_stratified_mean_percentile']
strat_pt = strat['present_minus_excluded']

# scope of R4: which excluded slots were scored, and what the dropped ones look like
EXC = in_cell & ~MEASV
sc_m, dr_m = EXC & ex_info['has_nb'], EXC & ~ex_info['has_nb']
OWN_N = np.broadcast_to(MEASV.sum(2, keepdims=True), V.shape)               # measured slots, own cell
with np.errstate(invalid='ignore'):
    OWN_MEAN = np.broadcast_to(np.nanmean(V, 2, keepdims=True), V.shape)   # own-cell mean yield
CAT_MAX = np.broadcast_to(TRUE_MAX[:, None, None], V.shape)
NEVER = np.broadcast_to(~MEASV.any(1, keepdims=True), V.shape)             # condition never measured at any T
assert sc_m.sum() == excl['overall_pooled_over_ratios']['n_excluded']
scope = {
    'population': 'excluded slots (in observed cells) of all_917, split by whether R4 could score them',
    'n_excluded_slots': int(EXC.sum()), 'n_scored': int(sc_m.sum()), 'n_dropped': int(dr_m.sum()),
    'share_scored': float(sc_m.sum() / EXC.sum()),
    'n_dropped_never_measured_at_any_temperature': int((dr_m & NEVER).sum()),
    'mean_own_cell_measured_slots_scored_vs_dropped': [float(OWN_N[sc_m].mean()), float(OWN_N[dr_m].mean())],
    'mean_own_cell_mean_yield_scored_vs_dropped': [float(OWN_MEAN[sc_m].mean()), float(OWN_MEAN[dr_m].mean())],
    'mean_catalyst_observed_max_scored_vs_dropped': [float(CAT_MAX[sc_m].mean()), float(CAT_MAX[dr_m].mean())],
    'own_cell_mean_yield_note': 'mean over the cell\'s measured slots, duplicated slots averaged'}
sc = scope
scope['text'] = (
    f"Scored {sc['n_scored']:,} of {sc['n_excluded_slots']:,} excluded slots ({sc['share_scored']:.0%}). "
    f"The other {sc['n_dropped']:,} have no measured T-50 or T+50 neighbour. Compared with the scored ones they "
    f"sit in smaller cells (mean {sc['mean_own_cell_measured_slots_scored_vs_dropped'][1]:.1f} vs "
    f"{sc['mean_own_cell_measured_slots_scored_vs_dropped'][0]:.1f} measured slots), lower-yield cells (mean "
    f"cell yield {sc['mean_own_cell_mean_yield_scored_vs_dropped'][1]:.2f} vs "
    f"{sc['mean_own_cell_mean_yield_scored_vs_dropped'][0]:.2f}) and lower-max catalysts (mean observed max "
    f"{sc['mean_catalyst_observed_max_scored_vs_dropped'][1]:.2f} vs "
    f"{sc['mean_catalyst_observed_max_scored_vs_dropped'][0]:.2f}). "
    f"{sc['n_dropped_never_measured_at_any_temperature']:,} are never measured at any temperature, so no "
    f"within-catalyst test covers them. Between-catalyst coupling is reported separately (between_catalyst).")

rat_d = [excl[f'ch4_o2_{r}']['mean_percentile'] for r in (2, 4, 6)]
n_ci = sum(d['ci95_low'] > 0 for d in rat_d)
rat_c27 = per_ratio(excl_c27)
n_ci_c27 = sum(v['ci95_low'] > 0 for v in rat_c27.values())
best_rev = [r for r in (2, 4, 6) if excl[f'ch4_o2_{r}']['share_cell_best']['ci95_high'] < 0]
sz = excl['overall_pooled_over_ratios']['mean_neighbour_cell_size']
excl_detail = (
    f"Per-ratio present-minus-excluded mean-percentile differences are "
    f"{min(d['present_minus_excluded'] for d in rat_d):.3f} to "
    f"{max(d['present_minus_excluded'] for d in rat_d):.3f}, below the 0.05 threshold; "
    f"their CIs exclude zero in {n_ci} of 3 ratios. "
    + ('A small present-minus-excluded difference is detectable; part of it may be cell composition. '
       if n_ci >= 1 else 'No detectable present-minus-excluded difference. ')
    + f"One slot's percentile depends on which other slots its cell holds, and excluded slots' neighbours "
      f"sit in smaller cells (mean {sz['excluded']:.1f} vs {sz['present']:.1f} measured slots). Restricted to "
      f"neighbours in complete 27-slot cells, the differences are "
    + ', '.join(f"{v['present_minus_excluded']:+.3f} [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}]"
                for v in rat_c27.values())
    + f" for ratios 2/4/6; the CI excludes zero in {n_ci_c27} of 3 (n excluded "
    + '/'.join(f"{v['n_excluded']:,}" for v in rat_c27.values()) + "; low power). "
    + (f"The cell-best share points the other way (higher for excluded slots, CI excludes zero) at "
       f"{'ratios' if len(best_rev) > 1 else 'ratio'} {' and '.join(map(str, best_rev))}. " if best_rev else '')
    + f"Scope: {sc['share_scored']:.0%} of excluded slots were scored (see scope).")
AUDIT_EXCL = {'n_excluded': 16577, 'pooled': (0.482, 0.530),
              'ch4_o2_2': (0.634, 0.645), 'ch4_o2_4': (0.514, 0.532), 'ch4_o2_6': (0.361, 0.385)}
_a = excl_aud['overall_pooled_over_ratios']
aud_match = bool(_a['n_excluded'] == AUDIT_EXCL['n_excluded']
                 and np.allclose((_a['mean_percentile']['excluded'], _a['mean_percentile']['present']),
                                 AUDIT_EXCL['pooled'], atol=5e-4)
                 and all(np.allclose((excl_aud[r]['mean_percentile']['excluded'],
                                      excl_aud[r]['mean_percentile']['present']), AUDIT_EXCL[r], atol=5e-4)
                         for r in ('ch4_o2_2', 'ch4_o2_4', 'ch4_o2_6')))
excl_share_by_ratio = {f'ch4_o2_{int(r)}': float(PC[1, RATIO_OF_COND == r].sum() / PC[1].sum())
                       for r in (2.0, 4.0, 6.0)}

# between-catalyst coupling and coverage facts
rows_per_cat = np.bincount(groups, minlength=n_cat)
n_cells_cat = CE.sum(1)
distinct_cat = MEASV.sum((1, 2))
excl_in_cells = 27 * n_cells_cat - distinct_cat
missing_of_135 = 135 - distinct_cat
absent = ~CE
mean_by_ratio = pd.Series(y).groupby(ax.ch4_o2.values).mean()
has_r2 = pd.Series(ax.ch4_o2.values == 2).groupby(groups).any().reindex(range(n_cat)).values
cells_per_cond = MEASV.sum((0, 1))                        # distinct cells holding each condition
n_obs_cells, n_pos_cells = int(CE.sum()), n_cat * 5
pres = pd.DataFrame({'total_flow': g27.total_flow, 'ar_fraction': g27.ar_fraction, 'ch4_o2': g27.ch4_o2,
                     'n_cells': cells_per_cond,
                     'pct_of_observed_cells': 100 * cells_per_cond / n_obs_cells,
                     'pct_of_possible_cells': 100 * cells_per_cond / n_pos_cells})

RES['exclusion_test'] = {
    'population': 'all_917; slots in observed cells; duplicated slots averaged first',
    'percentile_definition': '(average rank - 1) / (n - 1) within the cell; 0 = worst, 1 = best',
    'n_cells_with_one_measured_slot_no_percentile': n1_cells,
    'n_slots_in_observed_cells': n_slots_in_cells,
    'n_slots_dropped_no_measured_neighbour': n_dropped,
    'share_of_scored_excluded_slots_by_ratio': excl_share_by_ratio,
    'comparison': excl,
    'note_shares': ('Top-decile and cell-best shares depend on cell size (a slot in a small cell is more '
                    'often its best). A cell\'s average percentile is always 0.5, but one slot\'s percentile '
                    'depends on which other slots its cell holds. Excluded slots\' neighbours sit in smaller '
                    'cells (mean_neighbour_cell_size), so part of the mean-percentile gap may be mechanical; '
                    'see added_check_complete_neighbour_cells.'),
    'rule_R4': {'n_ratios_meeting_criterion': int(r4_hits), 'needed': 2,
                'outcome_coupled_within_catalyst': coupled,
                'verdict': ('outcome-coupled within catalyst' if coupled else
                            'not outcome-coupled within catalyst by this rule'),
                'scope': scope,
                'detail': excl_detail},
    'sensitivity_percentile_r_over_n': {
        'what': ('same test with percentile = average rank / n (defined for n = 1). Its cell mean is '
                 '(n + 1) / 2n, so it rises in small cells. This alone does not reproduce the audit; see '
                 'audit_variant_pairs_r_over_n.'),
        'n_cells_with_one_measured_slot': ex_info_rn['n_cells_with_one_measured_slot'],
        'n_slots_dropped_no_measured_neighbour': ex_info_rn['n_slots_dropped_no_measured_neighbour'],
        'mean_percentile': {k: excl_rn[k]['mean_percentile'] for k in
                            ('overall_pooled_over_ratios', 'ch4_o2_2', 'ch4_o2_4', 'ch4_o2_6')},
        'n_excluded_scored': excl_rn['overall_pooled_over_ratios']['n_excluded'],
        'R4_n_ratios_meeting_criterion': int(r4_hits_rn),
        'R4_outcome_coupled_within_catalyst': bool(r4_hits_rn >= 2)},
    'audit_variant_pairs_r_over_n': {
        'what': ('added check, not part of R4: percentile = r / n, and each (slot, measured neighbour) pair '
                 'is a separate record instead of averaging T-50 and T+50. Records per slot: 1 or 2. '
                 + ('It reproduces the audit to 3 decimals: n excluded 16,577; pooled 0.482 vs 0.530; '
                    'per-ratio means. So the audit most likely counted pairs this way.' if aud_match else
                    'It does NOT reproduce the audit to 3 decimals.')),
        'reproduces_audit_to_3_decimals': aud_match,
        'n_records_present': excl_aud['overall_pooled_over_ratios']['n_present'],
        'n_records_excluded': excl_aud['overall_pooled_over_ratios']['n_excluded'],
        'mean_percentile_pooled': excl_aud['overall_pooled_over_ratios']['mean_percentile'],
        'mean_percentile_by_ratio': per_ratio(excl_aud),
        'R4_n_ratios_meeting_criterion': int(r4_hits_aud),
        'R4_outcome_coupled_within_catalyst': bool(r4_hits_aud >= 2)},
    'added_check_complete_neighbour_cells': {
        'what': ('added check, not part of R4: primary test, but only neighbours in cells with all 27 '
                 'slots measured, so every neighbour percentile is taken against the same 27 conditions. '
                 'Smaller and drawn from better-covered catalysts; low power.'),
        'mean_percentile_by_ratio': rat_c27,
        'n_ratios_ci_excludes_zero': int(n_ci_c27),
        'R4_n_ratios_meeting_criterion': int(r4_hits_c27)},
    'added_check_nearest_measured_temperature': {
        'what': ('added check, not part of R4: primary test, but the neighbour is the nearest measured '
                 'temperature at any distance (both sides averaged at that distance), to score more slots. '
                 'n_excluded_dropped is slightly larger than rule_R4.scope.n_dropped_never_measured_at_any_'
                 'temperature: it also drops slots whose only measured neighbours sit in 1-slot cells, which '
                 'have no centred percentile.'),
        'n_excluded_scored': excl_nr['overall_pooled_over_ratios']['n_excluded'],
        'n_excluded_dropped': ex_info_nr['n_slots_dropped_no_measured_neighbour']['excluded'],
        'mean_percentile_by_ratio': per_ratio(excl_nr),
        'R4_n_ratios_meeting_criterion': int(r4_hits_nr)},
    'between_catalyst': {
        'population': 'all_917; observed max = max over all rows',
        'spearman_excluded_slots_in_observed_cells_vs_observed_max':
            float(spearmanr(excl_in_cells, TRUE_MAX)[0]),
        'spearman_missing_slots_of_135_incl_absent_cells_vs_observed_max':
            float(spearmanr(missing_of_135, TRUE_MAX)[0]),
        'spearman_rows_per_catalyst_vs_observed_max': float(spearmanr(rows_per_cat, TRUE_MAX)[0]),
        'note': ('Two definitions. "Excluded slots in observed cells" matches this section\'s definition of '
                 'excluded. "Missing of 135" also counts the slots of absent cells; the audit\'s -0.287 is '
                 'this one. Quote the one that matches the wording.')},
    'absent_cells': {
        'what': '(catalyst, temperature) cells with no row; not slots of the within-catalyst test',
        'n_absent_cells': int(absent.sum()), 'n_observed_cells': n_obs_cells, 'n_possible_cells': n_pos_cells,
        'by_temperature': {str(t): int(absent[:, i].sum()) for i, t in enumerate(TEMPS)},
        'n_catalysts_with_any_absent_cell': int(absent.any(1).sum()),
        'mean_observed_max_catalysts_with_absent_cell': float(TRUE_MAX[absent.any(1)].mean()),
        'mean_observed_max_catalysts_without': float(TRUE_MAX[~absent.any(1)].mean())},
    'mean_yield_by_ch4_o2_all_rows': {f'ch4_o2_{int(k)}': float(v) for k, v in mean_by_ratio.items()},
    'catalysts_with_no_ratio_2_row': {'n': int((~has_r2).sum()),
                                      'mean_observed_max': float(TRUE_MAX[~has_r2].mean()),
                                      'mean_observed_max_all_917': float(TRUE_MAX.mean())},
    'per_condition_presence_distinct_cells': {
        'note': 'cells holding each of the 27 conditions, a duplicated slot counted once',
        'pct_of_4399_observed_cells_min_max': [float(pres.pct_of_observed_cells.min()),
                                               float(pres.pct_of_observed_cells.max())],
        'pct_of_4585_possible_cells_min_max': [float(pres.pct_of_possible_cells.min()),
                                               float(pres.pct_of_possible_cells.max())],
        'by_condition': pres.to_dict(orient='records')},
}
ov = excl['overall_pooled_over_ratios']['mean_percentile']
ov_size = excl['overall_pooled_over_ratios']['mean_neighbour_cell_size']
log(f"4 exclusion (all_917, slots in observed cells, dup averaged): scored present "
    f"{excl['overall_pooled_over_ratios']['n_present']:,}, excluded {excl['overall_pooled_over_ratios']['n_excluded']:,}; "
    f"dropped (no neighbour) {n_dropped}; 1-slot cells {n1_cells}")
log(f"  mean neighbour percentile, pooled over ratios: present {ov['present']:.3f} vs excluded "
    f"{ov['excluded']:.3f}, diff {ov['present_minus_excluded']:+.3f} "
    f"[{ov['ci95_low']:+.3f}, {ov['ci95_high']:+.3f}] (catalyst bootstrap)")
for r in (2, 4, 6):
    d = excl[f'ch4_o2_{r}']
    log(f"  CH4:O2 {r}: present {d['mean_percentile']['present']:.3f} vs excluded "
        f"{d['mean_percentile']['excluded']:.3f}, diff {d['mean_percentile']['present_minus_excluded']:+.3f} "
        f"[{d['mean_percentile']['ci95_low']:+.3f}, {d['mean_percentile']['ci95_high']:+.3f}]; "
        f"top-decile share {d['share_top_decile']['present']:.3f} vs {d['share_top_decile']['excluded']:.3f}; "
        f"cell-best share {d['share_cell_best']['present']:.3f} vs {d['share_cell_best']['excluded']:.3f}; "
        f"n excl {d['n_excluded']:,}; R4 met {d['R4_criterion_met']}")
log(f"  mean neighbour cell size: present {ov_size['present']:.2f} vs excluded {ov_size['excluded']:.2f}")
log(f"  condition-stratified diff (added check) {strat_pt:+.3f} [{strat['ci95_low']:+.3f}, "
    f"{strat['ci95_high']:+.3f}]; R4: {r4_hits}/3 ratios -> {RES['exclusion_test']['rule_R4']['verdict']}")
log("  R4 detail: " + excl_detail)
log("  R4 scope: " + scope['text'])
srn = RES['exclusion_test']['sensitivity_percentile_r_over_n']['mean_percentile']
log("  sensitivity, percentile = r/n: " + '; '.join(
    f"{k} present {v['present']:.3f} vs excluded {v['excluded']:.3f} (diff {v['present_minus_excluded']:+.3f})"
    for k, v in srn.items()) + f"; n excluded scored {excl_rn['overall_pooled_over_ratios']['n_excluded']:,}; "
    f"R4 {r4_hits_rn}/3")


def log_ratios(label, pr_, extra=''):
    log(f"  {label}: " + '; '.join(
        f"{k} present {v['present']:.3f} vs excluded {v['excluded']:.3f}, diff {v['present_minus_excluded']:+.3f} "
        f"[{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}], n excl {v['n_excluded']:,}" for k, v in pr_.items()) + extra)


au = RES['exclusion_test']['audit_variant_pairs_r_over_n']
log(f"  audit variant (r/n, T+-50 pairs not averaged): records present {au['n_records_present']:,}, excluded "
    f"{au['n_records_excluded']:,}; pooled present {au['mean_percentile_pooled']['present']:.4f} vs excluded "
    f"{au['mean_percentile_pooled']['excluded']:.4f}; reproduces audit to 3 dp: {aud_match}; R4 {r4_hits_aud}/3")
log_ratios("audit variant by ratio", au['mean_percentile_by_ratio'])
log_ratios("complete 27-slot neighbour cells only (added check)", rat_c27,
           f"; CI excludes zero in {n_ci_c27}/3; R4 {r4_hits_c27}/3")
nr = RES['exclusion_test']['added_check_nearest_measured_temperature']
log_ratios("nearest measured T at any distance (added check)", nr['mean_percentile_by_ratio'],
           f"; excluded scored {nr['n_excluded_scored']:,}, dropped {nr['n_excluded_dropped']:,}; R4 {r4_hits_nr}/3")
bc = RES['exclusion_test']['between_catalyst']
log(f"  all_917 Spearman vs observed max: excluded slots in observed cells "
    f"{bc['spearman_excluded_slots_in_observed_cells_vs_observed_max']:+.4f}; missing of 135 "
    f"{bc['spearman_missing_slots_of_135_incl_absent_cells_vs_observed_max']:+.4f}; rows "
    f"{bc['spearman_rows_per_catalyst_vs_observed_max']:+.4f}")
log(f"  absent cells {absent.sum()}; mean yield by ratio (all rows) "
    f"{[round(float(v), 3) for v in mean_by_ratio.values]}; no ratio-2 row: {(~has_r2).sum()} catalysts, "
    f"mean observed max {TRUE_MAX[~has_r2].mean():.3f}; per-condition presence "
    f"{pres.pct_of_observed_cells.min():.1f}-{pres.pct_of_observed_cells.max():.1f}% of observed cells, "
    f"{pres.pct_of_possible_cells.min():.1f}-{pres.pct_of_possible_cells.max():.1f}% of possible")

# ===================================================================== 5. precision
sd19, df19 = resid_sd(D19, Ycc)
assert df19 == 8 and df7_cc == 20
LABEL5 = ('within-session condition-to-condition scatter on complete cells; an upper bound on independent '
          'per-condition noise; NOT run-to-run repeatability')


def pooled(sd):
    """Pooled residual SD over cells with equal residual df: sqrt of the mean per-cell variance."""
    return float(np.sqrt((sd ** 2).mean()))


q = pd.qcut(cc_mean, 5, labels=False)
by_q = [{'quintile': int(k + 1), 'n_cells': int((q == k).sum()),
         'cell_mean_range': [float(cc_mean[q == k].min()), float(cc_mean[q == k].max())],
         'pooled_resid_sd_19p': pooled(sd19[q == k]), 'median_resid_sd_19p': float(np.median(sd19[q == k])),
         'pooled_resid_sd_7p': pooled(sd7_cc[q == k]), 'median_resid_sd_7p': float(np.median(sd7_cc[q == k]))}
        for k in range(5)]
in_cc = np.isin(cell_id, comp_cells)
RES['precision'] = {
    'label': LABEL5,
    'population': 'complete cells: 27 distinct conditions, ND_889 catalysts only',
    'n_cells': int(len(comp_cells)), 'n_rows': int(in_cc.sum()),
    'model_19p': {'what': 'main effects + all two-way interactions of the 3 categorical axes; '
                          '19 parameters, 8 residual df per cell',
                  'pooled_resid_sd': pooled(sd19), 'median_per_cell_resid_sd': float(np.median(sd19)),
                  'spearman_cell_sd_vs_cell_mean': float(spearmanr(sd19, cc_mean)[0])},
    'model_7p': {'what': 'main effects only; 7 parameters, 20 residual df per cell',
                 'pooled_resid_sd': pooled(sd7_cc), 'median_per_cell_resid_sd': float(np.median(sd7_cc)),
                 'spearman_cell_sd_vs_cell_mean': float(spearmanr(sd7_cc, cc_mean)[0])},
    'by_cell_mean_quintile': by_q,
    'mean_yield_complete_cell_rows': float(y[in_cc].mean()),
    'mean_yield_all_917_rows_outside_the_complete_ND_cells': float(y[~in_cc].mean()),
    'n_rows_all_917_outside_the_complete_ND_cells': int((~in_cc).sum()),
    'note_other_rows': 'every row not in these complete ND_889 cells, including all doubled_28 rows',
}
pr = RES['precision']
log(f"5 precision ({len(comp_cells)} complete ND cells; {LABEL5}): 19p pooled "
    f"{pr['model_19p']['pooled_resid_sd']:.3f}, median {pr['model_19p']['median_per_cell_resid_sd']:.3f}, "
    f"Spearman(SD, mean) {pr['model_19p']['spearman_cell_sd_vs_cell_mean']:+.3f}; 7p pooled "
    f"{pr['model_7p']['pooled_resid_sd']:.3f}, median {pr['model_7p']['median_per_cell_resid_sd']:.3f}; "
    f"mean yield complete-cell rows {pr['mean_yield_complete_cell_rows']:.3f} vs all_917 rows outside "
    f"these cells {pr['mean_yield_all_917_rows_outside_the_complete_ND_cells']:.3f} "
    f"(n {pr['n_rows_all_917_outside_the_complete_ND_cells']:,})")
log("  19p by cell-mean quintile (pooled / median): "
    + ', '.join(f"Q{b['quintile']} {b['pooled_resid_sd_19p']:.3f}/{b['median_resid_sd_19p']:.3f}" for b in by_q))


# ===================================================================== 6. floors
def floor(yv, cid):
    cm = pd.Series(yv).groupby(cid).transform('mean').values
    w = ((yv - cm) ** 2).sum(); tss = ((yv - yv.mean()) ** 2).sum()
    return {'within_cell_variance_share': float(w / tss), 'rmse_floor': float(np.sqrt(w / len(yv))),
            'n_rows': int(len(yv))}


nd_rows = ~DBL[groups]
y_avg = sl['mean'].values
cid_avg = (sk // 135) * 5 + (sk % 135) // 27
RES['floors'] = {
    'grouping': '(catalyst, temperature); floor = RMSE of predicting each row by its cell mean',
    'all_rows_all_917': floor(y, cell_id),
    'rows_ND_889': floor(y[nd_rows], cell_id[nd_rows]),
    'all_917_duplicated_slots_averaged': floor(y_avg, cid_avg),
    'note': 'Not attainable for an unseen catalyst: it assumes the cell mean is known.'}
fl = RES['floors']
log("6 floors (cell = catalyst x temperature): "
    + '; '.join(f"{k} {fl[k]['within_cell_variance_share']:.1%} / {fl[k]['rmse_floor']:.3f} (n {fl[k]['n_rows']:,})"
                for k in ('all_rows_all_917', 'rows_ND_889', 'all_917_duplicated_slots_averaged')))

# ===================================================================== 7. work-note numbers
cg = pd.DataFrame({'cell': cell_id, 'y': y}).groupby('cell').y.agg(['size', 'max'])
assert len(cg) == n_obs_cells == 4399 and absent.sum() == 186
RES['work_note_numbers'] = {
    'population': 'the 4,399 observed (catalyst, temperature) cells, all_917',
    'spearman_cell_size_rows_vs_cell_max': float(spearmanr(cg['size'], cg['max'])[0]),
    'spearman_cell_distinct_conditions_vs_cell_max': float(
        spearmanr(MEASV.sum(2)[CE], np.nanmax(np.where(MEASV, V, np.nan), 2)[CE])[0]),
    'note_distinct': 'distinct-condition version uses duplicated slots averaged; the work note quotes '
                     'the row-count version',
    'n_absent_cells': int(absent.sum()), 'n_observed_cells': n_obs_cells, 'n_possible_cells': n_pos_cells}
wn = RES['work_note_numbers']
log(f"7 work note (4,399 observed cells): Spearman(cell rows, cell max) "
    f"{wn['spearman_cell_size_rows_vs_cell_max']:+.4f}; distinct-condition version "
    f"{wn['spearman_cell_distinct_conditions_vs_cell_max']:+.4f}; absent cells {wn['n_absent_cells']}")

# ===================================================================== reference check
MC = 'Monte Carlo; moves with the RNG seed'
PRIM = 'differs by design: primary averages T-50 and T+50 and uses (r-1)/(n-1); see the audit-variant row'
AV = 'audit variant: r/n, each (slot, measured neighbour) pair a separate record'
avr = au['mean_percentile_by_ratio']
REF = [('n duplicated slots', 1906, RES['two_run_structure']['n_duplicated_slots'], ''),
       ('cells with duplicated slots', 126, RES['two_run_structure']['n_cells_with_duplicated_slots'], ''),
       ('doubled catalysts duplicated at all 5 T', 22,
        RES['two_run_structure']['n_catalysts_duplicated_at_all_5_temperatures'], ''),
       ('pairs with lo = 0', 29, n_lo0, ''),
       ('median hi/lo, pooled over pairs', 2.29, ratio_pooled['overall'], ''),
       ('cells with >= 15 pairs', 73, smooth['n_cells'], ''),
       ('resid SD median, pair means', 0.75, smooth['median_resid_sd']['pair_mean'], ''),
       ('resid SD median, hi', 1.14, smooth['median_resid_sd']['hi'], ''),
       ('resid SD median, lo', 0.62, smooth['median_resid_sd']['lo'], ''),
       ('resid SD median, ordinary complete cells', 0.72, smooth['ordinary_complete_cells']['median_resid_sd'], ''),
       ('resid SD median, random member, per-cell mean over draws', 2.50,
        smooth['median_resid_sd']['random_member'], MC + '; primary aggregator'),
       ('resid SD median, random member, per-cell median over draws', 2.50,
        smooth['random_member_alternative_aggregator']['median_resid_sd'], MC + '; probably the audit aggregator'),
       ('cells with >= 10 pairs', 85, shared['n_cells'], ''),
       ('median Spearman(hi, lo)', 0.436, shared['observed_median_over_cells'], ''),
       ('null A', -0.005, shared['null_A']['median_over_cells_of_per_cell_null_mean'], MC),
       ('null B', 0.467, shared['null_B']['median_over_cells_of_per_cell_null_mean'],
        MC + '; 7/15 = 0.467 is the large-n value, the finite-sample expectation is lower'),
       ('top decile, doubled count', 10, topdec['raw_max']['count_doubled_in_top_K'], ''),
       ('top decile, expected', 2.8, exp_k, ''),
       ('top decile, P', 1.83e-4, topdec['raw_max']['hypergeom_p_X_ge_observed'], ''),
       ('top decile, pair mean', 4, topdec['duplicated_slots_as_pair_mean']['count_doubled_in_top_K'], ''),
       ('top decile, pair min', 4, topdec['duplicated_slots_as_pair_min']['count_doubled_in_top_K'], ''),
       ('n excluded scored, primary', 16577, excl['overall_pooled_over_ratios']['n_excluded'], PRIM),
       ('excluded mean pct, pooled, primary', 0.482, ov['excluded'], PRIM),
       ('present mean pct, pooled, primary', 0.530, ov['present'], PRIM),
       ('n excluded scored, audit variant', 16577, au['n_records_excluded'], AV),
       ('excluded mean pct, pooled, audit variant', 0.482, au['mean_percentile_pooled']['excluded'], AV),
       ('present mean pct, pooled, audit variant', 0.530, au['mean_percentile_pooled']['present'], AV),
       ('excluded mean pct, ratio 2, audit variant', 0.634, avr['ch4_o2_2']['excluded'], AV),
       ('present mean pct, ratio 2, audit variant', 0.645, avr['ch4_o2_2']['present'], AV),
       ('excluded mean pct, ratio 4, audit variant', 0.514, avr['ch4_o2_4']['excluded'], AV),
       ('present mean pct, ratio 4, audit variant', 0.532, avr['ch4_o2_4']['present'], AV),
       ('excluded mean pct, ratio 6, audit variant', 0.361, avr['ch4_o2_6']['excluded'], AV),
       ('present mean pct, ratio 6, audit variant', 0.385, avr['ch4_o2_6']['present'], AV),
       ('Spearman(missing of 135, max)', -0.287,
        bc['spearman_missing_slots_of_135_incl_absent_cells_vs_observed_max'],
        'includes absent cells; excluded-in-observed-cells version stored separately'),
       ('Spearman(rows, max)', 0.2932, bc['spearman_rows_per_catalyst_vs_observed_max'], ''),
       ('complete ND cells', 890, pr['n_cells'], ''),
       ('precision pooled 19p', 0.682, pr['model_19p']['pooled_resid_sd'],
        'differs by 0.001; the verifier'"'"'s independent code matched this script; probably audit rounding'),
       ('precision median 19p', 0.402, pr['model_19p']['median_per_cell_resid_sd'], ''),
       ('floor share all', 0.199, fl['all_rows_all_917']['within_cell_variance_share'], ''),
       ('floor RMSE all', 1.757, fl['all_rows_all_917']['rmse_floor'], ''),
       ('floor share ND', 0.178, fl['rows_ND_889']['within_cell_variance_share'], ''),
       ('floor RMSE ND', 1.651, fl['rows_ND_889']['rmse_floor'], ''),
       ('floor share averaged', 0.180, fl['all_917_duplicated_slots_averaged']['within_cell_variance_share'], ''),
       ('floor RMSE averaged', 1.658, fl['all_917_duplicated_slots_averaged']['rmse_floor'],
        'fourth-decimal rounding'),
       ('Spearman(cell size, cell max)', 0.441, wn['spearman_cell_size_rows_vs_cell_max'], ''),
       ('absent cells', 186, wn['n_absent_cells'], '')]
RES['reference_check'] = {
    'what': 'this script vs values from an independent audit, for checking only',
    'rows': [{'quantity': a, 'audit': b, 'this_script': float(c), **({'note': n} if n else {})}
             for a, b, c, n in REF]}
log("reference check (audit -> this script):")
fmt = lambda v: f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.4g}"
for a, b, c, n in REF:
    log(f"    {a:58s} {fmt(b):>10s} -> {fmt(c):<10s} {n}")

RES['about']['runtime_minutes'] = (time.time() - t0) / 60
if not DEV:
    json.dump(RES, open(OUT_JSON, 'w'), indent=1, default=_json_default)
    log(f"wrote {OUT_JSON}")
else:
    log("dev mode: no file written")
